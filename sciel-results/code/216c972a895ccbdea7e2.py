import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_selected = []
    best_size = 0
    
    def _is_valid_full(selected):
        """Full validation of a set."""
        n = len(selected)
        if n < 4:
            return True
        for a in selected:
            others = [v for v in selected if v != a]
            m = len(others)
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        if (others[i] + others[j] + others[k]) % a == 0:
                            return False
        return True

    def update_best(sel):
        nonlocal best_selected, best_size
        if len(sel) > best_size:
            if _is_valid_full(sel):
                best_selected = sel[:]
                best_size = len(best_selected)
                
    def greedy_search(order):
        """
        Greedy search with incremental validity checks.
        """
        selected = []
        all_pairs = set()
        all_triples = set()
        pair_sums_excl = {} # a -> set of pair sums in selected\{a}
        
        for x in order:
            # Check 1: Does x divide any existing triple sum?
            valid = True
            for s in all_triples:
                if s % x == 0:
                    valid = False
                    break
            
            if valid:
                # Check 2: For each a in selected, does (-x) % a exist in pair_sums_excl[a]?
                for a in selected:
                    target = (-x) % a
                    if target in pair_sums_excl[a]:
                        valid = False
                        break
            
            if valid:
                # Add x to selected
                selected.append(x)
                
                # Update structures
                old_selected = selected[:-1]
                n_old = len(old_selected)
                
                # Store old all_pairs for x's exclusion set
                new_excl_for_x = set(all_pairs)
                
                # Generate new pairs involving x
                # We need to update all_pairs and pair_sums_excl
                # New pairs are (x, b) for b in old_selected
                # New triples are (x, b, c) for b, c in old_selected, b != c
                
                # 1. Update all_pairs and pair_sums_excl for existing a
                # For each b in old_selected:
                #   new_pair_sum = x + b
                #   all_pairs.add(new_pair_sum)
                #   For each a in old_selected:
                #     if a != b:
                #       pair_sums_excl[a].add(new_pair_sum)
                
                for b in old_selected:
                    new_pair_sum = x + b
                    all_pairs.add(new_pair_sum)
                    for a in old_selected:
                        if a != b:
                            pair_sums_excl[a].add(new_pair_sum)
                            
                # 2. Update all_triples
                # For each pair (b, c) in old_selected:
                #   all_triples.add(x + b + c)
                
                for i in range(n_old):
                    b = old_selected[i]
                    for j in range(i+1, n_old):
                        c = old_selected[j]
                        all_triples.add(x + b + c)
                        
                # 3. Add x to pair_sums_excl
                pair_sums_excl[x] = new_excl_for_x
                
        return selected

    # Strategy 1: Descending
    selected = greedy_search(range(192, 0, -1))
    update_best(selected)
    
    # Strategy 2: Ascending
    selected = greedy_search(range(1, 193))
    update_best(selected)
    
    # Strategy 3: Randomized
    # Run random permutations until time runs out
    while time.time() < deadline - 2:
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        selected = greedy_search(candidates)
        update_best(selected)
        
        if best_size == 48:
            break
            
    return best_selected
