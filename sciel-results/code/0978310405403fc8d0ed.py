import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # State representation for fast incremental checks
    # selected: list of ints
    # triple_sums: set of sums of all triples
    # pair_sums_mod: dict {a: set of (b+c)%a for b,c in selected\{a}}
    
    def build_state(selected):
        """Rebuild state from scratch. O(N^3)."""
        n = len(selected)
        triple_sums = set()
        pair_sums_mod = {a: set() for a in selected}
        
        # Precompute pair sums for faster triple generation?
        # Just iterate.
        for i in range(n):
            a = selected[i]
            for j in range(i+1, n):
                b = selected[j]
                for k in range(j+1, n):
                    c = selected[k]
                    triple_sums.add(a+b+c)
        
        # Build pair_sums_mod
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i: continue
                b = selected[j]
                for k in range(j+1, n):
                    if k == i: continue
                    c = selected[k]
                    pair_sums_mod[a].add((b+c) % a)
                    
        return triple_sums, pair_sums_mod

    def can_add(selected, candidate, triple_sums, pair_sums_mod):
        """Check if candidate can be added. O(N^2 + |triple_sums|)."""
        # Check 1: No triple sum divisible by candidate
        for s in triple_sums:
            if s % candidate == 0:
                return False
        
        # Check 2: For each a in selected, no pair sum (b+c) from selected\{a}
        #          has (b+c+candidate) % a == 0
        #          => (b+c) % a == (-candidate) % a
        for a in selected:
            target = (-candidate) % a
            if target in pair_sums_mod[a]:
                return False
        
        return True

    def add_element(selected, candidate, triple_sums, pair_sums_mod):
        """Add candidate to state. O(N^2)."""
        n = len(selected)
        
        # Update triple_sums: add triples (candidate, b, c)
        for i in range(n):
            b = selected[i]
            for j in range(i+1, n):
                c = selected[j]
                triple_sums.add(candidate + b + c)
                
        # Update pair_sums_mod for existing a
        for a in selected:
            for b in selected:
                if a == b: continue
                pair_sums_mod[a].add((candidate + b) % a)
                
        # Initialize pair_sums_mod for candidate
        new_pairs = set()
        for i in range(n):
            b = selected[i]
            for j in range(i+1, n):
                c = selected[j]
                new_pairs.add((b+c) % candidate)
        pair_sums_mod[candidate] = new_pairs
        
        selected.append(candidate)

    # Main Search
    best_set = []
    best_len = 0
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline - 10:
            break
            
        random.seed(12345 + trial)
        order = list(range(192, 0, -1))
        # Perturb order to escape local minima
        if trial > 0:
            random.shuffle(order)
            # Sort with noise to keep large numbers somewhat prioritized
            order.sort(key=lambda x: -x + random.uniform(-20, 20))
            
        selected = []
        triple_sums = set()
        pair_sums_mod = {}
        
        for x in order:
            if len(selected) >= 48:
                break
            if can_add(selected, x, triple_sums, pair_sums_mod):
                add_element(selected, x, triple_sums, pair_sums_mod)
                
        if len(selected) > best_len:
            best_set = selected[:]
            best_len = len(selected)
            
        if best_len >= 48:
            # Try to see if we can get 49?
            # But we only return 48.
            # If we can't get 48, we stop.
            break
            
    # Local Search: 1-out-2-in swaps
    if best_len < 48 and time.time() < deadline - 5:
        current = best_set[:]
        current_len = best_len
        improved = True
        while improved and current_len < 48 and time.time() < deadline - 5:
            improved = False
            # Rebuild state for current
            triple_sums, pair_sums_mod = build_state(current)
            
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                # Remove a
                current_without_a = [v for j, v in enumerate(current) if j != i]
                # Rebuild state for current_without_a
                ts_wo_a, psm_wo_a = build_state(current_without_a)
                
                # Try to add b
                for b in range(192, 0, -1):
                    if b in current_without_a:
                        continue
                    if not can_add(current_without_a, b, ts_wo_a, psm_wo_a):
                        continue
                    
                    # Tentatively add b
                    cur_b = current_without_a + [b]
                    ts_b, psm_b = build_state(cur_b)
                    
                    # Try to add c
                    for c in range(192, 0, -1):
                        if c in cur_b:
                            continue
                        if can_add(cur_b, c, ts_b, psm_b):
                            new_set = cur_b + [c]
                            if len(new_set) > current_len:
                                current = new_set
                                current_len = len(current)
                                best_set = current[:]
                                best_len = current_len
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if current_len >= 48:
                break

    return best_set[:48]

# EVOLVE-BLOCK-END
