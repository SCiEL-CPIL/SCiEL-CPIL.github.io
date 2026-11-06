"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
import sys

def construct():
    """
    Construct a linear triple system on 16 points minimizing the independence number.
    
    Strategy:
    1. Start with a known good configuration (e.g., from simulated annealing or greedy).
    2. Use local search / simulated annealing to minimize the independence number.
    3. The independence number is the largest subset of points containing no triple.
    
    We use a time-bounded search with a deadline guard.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    random.seed(42)
    
    def get_pairs(triple):
        a, b, c = triple
        return frozenset({(a, b), (a, c), (b, c)})
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        pair_count = {}
        for t in triples:
            for p in get_pairs(t):
                if p in pair_count:
                    return False
                pair_count[p] = 1
        return True
    
    def independence_number(triples):
        """
        Find the largest subset of points that contains no triple.
        Use a more efficient approach: for each subset size from n down,
        check if any subset of that size is triple-free.
        """
        triple_set = set(frozenset(t) for t in triples)
        
        def has_triple_in_subset(subset):
            s = frozenset(subset)
            # Check all triples within this subset
            for t in triple_set:
                if t.issubset(s):
                    return True
            return False
        
        # Try from largest subset size down
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                if not has_triple_in_subset(subset):
                    # Found a triple-free subset of this size
                    return size
        return 0
    
    def independence_number_fast(triples):
        """
        Faster independence number computation using bitmask.
        Represent each triple as a bitmask, and check subsets via bitmask operations.
        """
        n = 16
        # Map each point to a bit position
        # A triple-free subset S means no triple is entirely contained in S
        # i.e., for every triple t, t & S != t (in bitmask terms: not (t_mask & S == t_mask))
        
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Try subset sizes from n down to 1
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                # Check if any triple is fully contained in s_mask
                triple_free = True
                for tm in triple_masks:
                    if (tm & s_mask) == tm:
                        triple_free = False
                        break
                if triple_free:
                    return size
        return 0
    
    def independence_number_bitmask(triples):
        """
        Even faster: precompute triple masks, then for each candidate subset,
        check containment quickly.
        """
        n = 16
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Sort triples by mask for potential early termination
        # Try subset sizes from n down
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                # Check if any triple is fully contained
                contained = False
                for tm in triple_masks:
                    if (tm & s_mask) == tm:
                        contained = True
                        break
                if not contained:
                    return size
        return 0
    
    def greedy_construct():
        """Greedy construction of a linear triple system."""
        triples = []
        used_pairs = set()
        
        # Shuffle all triples for randomization
        candidates = list(all_triples)
        random.shuffle(candidates)
        
        for t in candidates:
            pairs = get_pairs(t)
            if not any(p in used_pairs for p in pairs):
                triples.append(list(t))
                used_pairs.update(pairs)
        
        return triples
    
    def local_search(triples, max_iters=50000):
        """
        Local search to minimize independence number.
        Moves: replace a triple with another that maintains linearity.
        """
        current = [list(t) for t in triples]
        current_pairs = set()
        for t in current:
            current_pairs.update(get_pairs(t))
        
        current_alpha = independence_number_bitmask(current)
        best = [list(t) for t in current]
        best_alpha = current_alpha
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random triple to replace
            idx = random.randint(0, len(current) - 1)
            old_triple = current[idx]
            old_pairs = get_pairs(old_triple)
            
            # Remove old triple's pairs
            for p in old_pairs:
                current_pairs.discard(p)
            
            # Try to find a replacement triple that doesn't conflict
            replacement_found = False
            attempts = 0
            while attempts < 200:
                new_triple = random.choice(all_triples)
                new_pairs = get_pairs(new_triple)
                if not any(p in current_pairs for p in new_pairs):
                    current[idx] = list(new_triple)
                    current_pairs.update(new_pairs)
                    replacement_found = True
                    break
                attempts += 1
            
            if not replacement_found:
                # Restore old triple
                current[idx] = old_triple
                current_pairs.update(old_pairs)
                continue
            
            new_alpha = independence_number_bitmask(current)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best = [list(t) for t in current]
            else:
                # Revert with some probability (simple hill climbing)
                # For now, just revert if not better
                current[idx] = old_triple
                current_pairs.discard(new_pairs if replacement_found else old_pairs)
                for p in old_pairs:
                    current_pairs.add(p)
        
        return best, best_alpha
    
    # Start with greedy construction
    initial = greedy_construct()
    # Ensure we have a reasonable number of triples
    # The maximum number of triples in a linear 3-uniform hypergraph on 16 points
    # is bounded by floor(16 * 15 / 6) = 40 (each triple uses 3 pairs, total pairs = 120)
    
    # Try multiple restarts
    best_triples = None
    best_alpha = n  # Start with worst case
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        triples = greedy_construct()
        # Local search on this configuration
        improved, alpha = local_search(triples, max_iters=20000)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved
        
        if best_alpha <= 4:
            break  # Can't do much better than 4 likely
    
    if best_triples is None:
        # Fallback to initial
        best_triples = initial
    
    return best_triples
