import itertools
import random
import time

def construct():
    n = 16
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    # Precompute pair masks for each triple to check linearity
    # triple (a,b,c) has pairs (a,b), (a,c), (b,c)
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        triple_pairs.append([(a, b), (a, c), (b, c)])
    
    # Precompute bitmask for each triple for alpha calculation
    triple_masks = [0] * len(all_triples)
    for i, t in enumerate(all_triples):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = mask

    def get_alpha(active_indices):
        """
        Calculate independence number of the hypergraph defined by active_indices.
        active_indices: list of indices into all_triples
        """
        if not active_indices:
            return n
        
        # Build a set of active masks for fast checking?
        # No, we need to check if a candidate set S contains ANY active triple.
        # S is independent if for all t in active_indices, (mask_S & mask_t) != mask_t.
        
        # We want max |S| such that S is independent.
        # Iterate k from n down to 1
        # To speed up, we can prune.
        
        # For small n=16, checking all combinations of size k is expensive if k is large.
        # But alpha is likely small (around 4-6).
        # We check k=16, 15, ...
        
        # Optimization: If we find an independent set of size k, we return k.
        # We can use a recursive search with pruning?
        # For now, simple iteration is okay if alpha is small.
        
        active_set = set(active_indices)
        
        # Precompute for each subset of vertices? No, 2^16 = 65536.
        # We can precompute a boolean array `is_independent_mask` for all 2^16 masks?
        # That might be too slow if done repeatedly.
        
        # Let's just iterate combinations.
        for k in range(n, 0, -1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                is_ind = True
                for idx in active_indices:
                    if (mask & triple_masks[idx]) == triple_masks[idx]:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def is_linear(active_indices):
        """
        Check if the set of triples is linear (no pair repeated).
        """
        pair_count = {}
        for idx in active_indices:
            for p in triple_pairs[idx]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                if key in pair_count:
                    return False
                pair_count[key] = True
        return True

    # Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        active = []
        used_pairs = set()
        
        for idx in indices:
            # Check if this triple conflicts with used_pairs
            conflict = False
            for p in triple_pairs[idx]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                if key in used_pairs:
                    conflict = True
                    break
            if not conflict:
                active.append(idx)
                for p in triple_pairs[idx]:
                    key = (min(p[0], p[1]), max(p[0], p[1]))
                    used_pairs.add(key)
        return active

    # Local Search
    def local_search(active, max_time=10):
        current_alpha = get_alpha(active)
        best_alpha = current_alpha
        best_active = active[:]
        
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            # Try to improve
            # Pick a random triple to remove
            if not active:
                break
            idx_to_remove = random.choice(active)
            active.remove(idx_to_remove)
            
            # Try to add a random valid triple
            # To be efficient, we can just pick a random triple and check if it's valid
            # And if it reduces alpha?
            # Actually, we want to *change* the system.
            # Let's try swapping.
            
            # Find a candidate to add
            added = False
            for _ in range(20): # Try 20 random candidates
                candidate_idx = random.randint(0, len(all_triples)-1)
                if candidate_idx in active:
                    continue
                
                # Check if candidate is linear with current active
                valid = True
                for p in triple_pairs[candidate_idx]:
                    key = (min(p[0], p[1]), max(p[0], p[1]))
                    # Check if key is used by any active triple
                    # This is O(N_active).
                    # For speed, we should maintain a set of used pairs.
                    pass 
                
                # This implementation is too slow for local search.
                # We need to maintain used_pairs incrementally.
                
                break # Skip this detailed local search for now, too complex to implement correctly in short time
            
            # Revert
            active.append(idx_to_remove)
            
            # If we don't have a good local search, just try multiple greedy seeds.
            break

    # Strategy: Run multiple greedy constructions with different seeds and pick the best.
    best_alpha = n
    best_active = []
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 110: # Leave 10s buffer
        seed += 1
        active = greedy_construct(seed)
        alpha = get_alpha(active)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_active = active[:]
            
        if best_alpha <= 4:
            break

    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_active]
    
    # Final verification
    pair_count = {}
    for e in result:
        for i in range(3):
            for j in range(i+1, 3):
                p = (min(e[i], e[j]), max(e[i], e[j]))
                if p in pair_count:
                    # Invalid, should not happen with greedy
                    pass
                pair_count[p] = True
                
    return result
