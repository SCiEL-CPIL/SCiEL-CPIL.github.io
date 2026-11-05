import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Points 0..15
    n_points = 16
    all_points = list(range(n_points))
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(all_points, 3))
    
    # Helper to check if a set of triples is linear
    # We maintain a pair_count dictionary
    def is_linear(triples, pair_count):
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    p = (min(t[i], t[j]), max(t[i], t[j]))
                    if pair_count.get(p, 0) > 1:
                        return False
        return True

    # Helper to compute independence number
    def compute_alpha(triples):
        # Convert to sets for fast subset check
        triple_sets = [set(t) for t in triples]
        
        # Check subsets from largest to smallest
        # Max possible alpha is 16
        # We can prune: if we found an independent set of size k, alpha >= k
        # We want to find the MAX independent set.
        
        # Optimization: 
        # If a set S is independent, any subset of S is independent.
        # We can use a simple backtracking or just iterate.
        # Given 16 points, iterating all subsets is 2^16 = 65536.
        # For each subset, checking independence against 35 triples is 35 * 3 checks.
        # 65536 * 100 = 6.5 million ops. Very fast in Python? Maybe a bit slow if done in pure Python loops.
        # Let's optimize.
        
        # Represent triples as bitmasks
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
            
        # Iterate subsets
        # We can iterate from 16 down to 1
        # But checking all subsets of size k is expensive if k is large.
        # However, if we find one independent set of size 4, we know alpha >= 4.
        # We want the maximum.
        
        # Better: 
        # Start with all points as a candidate independent set? No.
        # Let's just iterate all masks from 1 to 2^16 - 1.
        # Keep track of max size.
        
        max_alpha = 0
        for mask in range(1, 1 << n_points):
            # Check if mask is independent
            is_ind = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_ind = False
                    break
            if is_ind:
                # Count bits
                # bin(mask).count("1") is slow
                # Use a lookup table or just popcount
                # Python 3.10+ has int.bit_count()
                size = mask.bit_count()
                if size > max_alpha:
                    max_alpha = size
                    # If max_alpha is 16, we can break early? No, 16 is max.
                    # If max_alpha is 4, we still need to check if 5 is possible.
        
        return max_alpha

    # Initialize a random system
    def random_system():
        pair_count = {}
        triples = []
        # Shuffle triples to get random order
        shuffled_triples = all_triples[:]
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            # Check if t can be added
            valid = True
            for i in range(3):
                for j in range(i+1, 3):
                    p = (min(t[i], t[j]), max(t[i], t[j]))
                    if pair_count.get(p, 0) >= 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                triples.append(t)
                for i in range(3):
                    for j in range(i+1, 3):
                        p = (min(t[i], t[j]), max(t[i], t[j]))
                        pair_count[p] = pair_count.get(p, 0) + 1
        
        return triples, pair_count

    # Local Search
    best_triples = []
    best_alpha = 16
    
    # Start with a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        triples, pair_count = random_system()
        current_alpha = compute_alpha(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            if best_alpha == 4: # Likely optimal
                return [list(t) for t in best_triples]
        
        # Local search
        current_triples = triples
        current_pair_count = pair_count
        current_alpha = current_alpha
        
        # Precompute valid triples for addition/removal?
        # This is dynamic.
        
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            if iterations % 100 == 0 and time.time() > deadline:
                break
                
            # Pick a random triple to remove
            if not current_triples:
                break
            idx = random.randint(0, len(current_triples) - 1)
            remove_t = current_triples[idx]
            
            # Remove it
            current_triples.pop(idx)
            for i in range(3):
                for j in range(i+1, 3):
                    p = (min(remove_t[i], remove_t[j]), max(remove_t[i], remove_t[j]))
                    current_pair_count[p] -= 1
                    if current_pair_count[p] == 0:
                        del current_pair_count[p]
            
            # Pick a random triple to add
            # We need to find a triple that doesn't conflict
            # To be efficient, we can pick a random triple from all_triples and check
            # Or we can pick a pair that is now free, and find a third point.
            
            # Let's pick a random free pair
            free_pairs = [p for p, c in current_pair_count.items() if c < 1] 
            # This is wrong. current_pair_count only has pairs with count >= 1?
            # No, I deleted pairs with count 0.
            # So I need to know which pairs are free.
            # Total pairs = 120.
            # Used pairs = keys of current_pair_count.
            # Free pairs = all_pairs - used_pairs.
            
            all_pairs = list(itertools.combinations(range(16), 2))
            used_pairs_set = set(current_pair_count.keys())
            free_pairs = [p for p in all_pairs if p not in used_pairs_set]
            
            if not free_pairs:
                # Can't add any triple? Put back removed triple
                current_triples.append(remove_t)
                for i in range(3):
                    for j in range(i+1, 3):
                        p = (min(remove_t[i], remove_t[j]), max(remove_t[i], remove_t[j]))
                        current_pair_count[p] = current_pair_count.get(p, 0) + 1
                continue
            
            p = random.choice(free_pairs)
            u, v = p
            # Find a w such that (u,v,w) is a valid triple
            # w must not form a conflict with u or v
            # Pairs (u,w) and (v,w) must be free.
            # And (u,v) is free (already checked).
            
            candidates_w = []
            for w in range(16):
                if w == u or w == v:
                    continue
                p_uw = (min(u, w), max(u, w))
                p_vw = (min(v, w), max(v, w))
                if p_uw not in used_pairs_set and p_vw not in used_pairs_set:
                    candidates_w.append(w)
            
            if not candidates_w:
                # Put back
                current_triples.append(remove_t)
                for i in range(3):
                    for j in range(i+1, 3):
                        p = (min(remove_t[i], remove_t[j]), max(remove_t[i], remove_t[j]))
                        current_pair_count[p] = current_pair_count.get(p, 0) + 1
                continue
            
            w = random.choice(candidates_w)
            add_t = tuple(sorted([u, v, w]))
            
            # Add it
            current_triples.append(add_t)
            for i in range(3):
                for j in range(i+1, 3):
                    p = (min(add_t[i], add_t[j]), max(add_t[i], add_t[j]))
                    current_pair_count[p] = current_pair_count.get(p, 0) + 1
            
            new_alpha = compute_alpha(current_triples)
            
            # Accept if better or equal (with probability)
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    if best_alpha == 4:
                        return [list(t) for t in best_triples]
            else:
                # Reject: revert
                current_triples.pop()
                for i in range(3):
                    for j in range(i+1, 3):
                        p = (min(add_t[i], add_t[j]), max(add_t[i], add_t[j]))
                        current_pair_count[p] -= 1
                        if current_pair_count[p] == 0:
                            del current_pair_count[p]
                current_triples.insert(idx, remove_t)
                for i in range(3):
                    for j in range(i+1, 3):
                        p = (min(remove_t[i], remove_t[j]), max(remove_t[i], remove_t[j]))
                        current_pair_count[p] = current_pair_count.get(p, 0) + 1

    return [list(t) for t in best_triples]
