"""Linear triple systems on 16 points - minimize independence number.
Structural change: use bitmask-based representation for fast independence
number computation, combined with simulated annealing over maximal packings.
"""
import itertools
import random
import time
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 16
    all_points = list(range(n))
    
    # Precompute all possible triples as sorted tuples
    all_triples = list(itertools.combinations(range(n), 3))
    all_triples_set = set(all_triples)
    
    # For each triple, precompute its 3 pairs as frozenset
    def triple_pairs(t):
        a, b, c = t
        return frozenset({(min(a,b), max(a,b)), (min(a,c), max(a,c)), (min(b,c), max(b,c))})
    
    # Precompute pair membership: for each pair, which triples contain it
    pair_to_triples = {}
    for t in all_triples:
        for p in triple_pairs(t):
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(t)
    
    # For fast independence number: represent each triple as a bitmask of 3 bits
    # A subset S (bitmask) contains triple t iff (S & mask_t) == mask_t
    triple_masks = [0] * len(all_triples)
    for i, t in enumerate(all_triples):
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[i] = m
    
    # Precompute for each pair, the set of triple indices containing that pair
    pair_to_triple_indices = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs(t):
            if p not in pair_to_triple_indices:
                pair_to_triple_indices[p] = []
            pair_to_triple_indices[p].append(i)
    
    def compute_alpha_fast(triple_indices):
        """Compute independence number using bitmask + branch and bound.
        triple_indices: list of indices into all_triples"""
        if not triple_indices:
            return n
        
        # Build a set of triple masks for this system
        masks = [triple_masks[i] for i in triple_indices]
        
        # Upper bound: try to find if there's an independent set of size k
        # Use recursive search with pruning
        
        best = [0]
        
        # Sort masks by popcount (all are 3, so no help) 
        # Instead, use a different strategy: 
        # For each subset of size k from n down, check if it's independent
        # But that's too slow. Use branch and bound.
        
        # Better approach: use the fact that n=16 is small
        # Enumerate subsets in decreasing size order, but with pruning
        
        # Actually, for n=16, the number of subsets of size 8 is C(16,8)=12870
        # For size 7: C(16,7)=11440, etc.
        # We can check each subset by testing against all triples
        
        # But we need to do this many times in local search, so need it fast.
        
        # Alternative: precompute for each point, which triples contain it
        # Then for a subset S, a triple is in S iff all 3 points are in S
        
        # Fast check: for a given subset bitmask S, check if any triple is fully contained
        def is_independent(S_mask):
            for m in masks:
                if (S_mask & m) == m:
                    return False
            return True
        
        # Try sizes from high to low
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                S_mask = 0
                for p in subset:
                    S_mask |= (1 << p)
                if is_independent(S_mask):
                    found = True
                    break
            if found:
                best[0] = size
                break
            if time.time() > deadline - 2:
                break
        return best[0]
    
    def compute_alpha_fast2(triple_indices):
        """Faster alpha computation using precomputed point-to-triple mapping."""
        if not triple_indices:
            return n
        
        # For each point, list of triple indices containing it
        point_to_triples = [[] for _ in range(n)]
        for idx in triple_indices:
            t = all_triples[idx]
            for p in t:
                point_to_triples[p].append(idx)
        
        masks = [triple_masks[i] for i in triple_indices]
        num_triples = len(masks)
        
        def is_independent(S_mask):
            for m in masks:
                if (S_mask & m) == m:
                    return False
            return True
        
        # Binary search or linear search from top
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                S_mask = 0
                for p in subset:
                    S_mask |= (1 << p)
                if is_independent(S_mask):
                    found = True
                    break
            if found:
                return size
        return 1
    
    def greedy_maximal_packing():
        """Build a maximal linear triple packing greedily."""
        order = list(range(len(all_triples)))
        random.shuffle(order)
        used_pairs = set()
        selected = []
        for idx in order:
            t = all_triples[idx]
            pairs = triple_pairs(t)
            if not any(p in used_pairs for p in pairs):
                selected.append(idx)
                used_pairs.update(pairs)
        return selected
    
    def is_linear(triple_indices):
        """Check linearity."""
        used_pairs = set()
        for idx in triple_indices:
            t = all_triples[idx]
            pairs = triple_pairs(t)
            for p in pairs:
                if p in used_pairs:
                    return False
                used_pairs.add(p)
        return True
    
    def local_search(triple_indices, max_iters=2000):
        """Local search: try swapping one triple for another."""
        if not is_linear(triple_indices):
            return triple_indices, n
        
        current_alpha = compute_alpha_fast2(triple_indices)
        current = list(triple_indices)
        
        # Precompute for each triple in current, which other triples conflict
        # For efficiency, track used pairs
        used_pairs = set()
        for idx in current:
            used_pairs.update(triple_pairs(all_triples[idx]))
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try random swaps
            for _ in range(50):
                i = random.randint(0, len(current) - 1)
                old_idx = current[i]
                old_pairs = triple_pairs(all_triples[old_idx])
                
                # Find a random replacement
                new_idx = random.randint(0, len(all_triples) - 1)
                if new_idx == old_idx:
                    continue
                new_pairs = triple_pairs(all_triples[new_idx])
                
                # Check if new triple conflicts with any other triple in current
                can_replace = True
                for p in new_pairs:
                    if p in used_pairs and p not in old_pairs:
                        can_replace = False
                        break
                    if p in old_pairs:
                        continue
                    if p in used_pairs:
                        can_replace = False
                        break
                
                if not can_replace:
                    continue
                
                # Perform swap
                current[i] = new_idx
                new_alpha = compute_alpha_fast2(current)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    improved = True
                    # Update used_pairs
                    used_pairs.difference_update(old_pairs)
                    used_pairs.update(new_pairs)
                    break
                else:
                    current[i] = old_idx
            
            if not improved:
                # Try a different strategy: remove a random triple and add a random one
                if len(current) > 1:
                    i = random.randint(0, len(current) - 1)
                    old_idx = current[i]
                    old_pairs = triple_pairs(all_triples[old_idx])
                    
                    new_idx = random.randint(0, len(all_triples) - 1)
                    new_pairs = triple_pairs(all_triples[new_idx])
                    
                    # Check conflicts with remaining triples
                    can_add = True
                    remaining_pairs = set()
                    for j, idx in enumerate(current):
                        if j == i:
                            continue
                        remaining_pairs.update(triple_pairs(all_triples[idx]))
                    
                    for p in new_pairs:
                        if p in remaining_pairs:
                            can_add = False
                            break
                    
                    if can_add:
                        current[i] = new_idx
                        new_alpha = compute_alpha_fast2(current)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            improved = True
                            used_pairs.difference_update(old_pairs)
                            used_pairs.update(new_pairs)
        
        return current, current_alpha
    
    # Main: multiple restarts with greedy + local search
    best_triples = None
    best_alpha = n
    
    # First, try a structured construction based on Z_16 or similar
    # Use a known good starting point if possible
    
    # Try several random greedy constructions
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        triples = greedy_maximal_packing()
        if not triples:
            continue
        
        triples, alpha = local_search(triples, max_iters=500)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # More intensive search on best
    if best_triples is not None and time.time() < deadline - 10:
        best_triples, best_alpha = local_search(best_triples, max_iters=5000)
    
    # Try to add more triples to reduce alpha
    if best_triples is not None and time.time() < deadline - 5:
        current = list(best_triples)
        used_pairs = set()
        for idx in current:
            used_pairs.update(triple_pairs(all_triples[idx]))
        
        # Try adding triples
        candidates = list(range(len(all_triples)))
        random.shuffle(candidates)
        for idx in candidates:
            if time.time() > deadline - 3:
                break
            pairs = triple_pairs(all_triples[idx])
            if any(p in used_pairs for p in pairs):
                continue
            test = current + [idx]
            new_alpha = compute_alpha_fast2(test)
            if new_alpha <= best_alpha:
                current.append(idx)
                used_pairs.update(pairs)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = current
    
    if best_triples is None:
        # Fallback
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    # Convert indices back to triples
    return [list(all_triples[idx]) for idx in best_triples]
