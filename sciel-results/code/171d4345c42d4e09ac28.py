import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Points 0..15
    n_points = 16
    
    # Initial construction: STS(15) on points 0..14
    # PG(3,2) has 15 points and 35 lines (triples).
    # We add point 15 and try to integrate it.
    
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    initial_triples = []
    for i in range(15):
        for j in range(i+1, 15):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            initial_triples.append(tuple(sorted([i, j, k])))
            
    # Remove duplicates (should be none in STS)
    initial_triples = list(set(initial_triples))
    
    # Try to add triples involving point 15
    # A point in STS(15) is in 7 triples.
    # Point 15 is currently isolated.
    # We can add triples (a, b, 15) if (a, b) is not a pair in an existing triple?
    # No, linearity requires that (a, b, 15) shares at most 1 point with any existing triple.
    # If we add (a, b, 15), it shares {a, b} with any triple containing pair {a, b}.
    # In STS(15), every pair {a, b} is in exactly one triple.
    # So (a, b, 15) will share 2 points with the unique triple containing a and b.
    # Thus, we cannot simply add (a, b, 15) without removing that triple.
    
    # Strategy: Start with STS(15). 
    # Randomly pick a triple T0 from STS(15). Remove it.
    # Add triples involving 15 that use pairs from T0?
    # If T0 = {a, b, c}, the pairs are {a,b}, {b,c}, {a,c}.
    # If we remove T0, the pairs {a,b}, {b,c}, {a,c} are "free" (no longer covered by a triple).
    # We can add (a, b, 15), (b, c, 15), (a, c, 15).
    # These 3 new triples share:
    # (a,b,15) and (b,c,15) share {b,15} -> 2 points. Conflict!
    # So we can't add all 3. We can add at most 1 or 2 if we are careful?
    # (a,b,15) and (a,c,15) share {a,15} -> 2 points. Conflict.
    # Any two triples (x, y, 15) and (x, z, 15) share {x, 15}.
    # So all triples involving 15 must be disjoint in their other two points?
    # No, they share 15. So they share 1 point (15) and potentially others.
    # If T1 = (a, b, 15) and T2 = (c, d, 15), they share {15} if {a,b} != {c,d} and no overlap in a,b,c,d.
    # If they share a point a, e.g., T1=(a,b,15), T2=(a,c,15), they share {a,15} -> 2 points. Invalid.
    # So, any two triples containing 15 must be disjoint in their other two points.
    # Max number of such triples is floor((15)/2) = 7.
    
    # Let's try to build a system with 15 well-integrated.
    # Random restarts might be better.
    
    def compute_alpha_and_set(triples):
        """
        Returns (alpha, max_independent_set)
        """
        # Precompute triple masks
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)
        
        # Check from k=16 down to 1
        for k in range(n_points, 0, -1):
            found = False
            best_set = None
            # Iterate through combinations
            for combo in itertools.combinations(range(n_points), k):
                combo_mask = 0
                for p in combo:
                    combo_mask |= (1 << p)
                
                # Check if any triple is subset of combo
                is_independent = True
                for tm in t_masks:
                    if (combo_mask & tm) == tm:
                        is_independent = False
                        break
                
                if is_independent:
                    found = True
                    best_set = set(combo)
                    break
            
            if found:
                return k, best_set
        
        return 1, set([0])

    def is_linear_with(triple, others):
        """Check if triple is linear with all triples in others."""
        t_set = set(triple)
        for t in others:
            if len(t_set & set(t)) > 1:
                return False
        return True

    best_alpha = n_points
    best_triples = []
    best_set = None
    
    # Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Generate a random valid packing
        # Start with empty, add random triples?
        # Or start with STS(15) and modify.
        
        current_triples = []
        
        # Try to add triples greedily
        # Sort all possible triples
        all_triples = list(itertools.combinations(range(n_points), 3))
        random.shuffle(all_triples)
        
        # Heuristic: prioritize triples that cover "isolated" points?
        # For now, just random greedy.
        
        for t in all_triples:
            if time.time() > deadline - 5:
                break
            if is_linear_with(t, current_triples):
                current_triples.append(list(t))
                
        # Local Search / Targeted Move Phase
        current_alpha, current_set = compute_alpha_and_set(current_triples)
        
        # Try to improve
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_alpha, current_set = compute_alpha_and_set(current_triples)
            
            # Generate candidates from current_set
            # If current_set is large, we have many candidates.
            # We want to add a triple T subset of current_set.
            candidates = list(itertools.combinations(sorted(current_set), 3))
            random.shuffle(candidates)
            
            added = False
            for cand in candidates:
                cand = list(cand)
                # Check linearity
                if is_linear_with(cand, current_triples):
                    # Add it
                    current_triples.append(cand)
                    added = True
                    improved = True
                    break
            
            if not added:
                # Try a random perturbation: remove a random triple and try to add from new MIS
                if current_triples and random.random() < 0.1:
                    idx = random.randint(0, len(current_triples)-1)
                    current_triples.pop(idx)
                    improved = True # Force re-evaluation
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current_triples[:]
            best_set = current_set
            
        # If we reached alpha=5, it's likely optimal or near optimal.
        if best_alpha <= 5:
            break

    return best_triples
