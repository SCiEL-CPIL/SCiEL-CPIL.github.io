"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    # Baseline: Weight 2 vectors
    baseline_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                baseline_vectors.append(v)
    baseline_count = len(baseline_vectors)
    
    # Strategy: Greedy selection of weight-4 vectors.
    # We select at most 1 vector per support to simplify constraints.
    # Constraint: For any two selected supports S, T with |S \cap T| = 3,
    # the sign projections on S \cap T must be different.
    
    supports = list(itertools.combinations(range(D), 4))
    random.seed(42)
    random.shuffle(supports)
    
    selected = [] # List of (support_tuple, sign_tuple)
    
    # Precompute all possible sign tuples for a 4-support
    all_signs = list(itertools.product([1, -1], repeat=4))
    
    for supp in supports:
        supp_set = set(supp)
        # Determine forbidden sign patterns for this support
        # A sign pattern s is forbidden if there exists a selected (t_supp, t_sign)
        # such that |supp \cap t_supp| == 3 and s|_{supp \cap t_supp} == t_sign|_{supp \cap t_supp}
        
        forbidden = set()
        
        for t_supp, t_sign in selected:
            t_supp_set = set(t_supp)
            inter = supp_set.intersection(t_supp_set)
            if len(inter) == 3:
                # Get projection of t_sign on inter
                # t_sign is a tuple of 4 signs corresponding to t_supp
                # We need to map indices to signs
                t_sign_map = dict(zip(t_supp, t_sign))
                t_proj = tuple(t_sign_map[idx] for idx in inter)
                
                # Any s for supp such that s|_inter == t_proj is forbidden
                # s is a tuple of 4 signs corresponding to supp
                # supp has 4 indices. inter has 3.
                # The index in supp not in inter is free.
                # So there are 2 such s.
                # We can just mark the projection as forbidden?
                # No, we need to check specific s.
                # It's easier to just iterate all 16 s for supp and check.
                pass
        
        # Check each possible sign for supp
        valid_signs = []
        for s in all_signs:
            s_map = dict(zip(supp, s))
            is_valid = True
            for t_supp, t_sign in selected:
                t_supp_set = set(t_supp)
                inter = supp_set.intersection(t_supp_set)
                if len(inter) == 3:
                    t_sign_map = dict(zip(t_supp, t_sign))
                    # Check if projections match
                    match = True
                    for idx in inter:
                        if s_map[idx] != t_sign_map[idx]:
                            match = False
                            break
                    if match:
                        is_valid = False
                        break
            if is_valid:
                valid_signs.append(s)
                
        if valid_signs:
            # Pick one randomly or first one
            chosen_sign = valid_signs[0]
            selected.append((supp, chosen_sign))
            
    # Convert selected to vectors
    if len(selected) > baseline_count:
        vectors = []
        for supp, signs in selected:
            v = [0] * D
            for idx, sign in zip(supp, signs):
                v[idx] = sign
            vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    else:
        return np.array(baseline_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
