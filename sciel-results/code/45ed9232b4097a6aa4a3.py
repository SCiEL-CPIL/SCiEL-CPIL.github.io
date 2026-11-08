import numpy as np
import itertools
import random
import time

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses a hybrid approach:
    1. Weight-2 vectors (norm^2=2) - pairwise compatible since distance^2=2 >= 2
    2. Weight-4 vectors (norm^2=4) - pairwise compatible if |dot|<=2
    3. Weight-6 vectors (norm^2=6) - pairwise compatible with weight-2 if |dot|<=2
    
    Actually, let me reconsider. The condition is:
    min_{i!=j} ||v_i - v_j|| >= max_i ||v_i||
    
    Equivalently: min_{i!=j} (||v_i||^2 + ||v_j||^2 - 2*v_i·v_j) >= max_i ||v_i||^2
    
    For two vectors of the same norm^2 = r:
    2r - 2*v_i·v_j >= r  =>  v_i·v_j <= r/2
    
    For weight-2 (r=2): dot <= 1, so they can share at most 1 coordinate with same sign
    Actually, if they share 1 coordinate with same sign, dot=1 <= 1, OK.
    If they share 1 coordinate with opposite sign, dot=-1 <= 1, OK.
    If they share 0 coordinates, dot=0 <= 1, OK.
    If they share 2 coordinates, dot = ±2 or 0. If ±2, that's > 1, NOT OK.
    So weight-2 vectors must have disjoint supports or... wait, if they share 2 coords,
    the dot product is sum of products. If both have 1 at position i and 1 at position j,
    dot includes 1+1=2, which is > 1. So NOT compatible.
    If one has +1 and other has -1 at shared positions, it could be 0 or -2.
    
    Hmm, this is getting complex. Let me just use the greedy approach with direct checking.
    
    Strategy:
    1. Generate weight-2 vectors: C(11,2)*4 = 220 vectors, all norm^2=2
       These are all pairwise compatible (max dot=1 when sharing one coord with same sign)
       Wait, no. Two weight-2 vectors sharing both coordinates: e.g., (1,1,0,...) and (1,1,0,...)
       dot=2 > 1. Not compatible.
       Sharing one coordinate with same sign: dot=1 <= 1. OK.
       Sharing one coordinate with opposite sign: dot=-1 <= 1. OK.
       Sharing zero coordinates: dot=0 <= 1. OK.
       
    2. For a set of weight-2 vectors, they form a graph where edges connect compatible pairs.
       We want a large independent set in the conflict graph.
    
    Actually, the simplest approach that's known to work well:
    - Take all weight-2 vectors: there are C(11,2)*4 = 220
    - Among these, two are compatible iff their supports intersect in at most 1 position
      AND if they intersect in 1 position, the signs must be opposite OR we need dot<=1.
      If they share position i: v1[i]*v2[i] is +1 or -1.
      If +1: dot=1, OK (1<=1)
      If -1: dot=-1, OK
      So actually ANY two weight-2 vectors that share at most 1 coordinate are compatible.
      Two that share 2 coordinates are NOT compatible (dot=±2 or 0; if 0, that means
      one has +1 and other -1 at one, and... wait, if they share 2 coords, say positions i,j,
      v1=(1,1,0,...), v2=(1,-1,0,...) then dot=1-1=0 <= 1. OK!
      v1=(1,1,0,...), v2=(-1,1,0,...) then dot=-1+1=0. OK!
      v1=(1,1,0,...), v2=(1,1,0,...) then dot=2 > 1. NOT OK.
      v1=(1,1,0,...), v2=(-1,-1,0,...) then dot=-2 <= 1. OK!
      
    So two weight-2 vectors with the same support are compatible iff they are not identical
    and not negatives... wait:
    Same support, same signs: dot=2, NOT compatible.
    Same support, opposite signs: dot=-2, compatible.
    Same support, mixed: dot=0, compatible.
    
    This is getting complicated. Let me just use a brute-force greedy with vectorized dot products.
    """
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Generate candidate pool: weight 2 and weight 4 vectors
    candidates = []
    
    # Weight 2 vectors: norm^2 = 2
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[comb[0]] = signs[0]
            v[comb[1]] = signs[1]
            candidates.append(v)
    
    # Weight 4 vectors: norm^2 = 4
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)
    
    N = len(candidates)
    cand_arr = np.array(candidates, dtype=np.float64)
    
    # Compute all pairwise dot products and norms
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    # We want to find a subset where for all i != j:
    # ||v_i - v_j||^2 >= max(norms_sq)
    # i.e., norms_sq[i] + norms_sq[j] - 2*dot(i,j) >= max_norm_sq
    
    # Greedy: sort by some heuristic, add if compatible with all already added
    best_set = []
    
    # Try multiple random orderings
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(N))
        random.shuffle(order)
        
        selected = []
        selected_arr = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            v = cand_arr[idx]
            norm_sq = norms_sq[idx]
            
            if not selected:
                selected.append(idx)
                selected_arr.append(v)
                continue
            
            sel_arr = np.array(selected_arr)
            # Check: ||v - u||^2 >= max(norm_sq, max_sel_norm_sq) for all u in selected
            dots = sel_arr @ v
            sel_norms_sq = norms_sq[np.array(selected)]
            
            # Condition: norm_sq + sel_norms_sq[i] - 2*dots[i] >= max(norm_sq, max(sel_norms_sq))
            min_rhs = max(norm_sq, np.max(sel_norms_sq))
            lhs = norm_sq + sel_norms_sq - 2 * dots
            
            if np.all(lhs >= min_rhs - 1e-9):
                selected.append(idx)
                selected_arr.append(v)
        
        if len(selected) > len(best_set):
            best_set = selected
            # If we've found something very good, we can continue but won't stop early
            # since we want to maximize
    
    if not best_set:
        # Fallback: just return weight-2 vectors (a safe subset)
        # All weight-2 vectors with distinct supports and... actually just return a small valid set
        result = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        return result
    
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
