# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Returns a kissing configuration in dimension 11.
    We construct a large set of integer lattice points in R^11 where:
    - No point is the zero vector.
    - The minimum squared distance between any pair >= the maximum squared norm.
    
    Strategy: 
    1. Generate a pool of candidate integer vectors.
    2. Use a greedy algorithm with random restarts to select a large subset satisfying the distance constraint.
    3. The constraint is: for all distinct p, q in set, ||p-q||^2 >= max( ||r||^2 for r in set ).
       Since we want to maximize k, and the score is k, we want as many points as possible.
       Note that if all points have the same norm R^2, the condition is ||p-q||^2 >= R^2.
    
    We will focus on vectors of norm^2 = 2 (weight 2) and potentially mix in norm^2 = 4 vectors if beneficial,
    but norm^2=2 vectors are very efficient. The 220 weight-2 vectors form a valid set with min_dist^2=2.
    Can we add more?
    If we add a vector v with norm^2 > 2, say 4, then max_norm becomes 4.
    Then we need all pairs to have dist^2 >= 4.
    The weight-2 vectors have min dist^2 = 2 (e.g., (1,1,0...) and (1,-1,0...) have diff (0,2,0...), norm^2=4. 
    Wait, (1,1,0) and (1,1,0) are same. (1,1,0) and (-1,-1,0) diff (2,2,0) norm^2=8.
    (1,1,0) and (1,-1,0) diff (0,2,0) norm^2=4.
    (1,1,0) and (0,1,1) diff (1,0,-1) norm^2=2.
    So weight-2 vectors can have min dist^2 = 2.
    If we include any vector with norm^2 >= 4, the max norm is >= 4.
    Then we need min dist >= 4.
    Since weight-2 vectors can have dist^2=2, we cannot mix weight-2 and weight-4 if we keep all weight-2 vectors.
    We would have to prune the weight-2 set so that all remaining pairs have dist^2 >= 4.
    And all pairs involving weight-4 vectors must have dist^2 >= 4.
    And weight-4 vectors among themselves must have dist^2 >= 4.
    
    Let's try to find a large set of vectors with uniform norm.
    Norm^2 = 2: 220 vectors. Min dist 2. Valid.
    Norm^2 = 4: 
       Weight 4: 4 non-zero coords. (1,1,1,1...), (1,1,1,-1...), (1,1,0,2...), etc.
       Count is large.
       Min dist between two weight-4 vectors?
       (1,1,1,1,0...) and (1,1,1,-1,0...) diff (0,0,0,2,0...) norm^2=4.
       (1,1,1,1,0...) and (1,1,0,1,1...) diff (0,0,1,0,-1...) norm^2=2.
       So min dist is 2.
       If we select a subset of weight-4 vectors with min dist >= 4, we can have max norm 4.
       How many?
       This is a coding problem: binary code of length 11, weight 4, min distance 4 (in terms of squared Euclidean dist for +/-1 entries? No, dist^2 = 4 * Hamming distance / 2? 
       If entries are +/-1, dist^2 = 4 * (# positions where they differ).
       So we need # positions where they differ >= 1? No, dist^2 >= 4 means # diff >= 1.
       Wait, if two vectors differ in 1 position, e.g. 1 vs -1, diff is 2, norm^2=4.
       So any two distinct weight-4 vectors (with +/-1 entries) have dist^2 >= 4?
       Let's check.
       v = (1,1,1,1,0...), u = (1,1,1,1,0...). Same.
       v = (1,1,1,1,0...), u = (1,1,1,-1,0...). Diff in 1 pos. Dist^2 = 4.
       v = (1,1,1,1,0...), u = (1,1,1,0,1...). Diff in 2 pos (pos 3: 1->0, pos 4: 0->1? No, weight 4 means 4 non-zeros).
       v has non-zeros at 1,2,3,4. u has non-zeros at 1,2,3,5.
       Diff at 4: 1-0=1. Diff at 5: 0-1=-1. Dist^2 = 1^2 + (-1)^2 = 2.
       So min dist can be 2.
       
       So we need a set of weight-4 vectors such that any two differ in at least 2 positions where they have non-zeros?
       Actually, if they share 3 non-zero positions with same signs, and differ in the 4th, dist^2=4.
       If they share 3 non-zero positions with same signs, and the 4th position of v is non-zero and u is 0 (and vice versa), dist^2=2.
       So we need to avoid pairs that share 3 signs and differ in the support in that way.
       
       This is getting complex. Let's stick to the greedy approach on a mixed pool.
       Generate all integer vectors with norm^2 <= 4.
       Use greedy to pick a large set.
    """
    d = 11
    deadline = time.time() + 580  # Leave 20s for safety
    
    # Generate candidates
    # Norm^2 = 2: (1,1,0...), (1,-1,0...), etc. 220 vectors.
    # Norm^2 = 4:
    #   (2,0,0...), (-2,0...), etc. 11*2 = 22 vectors.
    #   (1,1,1,1,0...), etc. C(11,4)*2^4 = 330 * 16 = 5280 vectors.
    #   (1,1,1,-1,0...), etc. included above.
    #   (1,1,0,2...)? No, 1+1+0+4=6.
    #   So only (2,0...) and (1,1,1,1...) types for norm^2=4.
    
    candidates = []
    
    # Type A: Norm^2 = 2
    for i in range(d):
        for j in range(i+1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    candidates.append(p)
    
    # Type B: Norm^2 = 4
    # Subtype B1: (2, 0, ...)
    for i in range(d):
        for s in (-2, 2):
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
            
    # Subtype B2: (1, 1, 1, 1, 0, ...)
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                p[idx] = val
            candidates.append(p)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    # Remove duplicates (just in case, though types are disjoint)
    # Candidates are all distinct.
    
    # Greedy selection with random restarts
    best_set = []
    best_size = 0
    
    n_cands = len(candidates)
    
    # We want to maximize the size.
    # A set S is valid if min_{p,q in S, p!=q} ||p-q||^2 >= max_{r in S} ||r||^2.
    # Let M = max_{r in S} ||r||^2.
    # If we only pick norm^2=2 vectors, M=2. We need min_dist^2 >= 2.
    # If we pick any norm^2=4 vector, M=4. We need min_dist^2 >= 4.
    
    # Let's try to find a large set with M=4 first, as it might allow more vectors if the distance constraint is satisfiable.
    # If not, fall back to M=2.
    
    # Strategy:
    # 1. Generate all subsets of size K that satisfy M=4 condition.
    # 2. Or, greedy:
    #    - Shuffle candidates.
    #    - Add vectors one by one.
    #    - Maintain current set S.
    #    - When considering adding v:
    #       - New M = max(M_old, ||v||^2).
    #       - Check if all pairs in S have dist^2 >= New M. (This is expensive to check fully every time).
    #       - Better: Keep M fixed.
    #       - Try with M=4. Only consider vectors with norm^2 <= 4.
    #         Check if adding v to S (where all pairs have dist^2 >= 4) keeps this property.
    #         i.e., dist(v, u) >= 4 for all u in S.
    #       - If we fail to get a large set, try M=2.
    
    def try_greedy(M_target, cands_filter):
        # cands_filter is a boolean mask or list of indices for candidates with norm^2 <= M_target
        # We actually want to select from all candidates, but the constraint is dist >= M_target.
        # And we can only use candidates with norm^2 <= M_target? No, if we use a vector with norm^2 > M_target, M increases.
        # So if we fix M_target, we must only use vectors with norm^2 <= M_target.
        
        S = []
        S_arr = []
        
        # Sort candidates by norm^2, then random?
        # Let's just iterate in random order.
        idxs = np.where(cands_filter)[0]
        np.random.shuffle(idxs)
        
        for idx in idxs:
            v = candidates[idx]
            v_norm_sq = np.dot(v, v)
            if v_norm_sq > M_target:
                continue
                
            if len(S_arr) == 0:
                S_arr.append(v)
                continue
                
            # Check dist(v, u) >= M_target for all u in S_arr
            # dist^2 = ||v||^2 + ||u||^2 - 2<v,u>
            # We need dist^2 >= M_target
            dots = np.dot(S_arr, v)
            S_norms_sq = np.sum(S_arr * S_arr, axis=1)
            dists_sq = v_norm_sq + S_norms_sq - 2 * dots
            
            if np.all(dists_sq >= M_target):
                S_arr.append(v)
                
        return np.array(S_arr)
        
    # Try M=4
    # Filter candidates with norm^2 <= 4
    norms_sq = np.sum(candidates * candidates, axis=1)
    mask_4 = norms_sq <= 4
    
    # Random restarts for M=4
    best_size_4 = 0
    best_set_4 = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
        # Shuffle candidates for this restart
        # We pass the mask, but try_greedy shuffles indices inside.
        # We should ensure we are using the correct mask.
        S = try_greedy(4, mask_4)
        if len(S) > best_size_4:
            best_size_4 = len(S)
            best_set_4 = S
        if best_size_4 > 300: # Heuristic cutoff
            break
            
    # Try M=2
    mask_2 = norms_sq <= 2
    best_size_2 = 0
    best_set_2 = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        S = try_greedy(2, mask_2)
        if len(S) > best_size_2:
            best_size_2 = len(S)
            best_set_2 = S
        if best_size_2 > 210: # We know 220 is possible
            break

    # Choose the best
    if best_size_4 > best_size_2:
        return best_set_4
    else:
        return best_set_2

# EVOLVE-BLOCK-END
