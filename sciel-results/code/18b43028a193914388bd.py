"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 120.0
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # For each support, precompute which other supports intersect in 0,1,2,3,4
    # and build a compatibility structure
    
    # We'll use a greedy approach:
    # 1. Start with empty set
    # 2. Repeatedly add the vector (support + sign assignment) that is compatible with the most existing vectors
    # 3. Use local search to improve
    
    # Represent a vector as (support_tuple, signs_tuple) where signs are +1/-1
    # Two vectors (s1, sig1) and (s2, sig2) are compatible if their dot product satisfies the constraint
    
    def dot_product(s1, sig1, s2, sig2):
        """Compute dot product of two weight-4 vectors."""
        # Find intersection of supports
        inter = set(s1) & set(s2)
        if len(inter) == 0:
            return 0
        total = 0
        for idx in inter:
            # Get sign at idx for both vectors
            i1 = s1.index(idx)
            i2 = s2.index(idx)
            total += sig1[i1] * sig2[i2]
        return total
    
    def is_compatible(s1, sig1, s2, sig2):
        """Check if two vectors satisfy distance >= norm constraint."""
        # Both have norm^2 = 4
        # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot
        # Need dist^2 >= 4 => 8 - 2*dot >= 4 => dot <= 2
        return dot_product(s1, sig1, s2, sig2) <= 2
    
    # For each support, precompute all 16 sign assignments
    all_signs = []
    for i in range(16):
        sig = []
        for j in range(4):
            sig.append(1 if (i >> j) & 1 else -1)
        all_signs.append(tuple(sig))
    
    # Precompute compatibility between all pairs of (support, signs)
    # This is 330*16 = 5280 vectors, so 5280^2/2 ~ 13.9M pairs - too many to precompute all
    
    # Instead, use a greedy approach with on-the-fly checking
    
    best_set = []
    best_score = 0
    
    # Precompute for each pair of supports, the intersection size
    inter_size = [[0]*n_supports for _ in range(n_supports)]
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i+1, n_supports):
            sj = set(supports[j])
            inter_size[i][j] = len(si & sj)
            inter_size[j][i] = inter_size[i][j]
    
    # Greedy: repeatedly pick the (support, signs) that maximizes compatibility with existing set
    # To speed up, only consider supports that haven't been used yet
    
    used_supports = set()
    
    def count_compatible(s_idx, sig):
        """Count how many existing vectors are compatible with supports[s_idx] and signs sig."""
        count = 0
        for (ex_s_idx, ex_sig) in best_set:
            if inter_size[s_idx][ex_s_idx] <= 2:
                # Intersection <= 2, always compatible (can choose signs)
                count += 1
            elif inter_size[s_idx][ex_s_idx] == 3:
                # Need to check actual sign assignment
                if is_compatible(supports[s_idx], sig, supports[ex_s_idx], ex_sig):
                    count += 1
            else:  # intersection == 4
                if is_compatible(supports[s_idx], sig, supports[ex_s_idx], ex_sig):
                    count += 1
        return count
    
    # Greedy construction
    for step in range(300):  # Try to add up to 300 vectors
        if time.time() > deadline - 10:
            break
        
        best_candidate = None
        best_compat = -1
        
        for s_idx in range(n_supports):
            if s_idx in used_supports:
                continue
            # Try all 16 sign assignments
            for sig in all_signs:
                compat = count_compatible(s_idx, sig)
                if compat > best_compat:
                    best_compat = compat
                    best_candidate = (s_idx, sig)
        
        if best_candidate is None or best_compat < len(best_set):
            # Can't add any more vectors that are compatible with all existing
            break
        
        s_idx, sig = best_candidate
        best_set.append((s_idx, sig))
        used_supports.add(s_idx)
    
    # Local search: try to replace one vector with two new ones
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 5:
                    break
                # Try removing vector i and adding two new ones
                removed = best_set.pop(i)
                used_supports.discard(removed[0])
                
                # Find best pair of new vectors to add
                best_pair = None
                best_pair_score = len(best_set) - 1  # Current size after removal
                
                for s1_idx in range(n_supports):
                    if s1_idx in used_supports:
                        continue
                    for sig1 in all_signs:
                        # Check compatibility with all remaining
                        ok1 = True
                        for (ex_s_idx, ex_sig) in best_set:
                            if not is_compatible(supports[s1_idx], sig1, supports[ex_s_idx], ex_sig):
                                ok1 = False
                                break
                        if not ok1:
                            continue
                        for s2_idx in range(n_supports):
                            if s2_idx in used_supports or s2_idx == s1_idx:
                                continue
                            for sig2 in all_signs:
                                # Check compatibility with all remaining and with (s1_idx, sig1)
                                if not is_compatible(supports[s1_idx], sig1, supports[s2_idx], sig2):
                                    continue
                                ok2 = True
                                for (ex_s_idx, ex_sig) in best_set:
                                    if not is_compatible(supports[s2_idx], sig2, supports[ex_s_idx], ex_sig):
                                        ok2 = False
                                        break
                                if ok2:
                                    new_size = len(best_set) + 2
                                    if new_size > best_pair_score:
                                        best_pair_score = new_size
                                        best_pair = ((s1_idx, sig1), (s2_idx, sig2))
                
                if best_pair and best_pair_score > len(best_set) - 1:
                    best_set.append(best_pair[0])
                    best_set.append(best_pair[1])
                    used_supports.add(best_pair[0][0])
                    used_supports.add(best_pair[1][0])
                    improved = True
                else:
                    # Put back the removed vector
                    best_set.append(removed)
                    used_supports.add(removed[0])
    
    # Convert to array
    if not best_set:
        # Fallback: return weight-2 vectors
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = []
    for (s_idx, sig) in best_set:
        v = [0] * D
        for k, idx in enumerate(supports[s_idx]):
            v[idx] = sig[k]
        result.append(v)
    
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
