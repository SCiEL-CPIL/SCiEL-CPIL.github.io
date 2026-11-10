import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620

    def generate_candidates_R2():
        # Norm squared 2: +/- e_i +/- e_j
        vecs = []
        for i, j in combinations(range(d), 2):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    vecs.append(tuple(v))
        return vecs

    def generate_candidates_R4():
        # Norm squared 4
        vecs = []
        # Type 1: +/- 2 e_i
        for i in range(d):
            vecs.append(tuple([0]*i + [2] + [0]*(d-i-1)))
            vecs.append(tuple([0]*i + [-2] + [0]*(d-i-1)))
        
        # Type 2: +/- e_i +/- e_j +/- e_k +/- e_l
        for i, j, k, l in combinations(range(d), 4):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    for s3 in (1, -1):
                        for s4 in (1, -1):
                            v = [0] * d
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            v[l] = s4
                            vecs.append(tuple(v))
        return vecs

    def validate(points):
        if len(points) < 2:
            return False
        if len(points[0]) != 11:
            return False
        for p in points:
            if all(x == 0 for x in p):
                return False
        max_norm = 0
        for p in points:
            nn = sum(x * x for x in p)
            if nn > max_norm:
                max_norm = nn
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d2 = 0
                for a, b in zip(points[i], points[j]):
                    d2 += (a - b) * (a - b)
                if d2 < max_norm:
                    return False
        return True

    def greedy_mis_R4(candidates, max_iter=50):
        """
        Greedy MIS for R=4 Type 2 vectors.
        Conflict: overlap 3 and same signs on overlap.
        """
        best_set = []
        n = len(candidates)
        if n == 0:
            return []
        
        # Precompute supports and signs for speed?
        # Or just check on fly.
        # On fly is O(11) per pair.
        
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Shuffle
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            current_set = []
            current_set_list = []
            
            for v in shuffled:
                # Check against current_set
                conflict = False
                for u in current_set_list:
                    # Check overlap
                    common_count = 0
                    sign_match = True
                    for i in range(d):
                        if v[i] != 0 and u[i] != 0:
                            common_count += 1
                            if v[i] != u[i]:
                                sign_match = False
                                # If common_count > 3, we can break early?
                                # No, we need to know if common_count == 3.
                                # But if sign_match is False, and common_count becomes 3,
                                # it's not a conflict.
                                # If common_count > 3, it's not a conflict (overlap 4 is same support, handled? 
                                # Wait, Type 2 has support size 4.
                                # If overlap 4, they are same support.
                                # Conflict for same support?
                                # u, v same support.
                                # If u != v, dot <= 2.
                                # So NO conflict for overlap 4.
                                # So we only care about overlap 3.
                                pass
                    
                    if common_count == 3 and sign_match:
                        conflict = True
                        break
                
                if not conflict:
                    current_set_list.append(v)
            
            if len(current_set_list) > len(best_set):
                best_set = current_set_list
                
        return best_set

    # Strategy:
    # 1. Try R=4.
    # 2. If it yields > 220, use it.
    # 3. Else use R=2 (220 points).

    # Generate R=2 points (guaranteed 220)
    r2_points = generate_candidates_R2()
    
    # Generate R=4 points
    r4_points = generate_candidates_R4()
    
    # Separate Type 1 and Type 2 for R=4
    type1 = []
    type2 = []
    for v in r4_points:
        norm2 = sum(x*x for x in v)
        if norm2 == 4:
            # Check if it's type 1 (only one non-zero)
            non_zero = sum(1 for x in v if x != 0)
            if non_zero == 1:
                type1.append(v)
            else:
                type2.append(v)
                
    # Greedy on Type 2
    # We can include all Type 1?
    # Yes, verified earlier.
    # But do Type 1 conflict with Type 2?
    # No, verified earlier.
    # So we just need MIS on Type 2.
    
    best_mis = greedy_mis_R4(type2, max_iter=20)
    
    candidate_set = type1 + best_mis
    
    # Validate
    if validate(candidate_set):
        if len(candidate_set) > len(r2_points):
            result = np.array(candidate_set, dtype=np.int64)
            result = np.around(result).astype(np.int64)
            return result
    
    # Fallback to R=2
    result = np.array(r2_points, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    return result

if __name__ == "__main__":
    res = kissing_number11()
    print("Size:", len(res))
