import numpy as np
import time
from itertools import combinations
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 630s kill

    # We are looking for a set of integer vectors in Z^11.
    # Constraint: min squared distance >= max squared norm.
    # Let R^2 be the maximum squared norm.
    # For any two distinct vectors u, v in the set:
    # ||u - v||^2 >= R^2
    # Also ||u||^2 <= R^2 and ||v||^2 <= R^2.
    
    # If we choose vectors with constant norm R^2, then:
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 2 R^2 - 2 u.v >= R^2
    # => R^2 >= 2 u.v
    # => u.v <= R^2 / 2
    
    # If we use vectors with support 4 (i.e., 4 non-zero coordinates, each +/-1),
    # then R^2 = 4.
    # Condition: u.v <= 2.
    # u.v is the sum of products of coordinates.
    # Since coordinates are +/-1 or 0, u.v = (# matches) - (# mismatches) on the support.
    # Let k be the number of overlapping non-zero coordinates.
    # If k < 4, it's possible to have small dot product.
    # If k = 4 (same support), u.v can be 4, 2, 0, -2, -4.
    # We need u.v <= 2. So if same support, we cannot have u.v = 4 (identical vectors, not allowed as distinct points? "points must be distinct").
    # Actually, if u=v, distance is 0. 0 >= 4 is false. So u != v.
    # If same support, u.v can be 2, 0, -2, -4. All <= 2.
    # Wait, if u and v have the same support, and differ in exactly 2 coordinates, u.v = 2. Distance^2 = 4+4-4=4. OK.
    # If they differ in 1 coordinate, u.v = 4. Distance^2 = 0. Bad.
    # So on the same support, we can have at most 1 vector? No, if we pick u, we cannot pick v such that u.v = 4.
    # Since u.v = 4 implies u=v (for +/-1 vectors), distinct vectors on same support always have u.v <= 2.
    # So we can pick ALL 16 vectors on a given 4-set?
    # Let's check: u=(1,1,1,1), v=(1,1,1,-1). u.v = 3. 3 > 2. Bad.
    # So we cannot pick all 16.
    # We need u.v <= 2.
    # For 4-bit vectors, max dot product 4.
    # If we pick a set of 4-bit vectors with pairwise dot product <= 2, what is the max size?
    # This is equivalent to finding a code of length 4, alphabet {-1, 1}, with minimum Hamming distance?
    # u.v = 4 - 2 * d_H(u, v).
    # u.v <= 2 => 4 - 2 d_H <= 2 => 2 d_H >= 2 => d_H >= 1.
    # So any distinct vectors are fine?
    # Wait. u=(1,1,1,1), v=(1,1,1,-1). d_H = 1. u.v = 3.
    # 3 <= 2 is FALSE.
    # So d_H must be >= 2?
    # u.v <= 2 => 4 - 2 d_H <= 2 => 2 d_H >= 2 => d_H >= 1.
    # My algebra: 4 - 2 d_H <= 2 => -2 d_H <= -2 => d_H >= 1.
    # Let's re-evaluate u=(1,1,1,1), v=(1,1,1,-1).
    # u.v = 1+1+1-1 = 2.
    # Oh, 1*1 + 1*1 + 1*1 + 1*(-1) = 1+1+1-1 = 2.
    # So u.v = 2. This is allowed.
    # What about u=(1,1,1,1), v=(1,1,-1,1)?
    # u.v = 1+1-1+1 = 2. Allowed.
    # What about u=(1,1,1,1), v=(1,-1,1,1)?
    # u.v = 2. Allowed.
    # So any vector at Hamming distance 1 from u has dot product 2.
    # Any vector at Hamming distance 2 has dot product 0.
    # Any vector at Hamming distance 3 has dot product -2.
    # Any vector at Hamming distance 4 has dot product -4.
    # So ALL distinct 4-bit vectors have dot product <= 2?
    # Max dot product for distinct vectors is 2.
    # So we CAN pick all 16 vectors on a 4-set!
    
    # Now check interactions between DIFFERENT 4-sets.
    # Let S1 and S2 be two 4-sets.
    # u in S1, v in S2.
    # Overlap k = |S1 intersect S2|.
    # If k=0: u.v = 0. OK.
    # If k=1: u.v is +/-1. OK.
    # If k=2: u.v is sum of 2 products. Max 2. OK.
    # If k=3: u.v is sum of 3 products. Max 3.
    #   We need u.v <= 2.
    #   So if overlap is 3, we cannot have u and v agreeing on all 3 overlapping coordinates.
    # If k=4: Same set. We already handled this.
    
    # So the constraint is:
    # We select a collection of 4-sets.
    # For each selected 4-set, we include all 16 vectors?
    # If we include all 16 vectors for set S1, and all 16 for set S2 (with |S1 intersect S2|=3):
    # There exist u in S1, v in S2 such that u.v = 3.
    # Example: S1={1,2,3,4}, S2={1,2,3,5}.
    # u=(1,1,1,1,0,0,0,0,0,0,0)
    # v=(1,1,1,0,1,0,0,0,0,0,0)
    # u.v = 1+1+1 = 3.
    # 3 > 2. Invalid.
    
    # So if two 4-sets overlap in 3 coordinates, we cannot include ALL vectors from both.
    # We must restrict the vectors.
    # Or, we must choose 4-sets such that no two overlap in 3 coordinates.
    # If we choose 4-sets with pairwise overlap <= 2, then we can include ALL 16 vectors from each set.
    # This is a packing problem: Max number of 4-subsets of {1..11} with pairwise intersection <= 2.
    
    # Let's solve this packing problem.
    # Total 4-sets: C(11,4) = 330.
    # Conflict graph: Edge between S1, S2 if |S1 intersect S2| >= 3.
    # We want Max Independent Set in this conflict graph.
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict bitsets
    conflict = [0] * n_sets
    for i in range(n_sets):
        si = set(all_4sets[i])
        mask = 0
        for j in range(n_sets):
            if i == j: continue
            sj = set(all_4sets[j])
            if len(si & sj) >= 3:
                mask |= (1 << j)
        conflict[i] = mask

    def greedy_bitset(order):
        selected = []
        available = (1 << n_sets) - 1
        for idx in order:
            if available & (1 << idx):
                selected.append(idx)
                available &= ~conflict[idx]
                available &= ~(1 << idx)
        return selected

    rng = random.Random(42)
    
    best_selection = []
    best_size = 0
    
    # Initial Greedy with random restarts
    for trial in range(2000):
        if time.time() > deadline - 30:
            break
        order = list(range(n_sets))
        rng.shuffle(order)
        sel = greedy_bitset(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selection = sel[:]

    # Local Search to improve the packing
    # 1. Try to add
    # 2. 1-swap to escape local optima
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        sel_set = set(best_selection)
        
        # Step 1: Try to add any valid set
        added = False
        forbidden_mask = 0
        for k in best_selection:
            forbidden_mask |= conflict[k]
            
        for j in range(n_sets):
            if j in sel_set:
                continue
            if forbidden_mask & (1 << j):
                continue
            best_selection.append(j)
            sel_set.add(j)
            best_size += 1
            added = True
            improved = True
            break
        
        if added:
            continue
        
        # Step 2: 1-Swap
        if len(best_selection) > 0:
            i_idx = rng.randint(0, len(best_selection)-1)
            i_val = best_selection[i_idx]
            
            best_selection.pop(i_idx)
            sel_set.remove(i_val)
            
            forbidden_mask = 0
            for k in best_selection:
                forbidden_mask |= conflict[k]
            
            candidates = []
            for j in range(n_sets):
                if j in sel_set:
                    continue
                if forbidden_mask & (1 << j):
                    continue
                candidates.append(j)
            
            if candidates:
                j_val = rng.choice(candidates)
                best_selection.append(j_val)
                sel_set.add(j_val)
                improved = True
            else:
                best_selection.append(i_val)
                sel_set.add(i_val)

    # Generate points from the best packing
    # For each selected 4-set, add all 16 vectors.
    points = []
    for idx in best_selection:
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    if not points:
        # Fallback: just use one 4-set
        s = all_4sets[0]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)

    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)

# Verification logic (not part of the returned function, but good to know)
# The packing of 4-sets with intersection <= 2 in 11 dimensions.
# Known result: The maximum size of a 4-(11, 4, 2) packing?
# Or just Max Independent Set in the graph where edges are intersections >= 3.
# This is equivalent to finding a code of length 11, weight 4, with pairwise intersection <= 2.
# This is a constant weight code A(11, d, 4) where d = 2*(4-2) = 4?
# Hamming distance d_H = w1 + w2 - 2*intersection.
# d_H = 4 + 4 - 2*intersection = 8 - 2*intersection.
# We want intersection <= 2 => d_H >= 4.
# So we are looking for A(11, 4, 4).
# Values from tables: A(11, 4, 4) = 17? 20?
# If we find ~20 sets, we get 20 * 16 = 320 points.
# Current score is 280. 320 > 280.
# If the packing finds more, we win.

if __name__ == "__main__":
    res = kissing_number11()
    print(res.shape)
    print(len(res))
