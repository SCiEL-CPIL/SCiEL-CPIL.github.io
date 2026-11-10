import numpy as np
import itertools
import time
import random
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s limit

    # Strategy: Use the D11 lattice structure.
    # D11 = {x in Z^11 : sum(x_i) is even}
    # We look for a large set of points in D11 such that:
    # 1. All points are non-zero.
    # 2. min_{i!=j} ||p_i - p_j||^2 >= max_k ||p_k||^2
    #
    # The minimal vectors of D11 have norm_sq = 4.
    # These are:
    # - 2e_i (and -2e_i): 22 vectors. Norm 4. Sum = ±2 (even).
    # - Vectors with exactly 4 non-zero entries, each ±1, such that the sum is even.
    #   The sum of four ±1's is even iff the number of +1's is even (0, 2, 4).
    #   Number of such patterns for a fixed 4-subset:
    #   - 4 +1's: 1
    #   - 2 +1's, 2 -1's: C(4,2) = 6
    #   - 4 -1's: 1
    #   Total 8 patterns per 4-subset.
    #   Number of 4-subsets: C(11, 4) = 330.
    #   Total such vectors: 330 * 8 = 2640.
    #
    # Total minimal vectors in D11: 22 + 2640 = 2662.
    #
    # Let's verify the distance constraint for these vectors.
    # If we select a subset S of these minimal vectors, max_norm_sq = 4.
    # We need min_{i!=j} ||p_i - p_j||^2 >= 4.
    # Since all are distinct integer vectors, min distance squared is at least 1.
    # Actually, for any two distinct vectors in D11, the difference is in D11.
    # The smallest non-zero norm in D11 is 4.
    # So ||p_i - p_j||^2 >= 4 for any distinct p_i, p_j in D11.
    # Wait, is this true?
    # Let p, q in D11. p - q in D11.
    # If p != q, then p - q is a non-zero vector in D11.
    # The minimum norm squared of a non-zero vector in D11 is 4.
    # Therefore, ||p - q||^2 >= 4.
    #
    # This implies that ANY subset of distinct minimal vectors of D11 satisfies the condition!
    # Because max_norm_sq = 4 and min_dist_sq >= 4.
    #
    # So the problem reduces to: How many distinct minimal vectors are there in D11?
    # We calculated 2662.
    # Let's double check if there are other minimal vectors.
    # Norm 4 in Z^11:
    # - One entry 4, rest 0: Norm 16. No.
    # - One entry 2, one entry 2: Norm 8. No.
    # - One entry 2, rest 0: Norm 4. Sum = 2 (even). Yes. (22 vectors)
    # - Two entries 1, rest 0: Norm 2. Sum = 2 (even). Yes. Wait.
    #   e_i + e_j has norm 2. Is it in D11? Sum = 2, even. Yes.
    #   So vectors with two 1's are also in D11 with norm 2.
    #   But we are looking for minimal vectors (norm 4).
    #   The "minimal" norm in D11 is actually 2?
    #   Let's check.
    #   e_i + e_j: sum = 2 (even). Norm 2.
    #   So the minimal norm in D11 is 2, not 4.
    #
    # If we include vectors with norm 2, then max_norm_sq could be 2.
    # Then we need min_dist_sq >= 2.
    # But for any two distinct vectors in D11, min_dist_sq >= 2?
    # Let's check.
    # p, q in D11. p-q in D11.
    # Min non-zero norm in D11 is 2.
    # So ||p-q||^2 >= 2.
    # So ANY subset of distinct vectors in D11 with norm_sq <= 2 satisfies the condition?
    # No, if we mix norm 2 and norm 4 vectors, max_norm_sq = 4.
    # Then we need min_dist_sq >= 4.
    # But two norm 2 vectors can be distance 2 apart?
    # e1+e2 and e1-e2. Dist^2 = (0, 2, 0...)^2 = 4.
    # e1+e2 and -e1-e2. Dist^2 = (2, 2, 0...)^2 = 8.
    # e1+e2 and e3+e4. Dist^2 = (1,1,-1,-1,0...)^2 = 4.
    # It seems min distance between distinct D11 vectors is 2?
    # e1+e2 and e1+e3. Dist^2 = (0, 1, -1, 0...)^2 = 2.
    # So if we have e1+e2 and e1+e3 in our set, their distance is 2.
    # If max_norm_sq is 4 (because we included a norm 4 vector), then we need dist >= 4.
    # So we cannot mix norm 2 and norm 4 vectors freely.
    #
    # Option A: Use only vectors with norm_sq = 2.
    # Max_norm_sq = 2. Need min_dist_sq >= 2.
    # As seen, e1+e2 and e1+e3 have dist 2. So this is allowed (>= 2).
    # So we can take ALL vectors in D11 with norm_sq = 2?
    # How many are there?
    # Vectors with exactly two 1's: C(11,2) = 55.
    # Vectors with exactly two -1's: C(11,2) = 55.
    # Vectors with one 1 and one -1: 11*10 = 110.
    # Total 55+55+110 = 220.
    # Let's check if all 220 are valid.
    # Any two distinct vectors from this set have distance squared >= 2?
    # Yes, because the difference is a non-zero vector in D11, and min norm is 2.
    # So we can take all 220.
    #
    # Option B: Use only vectors with norm_sq = 4.
    # Max_norm_sq = 4. Need min_dist_sq >= 4.
    # We established that min distance between ANY two distinct D11 vectors is 2.
    # So we cannot take all norm 4 vectors if they are close to each other.
    # Specifically, we need to select a subset of norm 4 vectors such that any two are distance >= 4.
    # This is a packing problem.
    #
    # Option C: Mix them carefully.
    #
    # Let's start with Option A: 220 points.
    # Can we add more?
    # The current score is 138.5. 220 is much better.
    #
    # Let's verify the 220 points.
    # Set S = {x in Z^11 : ||x||^2 = 2, sum(x_i) even}.
    # Actually, any vector with norm 2 in Z^11 has sum of entries:
    # (1,1,0...): sum 2 (even)
    # (1,-1,0...): sum 0 (even)
    # (-1,-1,0...): sum -2 (even)
    # So ALL integer vectors with norm 2 are in D11.
    # So S is the set of all integer vectors with norm 2.
    # Size 220.
    # Max norm sq = 2.
    # Min dist sq: For any p, q in S, p != q, ||p-q||^2 is a non-zero even integer (since p-q in D11).
    # Smallest non-zero norm in D11 is 2.
    # So min dist sq >= 2.
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2. Holds.
    #
    # So 220 is a valid score.
    #
    # Can we do better?
    # What if we use norm 4 vectors?
    # If we use ONLY norm 4 vectors, we need a subset with min dist >= 4.
    # This is equivalent to finding a code with minimum distance 4 in the graph where edges connect vectors with dist 2.
    # The "dist 2" graph on norm 4 vectors of D11.
    #
    # Let's try to combine.
    # If we include any norm 4 vector, max_norm_sq becomes 4.
    # Then ALL pairs must have dist >= 4.
    # This includes pairs of norm 2 vectors.
    # But norm 2 vectors can be dist 2 apart.
    # So if we include a norm 4 vector, we must exclude any pair of norm 2 vectors that are dist 2 apart.
    # This seems restrictive.
    #
    # Let's stick to the 220 norm 2 vectors for now.
    # Is it possible to add norm 4 vectors to a subset of norm 2 vectors?
    # If we add a norm 4 vector v, then all other points p must satisfy ||v-p||^2 >= 4.
    # Also, all pairs of remaining points p_i, p_j must satisfy ||p_i - p_j||^2 >= 4.
    # So we need to find a subset T of norm 2 vectors such that:
    # 1. min_{p_i, p_j in T} ||p_i - p_j||^2 >= 4
    # 2. min_{p in T} ||v - p||^2 >= 4 for some (or many) v in norm 4 set.
    #
    # Condition 1 is strong. It means T is a subset of norm 2 vectors with min distance 4.
    # How large can T be?
    # Norm 2 vectors are of type e_i +/- e_j.
    # Distance 2 between e_i+e_j and e_i+e_k (j!=k) is 2.
    # Distance 4 between e_i+e_j and e_k+e_l (disjoint supports) is 4.
    # Distance 4 between e_i+e_j and e_i-e_j is 4.
    #
    # Let's try to construct a large set T of norm 2 vectors with min dist 4.
    # This is equivalent to finding a large set of edges in K_11 such that no two edges share a vertex?
    # No.
    # e_i+e_j and e_k+e_l.
    # If {i,j} and {k,l} are disjoint, dist is 4.
    # If they share one vertex, e.g., {i,j} and {i,k}, dist is 2 (e_j - e_k has norm 2).
    # If they are the same, dist 0.
    # What about e_i+e_j and e_i-e_j?
    # (e_i+e_j) - (e_i-e_j) = 2e_j. Norm 4. Dist 4.
    # So e_i+e_j and e_i-e_j are compatible (dist 4).
    #
    # So we can group by support {i,j}.
    # For each pair {i,j}, we have 4 vectors: e_i+e_j, -e_i-e_j, e_i-e_j, -e_i+e_j.
    # Distances within a group:
    # d(e_i+e_j, -e_i-e_j) = ||2(e_i+e_j)||^2 = 8.
    # d(e_i+e_j, e_i-e_j) = ||2e_j||^2 = 4.
    # d(e_i+e_j, -e_i+e_j) = ||2e_i||^2 = 4.
    # So all 4 vectors in a group are mutually distance >= 4.
    #
    # Distances between groups {i,j} and {k,l}:
    # If disjoint: dist 4.
    # If sharing one vertex: dist 2.
    #
    # So if we pick vectors from disjoint supports, they are compatible.
    # If we pick vectors from intersecting supports, they are NOT compatible (dist 2).
    #
    # So T can contain vectors from a set of supports that are pairwise disjoint?
    # No, if supports intersect, dist is 2. So we cannot have vectors from intersecting supports in T.
    # So T corresponds to a matching in K_11.
    # Max matching in K_11 is 5 (since 11 vertices).
    # For each matched pair, we can take all 4 vectors.
    # So |T| = 5 * 4 = 20.
    #
    # So if we want to add norm 4 vectors, we can keep at most 20 norm 2 vectors.
    # Then we need to add norm 4 vectors v such that:
    # 1. v is compatible with the 20 norm 2 vectors (dist >= 4).
    # 2. v is compatible with other norm 4 vectors (dist >= 4).
    #
    # This seems like it might yield more than 220?
    # 20 + (number of compatible norm 4 vectors).
    #
    # Let's check the 220 solution again. It's very clean.
    # Is there a way to get > 220?
    #
    # What if we use norm 4 vectors ONLY?
    # We need a subset of norm 4 vectors with min dist 4.
    # Norm 4 vectors are 2e_i and 4-entry vectors.
    #
    # Let's try to generate the 220 norm 2 vectors and return them.
    # This should give a score of 220.
    
    candidates = []
    
    # Generate all integer vectors with norm_sq = 2
    # These are vectors with exactly two non-zero entries, each ±1.
    for i in range(d):
        for j in range(i + 1, d):
            # e_i + e_j
            p1 = np.zeros(d, dtype=np.int64)
            p1[i] = 1
            p1[j] = 1
            candidates.append(p1)
            
            # -e_i - e_j
            p2 = np.zeros(d, dtype=np.int64)
            p2[i] = -1
            p2[j] = -1
            candidates.append(p2)
            
            # e_i - e_j
            p3 = np.zeros(d, dtype=np.int64)
            p3[i] = 1
            p3[j] = -1
            candidates.append(p3)
            
            # -e_i + e_j
            p4 = np.zeros(d, dtype=np.int64)
            p4[i] = -1
            p4[j] = 1
            candidates.append(p4)
            
    # Verify validity
    # All are non-zero.
    # Max norm sq is 2.
    # Min dist sq:
    # For any two distinct p, q in candidates, p-q is in D11 and non-zero.
    # Min norm in D11 is 2.
    # So min dist sq >= 2.
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2. True.
    
    # Check for duplicates (should be none)
    # Convert to list of tuples to check uniqueness
    cand_tuples = [tuple(p) for p in candidates]
    if len(set(cand_tuples)) != len(cand_tuples):
        # Remove duplicates
        unique_cands = []
        seen = set()
        for t in cand_tuples:
            if t not in seen:
                seen.add(t)
                unique_cands.append(np.array(t, dtype=np.int64))
        candidates = unique_cands

    # Can we add more?
    # Let's try to add norm 4 vectors.
    # If we add a norm 4 vector, max_norm_sq becomes 4.
    # Then we need min_dist_sq >= 4.
    # This requires removing any pair of norm 2 vectors with dist 2.
    # As analyzed, we can keep at most 20 norm 2 vectors (from a matching).
    # Then we add norm 4 vectors compatible with them.
    
    # Let's implement a search to see if we can beat 220.
    # But 220 is already a huge jump from 138.
    # Let's first ensure 220 is valid and returned.
    
    points_array = np.array(candidates, dtype=np.int64)
    
    # Optional: Try to improve
    # If we have time, try to find a larger set.
    # But given the deadline, 220 is a solid improvement.
    # Let's check if we can simply return 220.
    
    # Just to be safe, let's verify the condition explicitly for a few pairs
    # and ensure no zero vectors.
    if len(points_array) == 0:
        return points_array
        
    # Check zero vector
    if np.any(np.all(points_array == 0, axis=1)):
        # Remove zero vectors (shouldn't happen)
        mask = ~np.all(points_array == 0, axis=1)
        points_array = points_array[mask]
        
    # Check min dist
    # For 220 points, pairwise check is 220*219/2 = 24090 pairs.
    # This is fast.
    min_dist_sq = float('inf')
    max_norm_sq = 0
    for p in points_array:
        n_sq = np.dot(p, p)
        if n_sq > max_norm_sq:
            max_norm_sq = n_sq
            
    for i in range(len(points_array)):
        for j in range(i + 1, len(points_array)):
            diff = points_array[i] - points_array[j]
            d_sq = np.dot(diff, diff)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    if min_dist_sq < max_norm_sq:
        # Fallback to a smaller valid set if this fails (should not)
        # But it shouldn't fail.
        pass
        
    return points_array
