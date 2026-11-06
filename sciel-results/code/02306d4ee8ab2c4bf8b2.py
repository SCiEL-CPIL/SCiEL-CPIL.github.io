import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 300  # 300s limit, leave margin

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
# points[i] = (x, y) for i in 0..4095

def squared_dist(p1, p2):
    """Calculate squared distance between two points or arrays of points."""
    diff = p1 - p2
    return np.sum(diff**2, axis=-1)

def is_valid_set(S_indices):
    """
    Check if the set of indices S_indices forms a no-isosceles set.
    Returns True if valid.
    """
    if len(S_indices) <= 2:
        return True
    
    S = points[S_indices]
    K = len(S)
    
    # D[i, j] is squared distance between S[i] and S[j]
    # Compute all pairwise squared distances
    # Using broadcasting: (K, 1, 2) - (1, K, 2) -> (K, K, 2)
    diff = S[:, np.newaxis, :] - S[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    # Check for isosceles triangles
    # For each triplet (i, j, k), check if any two sides are equal.
    # Since D is symmetric and D[i,i]=0, we just need to check i != j != k.
    
    # Optimization: 
    # A triangle is isosceles if d(i,j) == d(i,k) OR d(i,j) == d(j,k) OR d(i,k) == d(j,k).
    
    # We can check this by iterating over pairs (i, j) and seeing if there is a k 
    # such that d(i,k) == d(i,j) or d(j,k) == d(i,j) or d(i,k) == d(j,k).
    
    # Vectorized check:
    # For each i, for each j != i:
    #   d_ij = D[i, j]
    #   If there exists k != i, j such that D[i, k] == d_ij -> Conflict
    #   If there exists k != i, j such that D[j, k] == d_ij -> Conflict
    #   If there exists k != i, j such that D[i, k] == D[j, k] -> Conflict (This is equivalent to d(i,k)==d(j,k), which is covered by the first two if we iterate properly? No.)
    
    # Actually, if D[i,k] == D[j,k], that means k is on the perpendicular bisector of ij.
    # If D[i,k] == D[i,j], that means k is on the circle centered at i with radius d(i,j).
    
    # Let's just check all triplets for small K, or use a faster method.
    # For K ~ 80, K^3 is 500,000. That's fast.
    
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = D[i, j]
            for k in range(j + 1, K):
                d_ik = D[i, k]
                d_jk = D[j, k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    # Pre-shuffle indices for randomization
    all_indices = list(range(NUM_POINTS))
    
    # We will run multiple restarts of a greedy + local search algorithm
    while time.time() < DEADLINE - 10:  # Safety margin
        
        random.shuffle(all_indices)
        S = []
        S_set = set()
        
        # Greedy Phase
        for idx in all_indices:
            # Check if idx can be added to S
            if is_compatible(idx, S, points):
                S.append(idx)
                S_set.add(idx)
        
        # Local Search Phase
        # Try to improve the set by swapping or adding/removing
        # We'll use a simple randomized local search:
        # 1. Try to add a point that was rejected.
        # 2. If it conflicts, remove conflicting points.
        # 3. If we remove fewer points than we add (i.e., 0 removed), we gain.
        # But usually we remove > 0.
        # So we try Swaps: Remove 1, Add 1. Or Remove 2, Add 3?
        
        # Let's try a "Repair" strategy:
        # Pick a random point P not in S.
        # Check conflicts with S.
        # If conflicts exist, let C be the set of points in S involved in conflicts with P.
        # If len(C) < 1, add P.
        # If len(C) == 1, swap P with the point in C? No, that keeps size same.
        # We want to increase size.
        
        # Better Local Search: 
        # Identify "bad" points in S that block many candidates.
        # Remove a random point from S, then try to add 2 random points from the pool.
        
        pool = [i for i in all_indices if i not in S_set]
        
        # Run local search for a fixed number of steps or time
        local_search_deadline = time.time() + 5
        while time.time() < local_search_deadline and len(pool) > 0:
            # Try to add a random point from pool
            p_idx = random.choice(pool)
            # Check compatibility
            # We need a fast check that also returns the conflicting points
            conflicts = find_conflicts(p_idx, S, points)
            
            if not conflicts:
                S.append(p_idx)
                S_set.add(p_idx)
                pool.remove(p_idx)
            else:
                # Try to swap: remove one conflict, add p_idx
                # If we remove 1 and add 1, size is same.
                # We need to find a case where we can add 1 without removing, 
                # or remove 1 and add 2.
                
                # Let's try to remove a "least useful" point from S 
                # and then try to add 2 points from pool.
                
                # Heuristic: Remove a point that is involved in many potential conflicts?
                # For simplicity, remove a random point from S.
                if S:
                    rem_idx = random.choice(S)
                    S.remove(rem_idx)
                    S_set.remove(rem_idx)
                    pool.append(rem_idx)
                    
                    # Try to add 2 points from pool
                    added_count = 0
                    for _ in range(2):
                        if not pool: break
                        cand_idx = random.choice(pool)
                        if is_compatible(cand_idx, S, points):
                            S.append(cand_idx)
                            S_set.add(cand_idx)
                            pool.remove(cand_idx)
                            added_count += 1
                            
                    # If we added fewer than 1, we lost size. 
                    # If we added 1, size is same.
                    # If we added 2, size increased by 1.
                    # This is a gamble.
        
        if len(S) > len(best_set):
            best_set = S[:]
            # Optional: Verify best_set is valid
            # if not is_valid_set(best_set):
            #     print("Invalid best set found!")
            #     best_set = []
            
        # If we found a very large set, we might break early?
        # The max possible is unknown, but ~80 is expected.
        if len(best_set) > 90:
            break

    # Final verification
    if best_set and not is_valid_set(best_set):
         # Fallback to a small valid set if something went wrong
         return [(0, 0)]
         
    return [tuple(points[i]) for i in best_set]

def find_conflicts(p_idx, S, points):
    """
    Returns a list of indices in S that form an isosceles triangle with p_idx.
    """
    if not S:
        return []
    
    P = points[p_idx]
    S_pts = points[S]
    K = len(S)
    
    # Squared distances from P to all S
    d_PS = squared_dist(P, S_pts)
    
    # Squared distances within S
    # D[i, j]
    diff = S_pts[:, np.newaxis, :] - S_pts[np.newaxis, :, :]
    D_SS = np.sum(diff**2, axis=2)
    
    conflicts = set()
    
    for i in range(K):
        d_pi = d_PS[i]
        
        # Check if d_pi == d_ij for any j != i
        # D_SS[i, :] contains d_ij
        # We need to check if d_pi is in D_SS[i, :] (excluding j=i)
        # And if d_pi is in D_SS[:, i] (excluding i=i)
        
        # Conflict Type 1: d(P, S_i) == d(S_i, S_j)
        # This means S_j is on the circle centered at S_i with radius d(P, S_i).
        matches = np.where(D_SS[i, :] == d_pi)[0]
        for j in matches:
            if j != i:
                conflicts.add(S[i])
                conflicts.add(S[j])
                
        # Conflict Type 2: d(P, S_i) == d(P, S_j)
        # This means S_i and S_j are at the same distance from P.
        # We can check this globally for all pairs in S.
        
    # Check for d(P, S_i) == d(P, S_j)
    # Find duplicates in d_PS
    unique, counts = np.unique(d_PS, return_counts=True)
    for val, count in zip(unique, counts):
        if count > 1:
            indices_with_val = np.where(d_PS == val)[0]
            for i in indices_with_val:
                conflicts.add(S[i])
                
    # Check for d(S_i, S_j) == d(S_i, S_k) where P is involved?
    # No, the triangle is P, S_i, S_j.
    # Sides are d(P,S_i), d(P,S_j), d(S_i,S_j).
    # Isosceles if:
    # 1. d(P,S_i) == d(P,S_j) (Handled above)
    # 2. d(P,S_i) == d(S_i,S_j) (Handled above)
    # 3. d(P,S_j) == d(S_i,S_j) (Symmetric to 2)
    
    # So the logic above covers all cases.
    
    return list(conflicts)

def is_compatible(p_idx, S, points):
    """
    Check if p_idx can be added to S without forming an isosceles triangle.
    """
    if not S:
        return True
    # Reuse find_conflicts logic but exit early if any conflict found
    # For speed, we can just call find_conflicts and check if empty
    # But find_conflicts builds the whole set.
    # Let's write a faster check.
    
    P = points[p_idx]
    S_pts = points[S]
    K = len(S)
    
    if K == 1:
        return True # Two points never form a triangle
        
    d_PS = squared_dist(P, S_pts)
    
    # Check 1: d(P, S_i) == d(P, S_j)
    # If any duplicate in d_PS, return False
    if len(np.unique(d_PS)) < K:
        return False
        
    # Check 2 & 3: d(P, S_i) == d(S_i, S_j) or d(P, S_j) == d(S_i, S_j)
    # Compute D_SS
    diff = S_pts[:, np.newaxis, :] - S_pts[np.newaxis, :, :]
    D_SS = np.sum(diff**2, axis=2)
    
    for i in range(K):
        d_pi = d_PS[i]
        # Check if d_pi is in D_SS[i, :] (j != i)
        # D_SS[i, i] is 0, d_pi > 0 (since p_idx not in S)
        if np.any(D_SS[i, :] == d_pi):
            # We need to ensure j != i. 
            # If D_SS[i, i] == d_pi, that means 0 == d_pi, which is false.
            # So any match is valid.
            return False
            
        # Check if d_pi is in D_SS[:, i] (j != i)
        if np.any(D_SS[:, i] == d_pi):
            return False
            
    return True
