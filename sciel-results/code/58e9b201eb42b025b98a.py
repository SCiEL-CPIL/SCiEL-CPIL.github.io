"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    n_points = N * N
    
    # Index: point (x, y) -> index x * N + y
    def idx(x, y):
        return x * N + y
    
    def deidx(i):
        return (i // N, i % N)
    
    # Precompute squared distances
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2).astype(np.int32)
    
    def is_valid(p, S):
        """
        Check if point p can be added to set S (list of indices) without forming an isosceles triangle.
        S is a list of indices.
        """
        if len(S) < 2:
            return True
        
        S_arr = np.array(S, dtype=np.int32)
        k = len(S_arr)
        
        # Distances from p to all points in S
        d_p_S = dist_sq[p, S_arr]
        
        # Distances between points in S
        d_S_S = dist_sq[np.ix_(S_arr, S_arr)]
        
        # Check 1: Apex at p.
        # Are there any duplicate distances from p to S?
        # If d(p, a) == d(p, b) for a != b, then (a, b, p) is isosceles.
        if len(np.unique(d_p_S)) < k:
            return False
        
        # Check 2: Apex at S_i.
        # For each i, is d(p, S_i) equal to d(S_i, S_j) for some j != i?
        # We check if d_p_S[i] is in d_S_S[i, :].
        # Note: d_S_S[i, i] = 0. d_p_S[i] > 0 (since p not in S).
        # So if d_p_S[i] is in d_S_S[i, :], it must be at some j != i.
        for i in range(k):
            if d_p_S[i] in d_S_S[i, :]:
                return False
        
        # Check 3: Apex at S_j.
        # For each j, is d(p, S_j) equal to d(S_i, S_j) for some i != j?
        # This is symmetric to Check 2.
        # d_S_S is symmetric, so d_S_S[i, j] = d_S_S[j, i].
        # The condition "d_p_S[j] is in d_S_S[:, j]" is equivalent to "d_p_S[j] is in d_S_S[j, :]".
        # So Check 2 already covers Check 3?
        # Let's verify.
        # Triangle (a, b, p).
        # Case 2 (Apex a): d(a, p) == d(a, b).
        #   In our notation: d_p_S[i] == d_S_S[i, j].
        # Case 3 (Apex b): d(b, p) == d(a, b).
        #   In our notation: d_p_S[j] == d_S_S[i, j].
        #   Since d_S_S is symmetric, d_S_S[i, j] is in row i and column j.
        #   Check 2 checks if d_p_S[i] is in row i.
        #   Check 3 needs to check if d_p_S[j] is in column j.
        #   But column j is the same as row j (transposed).
        #   So if we check "for all i, d_p_S[i] is in d_S_S[i, :]", we cover all cases where the apex is one of the points in S.
        #   Wait.
        #   If d_p_S[j] == d_S_S[i, j], then d_p_S[j] is in column j.
        #   Since d_S_S is symmetric, d_S_S[i, j] = d_S_S[j, i].
        #   So d_p_S[j] is in row j.
        #   So yes, checking rows covers all apex-in-S cases.
        
        return True

    # Greedy Phase
    random.seed(42)
    point_order = list(range(n_points))
    random.shuffle(point_order)
    
    selected = []
    selected_set = set()
    
    for p in point_order:
        if time.time() > deadline:
            break
        
        if p in selected_set:
            continue
        
        if is_valid(p, selected):
            selected.append(p)
            selected_set.add(p)
    
    best_size = len(selected)
    
    # Local Search Phase
    # Try to improve by swapping or adding
    remaining = [p for p in range(n_points) if p not in selected_set]
    
    while time.time() < deadline:
        # Try to add a point
        if remaining:
            # Pick a few random candidates from remaining
            candidates = random.sample(remaining, min(10, len(remaining)))
            added = False
            for p in candidates:
                if is_valid(p, selected):
                    selected.append(p)
                    selected_set.add(p)
                    remaining.remove(p)
                    best_size = len(selected)
                    added = True
                    break
            
            if added:
                continue
        
        # If no point added, try a swap
        if len(selected) > 0:
            # Pick a random point to remove
            r_idx = random.randint(0, len(selected) - 1)
            r = selected[r_idx]
            
            # Pick a random point to add
            if remaining:
                p = random.choice(remaining)
                
                # Check if (selected - {r}) + {p} is valid
                # This is equivalent to checking if p is valid with (selected - {r})
                # AND that the remaining points are still valid (they are, since they were valid before)
                
                # Create a temporary list without r
                temp_selected = selected[:r_idx] + selected[r_idx+1:]
                
                if is_valid(p, temp_selected):
                    # Perform swap
                    selected[r_idx] = p
                    selected_set.remove(r)
                    selected_set.add(p)
                    remaining.remove(p)
                    remaining.append(r)
        
        # Occasionally shuffle remaining to avoid bias?
        if random.random() < 0.01:
            random.shuffle(remaining)

    result = [deidx(i) for i in selected]
    return result
# EVOLVE-BLOCK-END
