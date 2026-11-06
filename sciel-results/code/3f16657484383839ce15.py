import random
import time
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Precompute all points as numpy array
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    def check_isosceles_vectorized(existing, new_point):
        """Check if new_point creates any isosceles triangle with existing points."""
        if len(existing) < 2:
            return False
        existing = np.array(existing, dtype=np.int32)
        # Compute all pairwise squared distances among existing
        # and distances from new_point to each existing
        dx = existing[:, 0:1] - new_point[0]
        dy = existing[:, 1:2] - new_point[1]
        d_new = dx**2 + dy**2  # shape (n, 1)
        
        # For each pair (i, j) in existing, check if (i, j, new) is isosceles
        n = len(existing)
        for i in range(n):
            for j in range(i+1, n):
                d_ij_sq = (existing[i,0]-existing[j,0])**2 + (existing[i,1]-existing[j,1])**2
                d_i_new = d_new[i, 0]
                d_j_new = d_new[j, 0]
                if d_ij_sq == d_i_new or d_ij_sq == d_j_new or d_i_new == d_j_new:
                    return True
        return False
    
    best_result = []
    
    # Line-based baseline
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    best_result = [(x, 0) for x in xs]
    
    # Try random greedy
    attempts = 0
    while time.time() < deadline:
        order = np.random.permutation(len(all_points))
        result = []
        for idx in order:
            p = tuple(all_points[idx])
            if check_isosceles_vectorized(result, p):
                continue
            result.append(p)
        if len(result) > len(best_result):
            best_result = result
        attempts += 1
        if attempts % 20 == 0:
            # Try a different bias
            mask = (all_points[:, 0] % 2 == 0) & (all_points[:, 1] % 2 == 0)
            subset = all_points[mask]
            order2 = np.random.permutation(len(subset))
            result2 = []
            for idx in order2:
                p = tuple(subset[idx])
                if check_isosceles_vectorized(result2, p):
                    continue
                result2.append(p)
            if len(result2) > len(best_result):
                best_result = result2
    
    return best_result
