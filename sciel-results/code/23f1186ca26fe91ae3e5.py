"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290 second deadline to be safe
    
    best_points = []
    best_count = 0
    
    # Fast vectorized validity check
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        
        # Squared distances from p to all points in P
        # Shape: (N,)
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        n = len(P)
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        # This is equivalent to checking if there are duplicate values in d_p
        if len(set(d_p)) != n:
            return False
            
        # Squared distances between all pairs in P
        # We only need to check pairs (i, j) with i < j
        # Calculate all pairwise squared distances
        # D[i, j] = ||P_i - P_j||^2
        # We can compute this using broadcasting
        # P[:, np.newaxis, :] - P[np.newaxis, :, :]
        # This creates a (N, N, 2) array, which might be large for N=100 (100*100*2 = 20000, fine)
        # But we only need upper triangle.
        
        # Efficient calculation of pairwise squared distances
        # D_sq[i, j] = ||P_i||^2 + ||P_j||^2 - 2 P_i . P_j
        # P_i . P_j can be computed via matrix multiplication
        P_norm_sq = np.sum(P**2, axis=1) # (N,)
        Dot = P @ P.T # (N, N)
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        # D_sq is symmetric, diagonal is 0
        
        # We need to check if for any i < j:
        # d_p[i] == D_sq[i, j]  OR  d_p[j] == D_sq[i, j]
        
        # Create a mask for i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Get the relevant parts
        # d_p[i] compared to D_sq[i, j]
        # d_p[j] compared to D_sq[i, j]
        
        # This is tricky to vectorize directly without loops or complex masking.
        # Let's iterate over i, but vectorize over j.
        # For each i, we check:
        # 1. d_p[i] == D_sq[i, j] for any j > i
        # 2. d_p[j] == D_sq[i, j] for any j > i
        
        for i in range(n):
            if i >= n - 1:
                break
            # j indices are i+1 to n-1
            js = np.arange(i + 1, n)
            
            # Check d_p[i] == D_sq[i, js]
            if np.any(D_sq[i, js] == d_p[i]):
                return False
                
            # Check d_p[js] == D_sq[i, js]
            if np.any(D_sq[i, js] == d_p[js]):
                return False
                
        return True

    # Greedy construction with random restarts
    def greedy_construct():
        pts = []
        candidates = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, p):
                pts.append(p)
        return pts

    # Local search to improve
    def local_search(initial_pts):
        pts = list(initial_pts)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding a random point
            for _ in range(2000):
                if time.time() > deadline:
                    break
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p not in pts and is_valid_addition(pts, p):
                    pts.append(p)
                    improved = True
                    
            # Try swapping
            if len(pts) > 0:
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(pts) - 1)
                    old_pt = pts[idx]
                    pts_temp = pts[:idx] + pts[idx+1:]
                    
                    # Try to find a replacement
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        p = (random.randint(0, N-1), random.randint(0, N-1))
                        if p not in pts_temp and is_valid_addition(pts_temp, p):
                            pts = pts_temp + [p]
                            improved = True
                            break
        return pts

    # Run multiple greedy constructions
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        if len(pts) > best_count - 2 and time.time() < deadline - 20:
            improved_pts = local_search(pts)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
