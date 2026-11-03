"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Schoolbook baseline
    rank_sb = n * m * p  # 120
    def make_schoolbook():
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = rank_sb

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def round_to_half(x):
        return round(x * 2) / 2.0

    def solve_for_U(V, W, rank):
        """
        Solve for U given V and W.
        For each (i, j), we have a system of p equations for r unknowns.
        """
        U = np.zeros((n * m, rank))
        
        # Precompute V and W as numpy arrays for speed
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for j in range(m):
                # Build the coefficient matrix A of size (p, rank)
                # A[k, c] = V[j*p + k, c] * W[k*n + i, c]
                A = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = V_arr[j * p + k, c] * W_arr[k * n + i, c]
                
                # Solve A * u = 1
                # Using least squares
                try:
                    u, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
                except:
                    return None
        
        return U

    def try_rank(rank, trials=100):
        for t in range(trials):
            if time.time() > deadline - 10.0:
                return None
            
            # Initialize V and W
            # Use small random values to avoid singularities
            # Scaling: if V, W are O(1), U will be O(1/rank) maybe?
            # Let's try uniform [-1, 1]
            V = np.random.uniform(-1, 1, (m * p, rank))
            W = np.random.uniform(-1, 1, (p * n, rank))
            
            U = solve_for_U(V, W, rank)
            if U is None:
                continue
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Convert to lists for verification (or verify with numpy)
            # Numpy verification is faster
            # Check sum_{c} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_r[i * m + j, :] * V_r[j * p + k, :] * W_r[k * n + i, :])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        return None

    # Try to find a better rank
    # Start from 119 down to 100
    for r in range(119, 99, -1):
        if time.time() > deadline - 20.0:
            break
        
        result = try_rank(r, trials=200)
        if result:
            best_result = result
            best_rank = r
            # If we find one, we can stop or continue to find lower
            # Given time, let's stop at first success to ensure we have a valid answer
            # But we can try to improve if time permits
            break

    return best_result
# EVOLVE-BLOCK-END
