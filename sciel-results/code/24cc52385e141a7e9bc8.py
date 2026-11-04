"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    def verify_exact(U, V, W, rank):
        """Check if U, V, W (list of lists of floats) correctly decompose the tensor with half-integer entries."""
        # First, check if entries are close to half-integers
        # Then verify with Fractions
        
        # Convert to Fractions for exact check
        U_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in U]
        V_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in V]
        W_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in W]
        
        # Check if they are exactly half-integers (numerator is integer, denominator 1 or 2)
        # Actually, just checking the product sum is sufficient if we assume the rounding was correct.
        # But we must ensure the inputs to the sum are the rounded values.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def als_decompose(rank, seed):
        """Run ALS to find a rank-rank decomposition."""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent overflow/underflow
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        max_iters = 1000
        tol = 1e-10
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None, False
            
            # Update U
            # For each i, j: M (p x rank) * u (rank x 1) = ones (p x 1)
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M u = ones
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        return None, False
            U = new_U
            
            # Update V
            # For each j, k: N (n x rank) * v (rank x 1) = ones (n x 1)
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, rank))
                    for i in range(n):
                        # N[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank_N, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        return None, False
            V = new_V
            
            # Update W
            # For each k, i: P (m x rank) * w (rank x 1) = ones (m x 1)
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, rank))
                    for j in range(m):
                        # P[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, residuals, rank_P, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        return None, False
            W = new_W
            
            # Check convergence (optional, but good for early stop)
            # Compute error
            # err = 0
            # for i in range(n):
            #     for j in range(m):
            #         for k in range(p):
            #             val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            #             err += (val - 1.0)**2
            # if err < tol:
            #     break
            if it % 100 == 99:
                 # Quick error check
                 err = 0.0
                 for i in range(n):
                     for j in range(m):
                         for k in range(p):
                             val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                             err += (val - 1.0)**2
                 if err < 1e-8:
                     break
                     
        # Check final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0)**2
        if err > 1e-6:
            return None, False
            
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify exactness with snapped values
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if verify_exact(U_list, V_list, W_list, rank):
            return (U_list, V_list, W_list), True
        else:
            return None, False

    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try to find lower rank decompositions
    # Candidate ranks
    # Lower bound is roughly n*p = 12, but likely higher.
    # Try a range of ranks
    candidate_ranks = [55, 50, 45, 40, 35, 30, 25, 20, 15, 12]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            
            result, success = als_decompose(target_rank, seed)
            if success:
                best_U, best_V, best_W, best_rank = result, result[1], result[2], target_rank
                found = True
                break
        
        if found:
            # If we found a solution, we can stop or try to find even lower?
            # The task asks for rank-minimal.
            # If we found rank 50, we should check 45, etc.
            # The loop structure above handles this: it iterates down.
            # If we break, we return the best found so far.
            # But we want the MINIMAL.
            # So we should NOT break on the first found rank if we can try lower.
            # However, if we find rank 50, we continue to 45.
            # If we find nothing at 45, we keep 50.
            pass
            
    # If we didn't find anything better, return schoolbook
    if best_rank == rank_school:
        U, V, W, _ = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
