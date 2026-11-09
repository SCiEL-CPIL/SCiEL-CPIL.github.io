"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        if U_np.shape != (nm, rank) or V_np.shape != (mp, rank) or W_np.shape != (pn, rank):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m + j, c] * V_np[j*p + k, c] * W_np[k*n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(rank, max_attempts=50, seed_base=42):
        """Try to find a decomposition of given rank using random init + refinement."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(seed_base + attempt)
            
            # Initialize with random values
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Normalize each column to unit norm to avoid scaling issues
            for c in range(rank):
                u_norm = np.linalg.norm(U[:, c])
                v_norm = np.linalg.norm(V[:, c])
                w_norm = np.linalg.norm(W[:, c])
                
                if u_norm < 1e-10 or v_norm < 1e-10 or w_norm < 1e-10:
                    continue
                    
                # Scale so that product of norms = 1 for each column
                # This helps with numerical stability
                scale_u = 1.0 / (u_norm ** (1.0/3.0))
                scale_v = 1.0 / (v_norm ** (1.0/3.0))
                scale_w = 1.0 / (w_norm ** (1.0/3.0))
                
                U[:, c] *= scale_u
                V[:, c] *= scale_v
                W[:, c] *= scale_w
            
            # Try to refine using iterative updates
            # For each column, we want U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] contributions
            # to sum to 1 for all (i,j,k)
            
            # Simple iterative refinement: adjust each matrix to reduce error
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Check current error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    return U.tolist(), V.tolist(), W.tolist(), rank
                
                # Gradient-like update: for each term that's too high/low,
                # adjust the corresponding entries
                lr = 0.01 / (1.0 + 0.01 * iteration)
                
                # Compute gradient contributions
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                            err = val - 1.0
                            
                            if abs(err) < 1e-15:
                                continue
                            
                            # Update each column's contribution
                            for c in range(rank):
                                u_val = U[i*m + j, c]
                                v_val = V[j*p + k, c]
                                w_val = W[k*n + i, c]
                                
                                # Partial derivatives
                                # d(err)/dU = V[k*n+i,c] * W[k*n+i,c] ... wait
                                # err = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
                                # d(err)/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]
                                # d(err)/dV[j*p+k,c] = U[i*m+j,c]*W[k*n+i,c]
                                # d(err)/dW[k*n+i,c] = U[i*m+j,c]*V[j*p+k,c]
                                
                                U[i*m + j, c] -= lr * v_val * w_val * err
                                V[j*p + k, c] -= lr * u_val * w_val * err
                                W[k*n + i, c] -= lr * u_val * v_val * err
            
            # Final check
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist(), rank
        
        return None
    
    # Start with schoolbook (rank 64) as guaranteed fallback
    rank_schoolbook = n * m * p  # 64
    U_school = [[0] * rank_schoolbook for _ in range(nm)]
    V_school = [[0] * rank_schoolbook for _ in range(mp)]
    W_school = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known bounds suggest rank could be lower than 64
    # Try ranks from 40 to 63
    for target_rank in range(40, 64):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, max_attempts=10, seed_base=target_rank * 1000)
        if result is not None:
            U, V, W, r = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
    
    # Also try a structured approach: exploit the fact that n=2, m=4, p=8
    # The tensor has special structure
    # Try rank 32 with a block structure
    
    if time.time() < deadline - 30 and best_rank == 64:
        # Try a different approach: use the fact that we can combine terms
        # For each output element C[i,k], we sum over j
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Try rank = n*p*m/2 or similar heuristics
        for target_rank in [32, 36, 40, 44, 48, 52, 56]:
            if time.time() > deadline:
                break
            
            result = try_rank(target_rank, max_attempts=20, seed_base=target_rank * 2000 + 100000)
            if result is not None:
                U, V, W, r = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
