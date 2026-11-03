"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _check_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # T[i, j, k] = A[i,j] * B[j,k] = C[i,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note: W index is [k*n+i] per the trace convention W[k*n+i, r]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Tensor value for (i,j,k) is 1 (the canonical multiplication tensor)
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol

def _schoolbook(n, m, p):
    """Return the schoolbook (trivial) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_based(n, m, p):
    """
    Try to construct a decomposition using Strassen's 7-term formula for 2x2x2 blocks.
    For 2x4x8: think of A as 2x(4) = [A1, A2] where A1, A2 are 2x2,
    B as (4)x8 = [B1, B2, B3, B4] where each is 2x2.
    This doesn't directly map to Strassen because the middle dimension is split differently.
    
    Alternative: Pad to 2x8x8 or use a different tiling.
    
    Let's try: A (2x4) = [A1 | A2] with A1,A2 in R^{2x2}
    B (4x8) = [B1, B2, B3, B4] each in R^{2x2}
    C (2x8) = C1*B1 + C2*B2 + C3*B3 + C4*B4 where C = A1*B1 + A2*B2 + A3*B3 + A4*B4? No.
    Actually C = A * B = A1*(B1+B2) + A2*(B3+B4)? No, that's not right either.
    
    Let me re-derive: A is 2x4, B is 4x8.
    A = [A11 A12] where A11 is 2x2, A12 is 2x2
    B = [B11 B12 B13 B14] where each is 2x2
    C = A*B = A11*B11 + A12*B13 ... wait no.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    For i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    C = A1 * B_left + A2 * B_right where A1 = A[:,0:2], A2 = A[:,2:4]
    B_left = B[0:2,:], B_right = B[2:4,:]
    So C = A1 * B_left + A2 * B_right where A1,A2 are 2x2 and B_left,B_right are 2x8.
    
    Now B_left = [B11 B12] where B11,B12 are 2x4? No, B is 4x8, so B_left is 2x8.
    Let B_left = [Bl1 Bl2] where Bl1, Bl2 are 2x4.
    This is getting complicated. Let me just try a generic approach: search over small ranks.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find a lower rank decomposition
    # The tensor is T[i,j,k] = 1 for all i,j,k (canonical multiplication tensor)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a tensor completion problem. For rank r, we need 3*r*(nm+mp+pn)/3 = r*(nm+mp+pn) parameters
    # and n*m*p = 64 constraints. For r >= 64, it's always solvable.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) / something... 
    # For 2x2x2, rank is 7 (Strassen). For 2x4x8, we can try ranks below 64.
    
    # Let's try a simple approach: use Strassen's 7-term formula as a building block.
    # Pad the 2x4x8 problem to 2x8x8 by duplicating B's columns? No, that changes the problem.
    
    # Alternative: try to find a decomposition of rank r for r in range(some_min, 64)
    # Use a greedy/iterative approach: initialize with random matrices and do gradient descent
    # on the loss sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    best_rank = n * m * p
    U_best, V_best, W_best = U_school, V_school, W_school
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: the rank must be at least max(n*m, m*p, p*n) in the sense that
    # the flattening has that rank. Actually, the border rank is at least max(n*m, m*p, p*n) / 3? No.
    # The rank is at least max(n*m, m*p, p*n) / something... let's just try from 24 up.
    
    for target_rank in range(24, 64):
        if time.time() > deadline:
            break
        
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(hash((n, m, p, target_rank, restart)) % (2**32))
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple iterative refinement: for each (i,j,k), adjust the column r that best fits
            # Actually, let's just do a few steps of gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] * 2*(val - 1)
                            # val = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = 2.0 * (val - 1.0)
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-7:
                    break
            
            # Final check
            if _check_decomposition(n, m, p, U, V, W, tol=1e-6):
                best_rank = target_rank
                U_best, V_best, W_best = U, V, W
                break  # Found a valid decomposition for this rank, try lower
        
        if best_rank == target_rank:
            continue  # Found it, try lower rank
        # If we didn't find it for this rank, keep trying
    
    # Convert to lists of lists for the return format
    U_ret = U_best.tolist()
    V_ret = V_best.tolist()
    W_ret = W_best.tolist()
    
    return {"U": U_ret, "V": V_ret, "W": W_ret}
# EVOLVE-BLOCK-END
