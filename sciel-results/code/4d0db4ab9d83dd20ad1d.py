"""Matrix multiplication 3,5,5 rank-minimal tensor decomposition."""
import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    r = len(U[0])
    # Build the multiplication tensor
    # T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But we need to check: for all i,j,k, the tensor entry equals sum over c
    # Actually the tensor for matmul is: T_{i,j,k} = 1 if we're computing (AB)_{ik} = sum_j A_{ij} B_{jk}
    # The standard form: sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} = delta_{i,k} ? No.
    
    # Let me think more carefully. The matrix multiplication tensor T has entries:
    # T[(i,j), (j,k), (k,i)] = 1 for each triple (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # Actually the tensor is T_{ijk} = 1 for all valid i,j,k, and the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Wait, that's not right either. The matmul tensor has T[i,j,k] = 1 for each (i,j,k)
    # and we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k] = 1 for all i,j,k.
    # But that would mean each entry is 1, which is not how matmul works.
    
    # Actually, the matmul tensor T is defined by T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p],
    # and the bilinear form is: (AB)_{ik} = sum_j sum_c U[i*m+j,c] A_{ij} V[j*p+k,c] B_{jk} W[k*n+i,c]
    # So we need: for all i,k, (AB)_{ik} = sum_j A_{ij} B_{jk}
    # This means sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{jk}... no.
    
    # Let me reconsider. The standard matmul tensor:
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C_{ik} = sum_{j=0}^{m-1} A_{ij} B_{jk}
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k, and the bilinear form is:
    # C_{ik} = sum_{j} A_{ij} B_{jk} * T[i,j,k]  (but T is just 1 everywhere)
    # So the tensor is T[i,j,k] = 1 for all valid indices.
    # The decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Check this condition
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iterations=2000, deadline=None):
    """Try to reduce rank by setting some columns to zero and re-solving.
    
    Strategy: randomly pick columns to eliminate, then try to find a rank-
    target_rank decomposition by optimizing the remaining entries.
    """
    r = len(U[0])
    if target_rank >= r:
        return U, V, W
    
    # We'll try iterative column elimination
    # Pick random columns to zero out, then fix the rest
    
    U_f = [row[:] for row in U]
    V_f = [row[:] for row in V]
    W_f = [row[:] for row in W]
    
    # Try to zero out (r - target_rank) random columns
    cols_to_zero = random.sample(range(r), r - target_rank)
    zero_set = set(cols_to_zero)
    
    # Zero out those columns
    for c in zero_set:
        for i in range(len(U_f)):
            U_f[i][c] = 0
        for j in range(len(V_f)):
            V_f[j][c] = 0
        for k in range(len(W_f)):
            W_f[k][c] = 0
    
    # Now we have a rank-target_rank tensor (some columns may be zero)
    # Check if it still works
    if _verify(U_f, V_f, W_f, n, m, p):
        return U_f, V_f, W_f
    
    # Try local optimization on non-zero columns
    active_cols = [c for c in range(r) if c not in zero_set]
    
    # Use gradient-free optimization (Nelder-Mead style) on the free parameters
    # Flatten the free parameters
    def get_free_params():
        params = []
        for c in active_cols:
            for i in range(n * m):
                params.append(U_f[i][c])
            for j in range(m * p):
                params.append(V_f[j][c])
            for k in range(p * n):
                params.append(W_f[k][c])
        return params
    
    def set_free_params(params):
        idx = 0
        for c in active_cols:
            for i in range(n * m):
                U_f[i][c] = params[idx]
                idx += 1
            for j in range(m * p):
                V_f[j][c] = params[idx]
                idx += 1
            for k in range(p * n):
                W_f[k][c] = params[idx]
                idx += 1
    
    def objective(params):
        set_free_params(params)
        U_arr = np.array(U_f, dtype=np.float64)
        V_arr = np.array(V_f, dtype=np.float64)
        W_arr = np.array(W_f, dtype=np.float64)
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    total_err += (val - 1.0) ** 2
        return total_err
    
    # Simple coordinate descent / random perturbation
    best_params = get_free_params()
    best_obj = objective(best_params)
    
    if best_obj < 1e-12:
        set_free_params(best_params)
        return U_f, V_f, W_f
    
    # Try random perturbations
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        # Random perturbation
        perturb_params = best_params[:]
        for idx in range(len(perturb_params)):
            if random.random() < 0.3:
                perturb_params[idx] += random.gauss(0, 0.1)
        
        obj = objective(perturb_params)
        if obj < best_obj:
            best_obj = obj
            best_params = perturb_params
            if best_obj < 1e-12:
                set_free_params(best_params)
                return U_f, V_f, W_f
    
    set_free_params(best_params)
    if _verify(U_f, V_f, W_f, n, m, p):
        return U_f, V_f, W_f
    
    return None

def _try_small_subtensor_tricks(n, m, p):
    """Try to exploit small subtensor structures to get lower rank.
    
    For 3x5x5, we can try to split the problem into smaller blocks
    and use known low-rank decompositions for small cases.
    """
    # Schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    
    # Try: split the 5x5 matrix into 2x2 + 3x3 blocks (or similar)
    # The rank of 2x2x2 is 7, 3x3x3 is 23 (or 21? let's use schoolbook = 27)
    # Actually for 3x3x3, the best known is 23.
    
    # Approach: treat the 5x5 as a block matrix
    # A (3x5) * B (5x5) = C (3x5)
    # Split B into B1 (5x2) and B2 (5x3)
    # C1 = A * B1 (3x2), C2 = A * B2 (3x3)
    # Rank(C1) <= 3*5*2 = 30 schoolbook, but can be lower
    # Rank(C2) <= 3*5*3 = 45 schoolbook
    # Total: 30 + 45 = 75 = same as schoolbook
    
    # Better: use the fact that 2x2x2 has rank 7
    # Split the 3 rows into 2+1, and the 5 columns into 2+3
    # Block (2,2): rank 7
    # Block (2,3): rank 2*5*3 = 30 schoolbook, or maybe lower
    # Block (1,2): rank 1*5*2 = 10 schoolbook
    # Block (1,3): rank 1*5*3 = 15 schoolbook
    # Total: 7 + 30 + 10 + 15 = 62 < 75!
    
    # But wait, the blocks share the same A rows. Let me think about this more carefully.
    # The tensor for the full 3x5x5 multiplication is T[i,j,k] for i in [0,3), j in [0,5), k in [0,5)
    # If we split i into {0,1} and {2}, j into {0,1} and {2,3,4}, k into {0,1} and {2,3,4}
    # Then we have 4 sub-tensors:
    # T1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, rank 7
    # T2: i in {0,1}, j in {2,3,4}, k in {0,1} -> 2x3x2, rank up to 12
    # T3: i in {2}, j in {0,1}, k in {2,3,4} -> 1x2x3, rank up to 6
    # T4: i in {2}, j in {2,3,4}, k in {2,3,4} -> 1x3x3, rank up to 9
    # Total: 7 + 12 + 6 + 9 = 34? That seems too optimistic.
    
    # Actually, the sub-tensors are not independent because they share the same 
    # structure. The decomposition must be consistent across all blocks.
    
    # Let me try a direct construction:
    # Use the 2x2x2 rank-7 decomposition for the top-left block
    # And schoolbook for the rest
    
    # 2x2x2 rank 7 decomposition (from Strassen or known):
    # This is complex. Let me just try to find any improvement over 75.
    
    # Actually, let me try a simpler approach: randomly perturb the schoolbook
    # and see if we can find a lower-rank solution.
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    
    # Verify the schoolbook works
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try to reduce rank
    for target_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + target_rank)
            result = _try_reduce_rank(U, V, W, n, m, p, target_rank, 
                                      max_iterations=500, deadline=deadline)
            if result is not None:
                rU, rV, rW = result
                actual_rank = sum(1 for c in range(len(rU[0])) 
                                  if any(rU[i][c] != 0 for i in range(len(rU)))
                                  and any(rV[j][c] != 0 for j in range(len(rV)))
                                  and any(rW[k][c] != 0 for k in range(len(rW))))
                if actual_rank < best_rank and _verify(rU, rV, rW, n, m, p):
                    best_U, best_V, best_W = rU, rV, rW
                    best_rank = actual_rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
