"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if U, V, W form a valid decomposition."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _find_2x3x3_decomposition(target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition for 2x3x3.
    Uses a simple randomized search with local optimization if needed.
    For now, we try to construct one using known structures or random restarts.
    """
    n2, m2, p2 = 2, 3, 3
    
    # We want T[i,j,k] = 1 decomposed into rank r.
    # T = sum u_t v_t w_t
    # This is equivalent to finding rank-r matrices U, V, W such that
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Strategy: Fix W to be a matrix with simple structure.
    # Or use random initialization and ALS.
    
    # Given time constraints, we will try a few random initializations.
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W with small random values
        U = np.random.randn(n2*m2, target_rank) * 0.1
        V = np.random.randn(m2*p2, target_rank) * 0.1
        W = np.random.randn(p2*n2, target_rank) * 0.1
        
        # Simple gradient descent step? Too slow.
        # Instead, we can try to solve for W given U, V.
        # For each (i,k), we have:
        # sum_j U[i*m2+j, :] * V[j*p2+k, :] = W[k*n2+i, :]
        # This is linear in W.
        
        # Let's solve for W.
        W_sol = np.zeros((p2*n2, target_rank))
        for i in range(n2):
            for k in range(p2):
                # b = sum_j U[i*m2+j, :] * V[j*p2+k, :]
                b = np.zeros(target_rank)
                for j in range(m2):
                    b += U[i*m2+j, :] * V[j*p2+k, :]
                W_sol[k*n2+i, :] = b
                
        # Now check if this W_sol works.
        # We need sum_r U[i,j,r] V[j,k,r] W_sol[k,i,r] = 1
        # But W_sol was defined as sum_j U[i,j,r] V[j,k,r].
        # So the condition becomes sum_j U[i,j,r] V[j,k,r] (sum_{j'} U[i,j',r] V[j',k,r]) = 1?
        # No, W_sol is the *result* of the sum.
        # The condition is that the *tensor* equals 1.
        # If we set W = W_sol, then sum_r U V W = sum_r U V (sum U V) = sum (UV)^2.
        # This is not 1.
        
        # The correct approach:
        # We need U, V, W such that the tensor contraction is 1.
        # This is a nonlinear system.
        
        # Let's try a different strategy:
        # Assume W is the identity-like matrix.
        # W[k*n2+i, r] = 1 if r == k*n2+i, else 0.
        # Then sum_r U[i,j,r] V[j,k,r] delta_{r, k*n2+i} = U[i,j, k*n2+i] V[j,k, k*n2+i] = 1.
        # This requires U[i,j, k*n2+i] V[j,k, k*n2+i] = 1 for all i,j,k.
        # This implies U and V are sparse.
        # If we set U[i,j, k*n2+i] = 1 and V[j,k, k*n2+i] = 1, then the product is 1.
        # But this uses rank n2*p2 = 6.
        # Let's check if this works.
        
        if target_rank >= 6:
            U_try = np.zeros((n2*m2, target_rank))
            V_try = np.zeros((m2*p2, target_rank))
            W_try = np.zeros((p2*n2, target_rank))
            
            # Map (k, i) to rank index r = k*n2 + i
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        r = k * n2 + i
                        if r < target_rank:
                            U_try[i*m2+j, r] = 1
                            V_try[j*p2+k, r] = 1
                            W_try[k*n2+i, r] = 1
                            
            if _verify_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist(), n2, m2, p2):
                return U_try.tolist(), V_try.tolist(), W_try.tolist()
                
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 36
    
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition using the 2x3x3 block structure
    # If we find a rank-r decomposition for 2x3x3, we can use it for 2x3x6 with rank 2r.
    
    # Try ranks for 2x3x3: 6, 7, ..., 12
    for r_233 in range(6, 13):
        if time.time() > deadline:
            break
            
        decomp = _find_23x3_decomposition(r_233, deadline)
        if decomp:
            U_233, V_233, W_233 = decomp
            # Construct 2x3x6 decomposition
            # U: (6, 2r), V: (18, 2r), W: (12, 2r)
            rank_total = 2 * r_233
            
            U_new = [[0] * rank_total for _ in range(n * m)]
            V_new = [[0] * rank_total for _ in range(m * p)]
            W_new = [[0] * rank_total for _ in range(p * n)]
            
            # First block (k=0,1,2)
            for i in range(n):
                for j in range(m):
                    for t in range(r_233):
                        U_new[i*m+j][t] = U_233[i*m2+j][t] if False else U_233[i*3+j][t]
                        
            # Wait, indices.
            # U_233 is (6, r). U_new is (6, 2r).
            # For k in 0..2, use U_233 columns 0..r-1.
            # For k in 3..5, use U_233 columns r..2r-1 (same values).
            
            for i in range(n):
                for j in range(m):
                    for t in range(r_233):
                        U_new[i*m+j][t] = U_233[i*3+j][t]
                        U_new[i*m+j][t + r_233] = U_233[i*3+j][t]
                        
            # V: (18, 2r). Rows 0..8 are for B1 (j, k in 0..2). Rows 9..17 for B2 (j, k in 3..5).
            for j in range(m):
                for k in range(3):
                    for t in range(r_233):
                        V_new[j*6+k][t] = V_233[j*3+k][t]
                        
                for k in range(3, 6):
                    for t in range(r_233):
                        V_new[j*6+k][t + r_233] = V_233[j*3+(k-3)][t]
                        
            # W: (12, 2r). Rows 0..5 for C1 (k, i in 0..2). Rows 6..11 for C2 (k, i in 3..5).
            for k in range(3):
                for i in range(n):
                    for t in range(r_233):
                        W_new[k*n+i][t] = W_233[k*2+i][t]
                        
            for k in range(3, 6):
                for i in range(n):
                    for t in range(r_233):
                        W_new[k*n+i][t + r_233] = W_233[(k-3)*2+i][t]
                        
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                if rank_total < best_rank:
                    best_rank = rank_total
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    
    return {"U": best_U, "V": best_V, "W": best_W}

def _find_23x3_decomposition(target_rank, deadline):
    # Wrapper for _find_2x3x3_decomposition with correct name
    return _find_2x3x3_decomposition(target_rank, deadline)
# EVOLVE-BLOCK-END
