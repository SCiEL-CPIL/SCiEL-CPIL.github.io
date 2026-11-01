import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _construct_strassen_56():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [0, 0, 0, 1], [1, 0, -1, 0], [0, -1, 0, 1],
        [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    blocks = []
    for i in range(4):
        b_top = [0*8+2*i, 0*8+2*i+1, 1*8+2*i, 1*8+2*i+1]
        b_bot = [2*8+2*i, 2*8+2*i+1, 3*8+2*i, 3*8+2*i+1]
        c_idx = [2*i*2+0, 2*i*2+1, (2*i+1)*2+0, (2*i+1)*2+1]
        
        blocks.append((A1_idx, b_top, c_idx))
        blocks.append((A2_idx, b_bot, c_idx))
        
    col_offset = 0
    for a_idx, b_idx, c_idx in blocks:
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[a_idx[r]][col] = U_strassen[f][r]
                V[b_idx[r]][col] = V_strassen[f][r]
                W[c_idx[r]][col] = W_strassen[r][f]
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}

def _find_rank_12_2x4x2():
    """
    Attempts to find a rank-12 decomposition for 2x4 * 4x2 multiplication.
    Tensor shape: (2, 4, 2).
    U: (2*4, 12)
    V: (4*2, 12)
    W: (2*2, 12)
    """
    n, m, p = 2, 4, 2
    rank = 12
    
    # Initialize with random values
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Define the target tensor
    # T[i, j, k] = 1 if j is the middle index connecting A[i,j] and B[j,k]
    # A is 2x4, B is 4x2, C is 2x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor entry T[i, j, k] = 1 if i,j,k are valid indices.
    # Actually, the tensor is defined by T[i, j, k] = delta_{i, i} delta_{k, k}? No.
    # The multiplication tensor T_{ijk} is 1 if the indices correspond to the product term.
    # For standard MM, T_{ijk} = 1 if j is the shared index.
    # Here i in 0..1, j in 0..3, k in 0..1.
    # T[i, j, k] = 1 for all valid i,j,k? No.
    # The tensor is a 3-way array where T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Since C[i,k] = sum_j A[i,j]B[j,k], the tensor has T[i,j,k]=1 for all i,j,k.
    # Wait, no. The tensor decomposition is T = sum u_c v_c w_c.
    # The target tensor T has entries T[i, j, k] = 1.
    # Let's verify: sum_c U[i,j,c] V[j,k,c] W[k,i,c] should equal 1 for all i,j,k?
    # No, the verification checks C[i,k] = sum_j A[i,j] B[j,k].
    # This is equivalent to sum_c (sum_j A[i,j] U[i,j,c]) (sum_l B[l,k] V[l,k,c]) W[k,i,c] = C[i,k].
    # This requires that sum_c U[i,j,c] V[j',k,c] W[k,i,c] = delta_{j,j'}? No.
    # It requires that the reconstructed tensor matches the multiplication tensor.
    # The multiplication tensor for 2x4*4x2 has T[i, j, k] = 1 if j is the index?
    # Actually, the standard tensor for MM(n,m,p) has T[i, j, k] = 1 if the indices align.
    # For 2x4x2, T[i, j, k] = 1 for all i,j,k?
    # Let's check the verify function.
    # C_computed[i,k] += factor_A[i] * factor_B[k] * W[k*n+i][c]
    # factor_A[i] = sum_j A[i,j] U[i*m+j][c]
    # factor_B[k] = sum_j V[j*p+k][c] B[j,k]
    # C_computed[i,k] = sum_c (sum_j A[i,j] U[i,j,c]) (sum_l B[l,k] V[l,k,c]) W[k,i,c]
    # = sum_j sum_l A[i,j] B[l,k] W[k,i,c] sum_c U[i,j,c] V[l,k,c]
    # We want this to equal sum_j A[i,j] B[j,k].
    # So we need W[k,i,c] sum_c U[i,j,c] V[l,k,c] = delta_{j,l} ?
    # No, the sum over c is inside.
    # We need: sum_c U[i,j,c] V[l,k,c] W[k,i,c] = delta_{j,l} * (something)?
    # Actually, if we set W[k,i,c] = 1 for all c, and sum_c U[i,j,c] V[l,k,c] = delta_{j,l}, that's not possible with rank 12 for 4x4 identity?
    # The condition is that the bilinear form in A and B matches the standard product.
    # This is equivalent to finding a decomposition of the identity tensor of size 4x4?
    # No, it's a specific tensor.
    
    # Let's just use the verify function to check.
    # We will minimize the error over random A, B.
    
    def error(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U_mat = U_flat.reshape(n*m, rank)
        V_mat = V_flat.reshape(m*p, rank)
        W_mat = W_flat.reshape(p*n, rank)
        
        err = 0.0
        # Use a few random matrices
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_exp = A @ B
            C_comp = np.zeros((n, p))
            for c in range(rank):
                fA = np.array([sum(A[i, j] * U_mat[i*m+j, c] for j in range(m)) for i in range(n)])
                fB = np.array([sum(V_mat[j*p+k, c] * B[j, k] for j in range(m)) for k in range(p)])
                for i in range(n):
                    for k in range(p):
                        C_comp[i, k] += fA[i] * fB[k] * W_mat[k*n+i, c]
            err += np.sum((C_comp - C_exp)**2)
        return err
    
    # Initial guess: Strassen-like?
    # Just use random start
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    res = minimize(error, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
    
    if res.success and res.fun < 1e-6:
        U_mat = res.x[:n*m*rank].reshape(n*m, rank)
        V_mat = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_mat = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Round to half-integers?
        # The task allows integer/half-integer.
        # We might need to snap to nearest half-integer if close.
        # For now, return the float solution, but the task says "integer/half-integer entries".
        # We must ensure the output satisfies this.
        # If the solution is not exactly half-integer, it might fail strict checks?
        # "exactly decompose" implies the values must be such that the identity holds.
        # If we use floats, it might be "exact" in float precision, but the contract says "integer/half-integer".
        # We should try to snap.
        
        U_snap = np.round(U_mat * 2) / 2
        V_snap = np.round(V_mat * 2) / 2
        W_snap = np.round(W_mat * 2) / 2
        
        # Verify snapped version
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        # Convert to list of lists of ints/floats
        U_final = [[float(x) for x in row] for row in U_list]
        V_final = [[float(x) for x in row] for row in V_list]
        W_final = [[float(x) for x in row] for row in W_list]
        
        if _verify(U_final, V_final, n=2, m=4, p=2, trials=100):
            return U_final, V_final, W_final
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a rank-48 solution (4x rank-12)
    sol_48 = None
    if time.time() < deadline - 30:
        sub_sol = _find_rank_12_2x4x2()
        if sub_sol is not None:
            U_sub, V_sub, W_sub = sub_sol
            n, m, p = 2, 4, 8
            rank = 48
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Map the 4 sub-problems
            # Sub-problem r corresponds to columns 2*r, 2*r+1 of B and C
            for r in range(4):
                offset = r * 12
                # U is same for all sub-problems?
                # A is shared. So U is the same for all 4 blocks?
                # In the 2x4x2 sub-problem, U is 2x4x12.
                # In the full 2x4x8 problem, A is 2x4.
                # So U is identical for all 4 blocks.
                for i in range(n * m):
                    for c in range(12):
                        U[i][offset + c] = U_sub[i][c]
                
                # V is different for each block
                # V_sub is 4x2x12.
                # V_full is 4x8x48.
                # Block r uses columns 2*r, 2*r+1 of V_full.
                for j in range(m):
                    for k in range(2):
                        idx_k = 2 * r + k
                        for c in range(12):
                            V[j * p + idx_k][offset + c] = V_sub[j * 2 + k][c]
                
                # W is different for each block
                # W_sub is 2x2x12.
                # W_full is 8x2x48 (p*n = 8*2=16).
                # Block r uses rows 2*r, 2*r+1 of W_full.
                for i in range(n):
                    for k in range(2):
                        idx_k = 2 * r + k
                        for c in range(12):
                            W[idx_k * n + i][offset + c] = W_sub[k * 2 + i][c]
                
            sol_48 = {"U": U, "V": V, "W": W}
            
            if _verify(sol_48["U"], sol_48["V"], sol_48["W"], trials=10):
                return sol_48
    
    # Fallback to 56
    sol = _construct_strassen_56()
    if _verify(sol["U"], sol["V"], sol["W"], trials=10):
        return sol
    else:
        n, m, p = 2, 4, 8
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
        return {"U": U, "V": V, "W": W}
