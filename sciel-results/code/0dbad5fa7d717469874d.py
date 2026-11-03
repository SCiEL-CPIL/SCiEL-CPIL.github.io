"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def construct():
    """Find a low-rank tensor decomposition for 4x4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = 4, 4, 5
    
    # The tensor T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum to T[i,j,k] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # Start with schoolbook as fallback
    rank_fallback = n * m * p  # 80
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using a structured approach
    # We'll use a Kronecker product based approach with Strassen-style block decomposition
    
    # Strategy: Split the problem into blocks and use Strassen multiplication
    # For 4x4 * 4x5, we can split as:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x? 
    # Actually let's do: A is 4x4, B is 4x5
    # Split A into 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    # Split B into 2x3 blocks of 2x?: B = [[B11, B12, B13], [B21, B22, B23]]
    # where B11, B21 are 2x2, B12, B22 are 2x2, B13, B23 are 2x1
    # This is getting complicated. Let me use a different approach.
    
    # Use a numerical optimization approach with ALS (Alternating Least Squares)
    # Start from a known good initialization and refine
    
    # The tensor entries are T[i,j,k] = 1 for the multiplication tensor
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Let's try rank r from low to high, using random initialization and ALS
    
    def build_tensor():
        """Build the target tensor T[i,j,k] = 1"""
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T
    
    T = build_tensor()
    
    def flatten_U(U):
        """U is (n*m, rank), flatten to (n, m, rank)"""
        return U.reshape(n, m, -1)
    
    def flatten_V(V):
        """V is (m*p, rank), flatten to (m, p, rank)"""
        return V.reshape(m, p, -1)
    
    def flatten_W(W):
        """W is (p*n, rank), flatten to (p, n, rank)"""
        return W.reshape(p, n, -1)
    
    def contract(U, V, W):
        """Contract the three factors to get the tensor"""
        Uf = flatten_U(U)
        Vf = flatten_V(V)
        Wf = flatten_W(W)
        # Result[i,j,k] = sum_r Uf[i,j,r] * Vf[j,k,r] * Wf[k,i,r]
        result = np.zeros((n, m, p), dtype=float)
        for r in range(U.shape[1]):
            # Uf[:,:,r] is (n,m), Vf[:,:,r] is (m,p), Wf[:,:,r] is (p,n)
            # We need result[i,j,k] += Uf[i,j,r] * Vf[j,k,r] * Wf[k,i,r]
            # This is an outer product in a sense
            for i in range(n):
                for j in range(m):
                    u_val = Uf[i, j, r]
                    if u_val != 0:
                        for k in range(p):
                            v_val = Vf[j, k, r]
                            if v_val != 0:
                                w_val = Wf[k, i, r]
                                result[i, j, k] += u_val * v_val * w_val
        return result
    
    def contract_fast(U, V, W):
        """Faster contraction using einsum"""
        Uf = U.reshape(n, m, U.shape[1])
        Vf = V.reshape(m, p, V.shape[1])
        Wf = W.reshape(p, n, W.shape[1])
        # result[i,j,k] = sum_r Uf[i,j,r] * Vf[j,k,r] * Wf[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', Uf, Vf, Wf)
        return result
    
    def als_solve_U(U, V, W, T, rank):
        """Solve for U given V and W"""
        Uf = U.reshape(n, m, rank)
        Vf = V.reshape(m, p, rank)
        Wf = W.reshape(p, n, rank)
        U_new = np.zeros((n * m, rank), dtype=float)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # For fixed r: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] + other terms
                    # We want to minimize sum_{i,j,k} (T[i,j,k] - sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'])^2
                    # With U[i,j,r] as variable:
                    # partial loss / partial U[i,j,r] = -2 * sum_k V[j,k,r] * W[k,i,r] * (T[i,j,k] - sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'])
                    # Setting to 0: U[i,j,r] * sum_k V[j,k,r]*W[k,i,r]^2 = sum_k V[j,k,r]*W[k,i,r]*T[i,j,k]
                    # Wait, that's not right. Let me redo.
                    # Loss = sum_{i,j,k} (T[i,j,k] - sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'])^2
                    # Let S[i,j,k] = sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'] (current contraction)
                    # Loss = sum (T - S)^2
                    # dLoss/dU[i,j,r] = -2 * sum_k (T[i,j,k] - S[i,j,k]) * V[j,k,r] * W[k,i,r]
                    # Set to 0: sum_k (T[i,j,k] - S[i,j,k]) * V[j,k,r] * W[k,i,r] = 0
                    # sum_k T[i,j,k]*V[j,k,r]*W[k,i,r] = sum_k S[i,j,k]*V[j,k,r]*W[k,i,r]
                    # But S includes U[i,j,r]*V[j,k,r]*W[k,i,r] term
                    # So: sum_k T[i,j,k]*V[j,k,r]*W[k,i,r] = U[i,j,r]*sum_k V[j,k,r]^2*W[k,i,r]^2 + sum_{r'!=r} U[i,j,r']*sum_k V[j,k,r']*V[j,k,r']*W[k,i,r]*W[k,i,r]
                    # This is getting complex. Let's use a linear system approach.
                    
                    # Actually, for fixed (i,j), and all r:
                    # T[i,j,:] = sum_r U[i,j,r] * (V[:,j,r] * W[:,i,r])  where V[:,j,r] is the vector over k
                    # Let's define B_r[k] = V[j,k,r] * W[k,i,r]
                    # Then T[i,j,k] = sum_r U[i,j,r] * B_r[k]
                    # This is a linear system: U[i,j,:] * B^T = T[i,j,:]
                    # where B is (rank, p) with B[r,k] = V[j,k,r]*W[k,i,r]
                    
                    B = np.zeros((rank, p), dtype=float)
                    for r_idx in range(rank):
                        for k in range(p):
                            B[r_idx, k] = Vf[j, k, r_idx] * Wf[k, i, r_idx]
                    # Now T[i,j,:] = U[i,j,:] @ B^T
                    # So U[i,j,:] = T[i,j,:] @ pinv(B^T)
                    # Or solve: B^T @ U[i,j,:].T = T[i,j,:].T
                    # U[i,j,:] = lstsq(B.T, T[i,j,:])
                    
                    if np.allclose(B, 0):
                        U_new[i * m + j, r] = Uf[i, j, r]  # Keep current value
                    else:
                        # Solve B.T @ x = T[i,j,:] for x = U[i,j,:]
                        # B.T is (p, rank), T[i,j,:] is (p,)
                        try:
                            solution, residuals, rank_b, sv = np.linalg.lstsq(B.T, T[i, j, :], rcond=None)
                            U_new[i * m + j, r] = solution[r]
                        except:
                            U_new[i * m + j, r] = Uf[i, j, r]
        return U_new
    
    def als_solve_V(U, V, W, T, rank):
        """Solve for V given U and W"""
        Uf = U.reshape(n, m, rank)
        Vf = V.reshape(m, p, rank)
        Wf = W.reshape(p, n, rank)
        V_new = np.zeros((m * p, rank), dtype=float)
        for r in range(rank):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # For fixed (j,k,r): T[i,j,k] = V[j,k,r] * sum_i U[i,j,r] * W[k,i,r] + other terms
                    # Let's define C_r[i] = U[i,j,r] * W[k,i,r]
                    # Then T[i,j,k] = sum_r V[j,k,r] * C_r[i]
                    # Linear system: V[j,k,:] @ C^T = T[:,j,k] where C is (rank, n)
                    
                    C = np.zeros((rank, n), dtype=float)
                    for r_idx in range(rank):
                        for i in range(n):
                            C[r_idx, i] = Uf[i, j, r_idx] * Wf[k, i, r_idx]
                    
                    if np.allclose(C, 0):
                        V_new[j * p + k, r] = Vf[j, k, r]
                    else:
                        try:
                            solution, residuals, rank_c, sv = np.linalg.lstsq(C.T, T[:, j, k, ], rcond=None)
                            V_new[j * p + k, r] = solution[r]
                        except:
                            V_new[j * p + k, r] = Vf[j, k, r]
        return V_new
    
    def als_solve_W(U, V, W, T, rank):
        """Solve for W given U and V"""
        Uf = U.reshape(n, m, rank)
        Vf = V.reshape(m, p, rank)
        Wf = W.reshape(p, n, rank)
        W_new = np.zeros((p * n, rank), dtype=float)
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # For fixed (k,i,r): T[i,j,k] = W[k,i,r] * sum_j U[i,j,r] * V[j,k,r] + other terms
                    # Let's define D_r[j] = U[i,j,r] * V[j,k,r]
                    # Then T[i,j,k] = sum_r W[k,i,r] * D_r[j]
                    # Linear system: W[k,i,:] @ D^T = T[i,:,k] where D is (rank, m)
                    
                    D = np.zeros((rank, m), dtype=float)
                    for r_idx in range(rank):
                        for j in range(m):
                            D[r_idx, j] = Uf[i, j, r_idx] * Vf[j, k, r_idx]
                    
                    if np.allclose(D, 0):
                        W_new[k * n + i, r] = Wf[k, i, r]
                    else:
                        try:
                            solution, residuals, rank_d, sv = np.linalg.lstsq(D.T, T[i, :, k, ], rcond=None)
                            W_new[k * n + i, r] = solution[r]
                        except:
                            W_new[k * n + i, r] = Wf[k, i, r]
        return W_new
    
    def compute_error(U, V, W):
        """Compute the error ||T - contract(U,V,W)||"""
        contracted = contract_fast(U, V, W)
        return np.linalg.norm(T - contracted)
    
    def round_to_halfint(arr):
        """Round array entries to nearest half-integer (0, 0.5, 1, 1.5, etc.)"""
        return np.round(arr * 2) / 2
    
    def is_valid(U, V, W):
        """Check if the decomposition is exact"""
        contracted = contract_fast(U, V, W)
        return np.allclose(T - contracted, 0, atol=1e-6)
    
    def try_rank(r, max_iters=50, n_restarts=3):
        """Try to find a valid decomposition with given rank"""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            current_error = compute_error(U, V, W)
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                U = als_solve_U(U, V, W, T, r)
                V = als_solve_V(U, V, W, T, r)
                W = als_solve_W(U, V, W, T, r)
                
                error = compute_error(U, V, W)
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-6:
                    break
        
        if best_error < 1e-6:
            # Found a solution, round to half-integers
            U_rounded = round_to_halfint(best_U)
            V_rounded = round_to_halfint(best_V)
            W_rounded = round_to_halfint(best_W)
            
            # Check if rounded version is still valid
            if is_valid(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded, True
            
            # If rounding broke it, try the unrounded version
            return best_U, best_V, best_W, True
        
        return best_U, best_V, best_W, False
    
    # Try increasing ranks
    best_solution = None
    best_rank = rank_fallback
    
    for r in range(15, min(30, rank_fallback)):
        if time.time() > deadline - 15:
            break
        
        U, V, W, valid = try_rank(r, max_iters=30, n_restarts=2)
        
        if valid:
            best_solution = (U, V, W)
            best_rank = r
            break
    
    if best_solution is None:
        # Use fallback
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    U, V, W = best_solution
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
