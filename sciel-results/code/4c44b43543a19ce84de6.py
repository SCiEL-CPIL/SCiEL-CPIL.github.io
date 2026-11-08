import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    deadline = time.time() + 130  # Leave 20s margin for safety
    
    def verify(U, V, W):
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T, T_comp, atol=1e-10)
    
    def try_rank(rank, seed, max_iter=2000):
        """Try to find a rank-r decomposition using iterative optimization."""
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Use alternating least squares / gradient descent approach
        # The objective is sum_{i,j,k} (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
        
        best_loss = np.inf
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each i, minimize sum_{j,k} (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
            # This is a linear system in U[i,:]
            # Let's use gradient descent with adaptive step size
            
            # Compute residual
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = T - T_comp
            loss = np.sum(residual**2)
            
            if loss < best_loss:
                best_loss = loss
                if best_loss < 1e-12:
                    return U, V, W, True
            
            # Gradient with respect to U: 
            # d/dU[i,r] = -2 * sum_{j,k} residual[i,j,k] * V[j,r] * W[k,r]
            grad_U = -2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
            grad_V = -2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
            grad_W = -2 * np.einsum('ijk,ir,jr->kr', residual, U, V)
            
            # Adaptive step size
            step_U = 0.01 / (1 + best_loss)
            step_V = 0.01 / (1 + best_loss)
            step_W = 0.01 / (1 + best_loss)
            
            U -= step_U * grad_U
            V -= step_V * grad_V
            W -= step_W * grad_W
            
            # Normalize to prevent divergence
            U_norm = np.linalg.norm(U)
            V_norm = np.linalg.norm(V)
            W_norm = np.linalg.norm(W)
            if U_norm > 10: U /= U_norm * 0.1
            if V_norm > 10: V /= V_norm * 0.1
            if W_norm > 10: W /= W_norm * 0.1
        
        return U, V, W, verify(U, V, W)
    
    # First, try to build a Strassen-based rank-28 solution as fallback
    def build_strassen_rank28():
        rank = 28
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Strassen for 2x2: 7 multiplications
        # We have 2x4 * 4x4 = (2x2 blocks) * (4x4 blocks)
        # Split into 4 quadrants of 2x2, each needing 7 mults = 28 total
        
        strassen_7 = [
            # Each entry: (U_factors, V_factors, W_factors)
            # Factor format: list of (index, coefficient)
            # U index: (block_row, col) where block_row in [0,1], col in [0,1] for 2x2
            # V index: (row, block_col) where row in [0,1], block_col in [0,1] for 2x2
            # W index: (row, col) for 2x2 output
            
            # M1 = (A11+A22)*(B12-B22) -> C12 += M1, C22 -= M1
            ([(0,0,1),(1,1,1)], [(0,1,1),(1,0,-1)], [(0,1,1),(1,1,-1)]),
            # M2 = (A21+A22)*B11 -> C21 += M2, C22 += M2  
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
            # M3 = A11*(B11-B21) -> C11 += M3, C12 += M3
            ([(0,0,1)], [(0,0,1),(1,0,-1)], [(0,0,1),(0,1,1)]),
            # M4 = A22*(B22-B12) -> C22 += M4, C21 += M4
            ([(1,1,1)], [(1,1,1),(0,1,-1)], [(1,1,1),(1,0,1)]),
            # M5 = (A11+A12)*B22 -> C11 += M5, C12 += M5
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
            # M6 = (A21-A11)*(B11+B12) -> C21 += M6, C11 -= M6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(0,0,-1)]),
            # M7 = (A12-A22)*(B21+B22) -> C12 += M7, C22 -= M7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1),(1,1,-1)]),
        ]
        
        # For 2x4 * 4x4:
        # A is 2x4: split into 2 columns of 2x2: A[:, 0:2], A[:, 2:4]
        # B is 4x4: split into 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
        # C is 2x4: split into 2 columns of 2x2: C[:, 0:2], C[:, 2:4]
        
        # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
        # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
        
        # Each 2x2*2x2 product uses 7 Strassen multiplications
        # Total: 4 products * 7 = 28
        
        def map_u_idx(block_col, local_r, local_c):
            """Map Strassen local index to global U index.
            block_col: which 2x2 block of A (0 or 1), so columns 2*block_col and 2*block_col+1
            local_r: row in 2x2 (0 or 1)
            local_c: col in 2x2 (0 or 1)
            """
            global_r = local_r
            global_c = 2 * block_col + local_c
            return global_r * m + global_c
        
        def map_v_idx(row_block, col_block, local_r, local_c):
            """Map Strassen local index to global V index.
            row_block: which row block of B (0 or 1), so rows 2*row_block and 2*row_block+1
            col_block: which col block of B (0 or 1), so cols 2*col_block and 2*col_block+1
            local_r: row in 2x2 (0 or 1)
            local_c: col in 2x2 (0 or 1)
            """
            global_r = 2 * row_block + local_r
            global_c = 2 * col_block + local_c
            return global_r * p + global_c
        
        def map_w_idx(col_block, local_r, local_c):
            """Map Strassen local index to global W index (trace convention).
            col_block: which col block of C (0 or 1), so cols 2*col_block and 2*col_block+1
            local_r: row in 2x2 (0 or 1)
            local_c: col in 2x2 (0 or 1)
            Trace convention: W[k*n+i, r] where k is col, i is row
            So index = local_c_block_col * n + local_r
            """
            global_k = 2 * col_block + local_c
            global_i = local_r
            return global_k * n + global_i
        
        # Four 2x2*2x2 products:
        # Product 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        # Product 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        # Product 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        # Product 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
        
        products = [
            (0, 0, 0, 0),  # A_block=0, B_row=0, B_col=0, C_col=0
            (1, 1, 0, 0),  # A_block=1, B_row=1, B_col=0, C_col=0
            (0, 0, 1, 1),  # A_block=0, B_row=0, B_col=1, C_col=1
            (1, 1, 1, 1),  # A_block=1, B_row=1, B_col=1, C_col=1
        ]
        
        for prod_idx, (a_block, b_row, b_col, c_col) in enumerate(products):
            base_col = prod_idx * 7
            for s_idx, (u_facs, v_facs, w_facs) in enumerate(strassen_7):
                col = base_col + s_idx
                for (r, c, coeff) in u_facs:
                    U[map_u_idx(a_block, r, c), col] += coeff
                for (r, c, coeff) in v_facs:
                    V[map_v_idx(b_row, b_col, r, c), col] += coeff
                for (r, c, coeff) in w_facs:
                    W[map_w_idx(c_col, r, c), col] += coeff
        
        if verify(U, V, W):
            return U, V, W
        
        return None
    
    # Try to find lower rank solutions
    best_result = None
    
    # First get the rank-28 baseline
    strassen_result = build_strassen_rank28()
    if strassen_result is not None:
        best_result = strassen_result
    
    # Try ranks 24 through 27 with multiple seeds
    for rank in [24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            
            U, V, W, success = try_rank(rank, seed, max_iter=500)
            if success:
                found = True
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    if best_result is None:
        # Should not happen, but fallback
        return {"U": [], "V": [], "W": []}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
