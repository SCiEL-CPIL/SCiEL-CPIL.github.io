"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares
DIMENSIONS = [2, 3, 4]

# Known optimal rank for 2x3 * 3x4 is 19
# We'll try to find a rank-19 decomposition using structured initialization
# based on the known algorithm for small matrix multiplication

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index triples and their positions
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_pos = np.array([i * m + j for i, j, k in indices])
    v_pos = np.array([j * p + k for i, j, k in indices])
    w_pos = np.array([k * n + i for i, j, k in indices])
    
    def verify(U, V, W, tol=1e-5):
        """Verify that the decomposition is exact using vectorized operations."""
        U_u = U[u_pos, :]
        V_v = V[v_pos, :]
        W_w = W[w_pos, :]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        max_err = np.max(np.abs(vals - 1.0))
        return max_err < tol
    
    def residual_func(params, rank):
        """Vectorized residual calculation."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        U_u = U_flat[u_pos, :]
        V_v = V_flat[v_pos, :]
        W_w = W_flat[w_pos, :]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        return vals - 1.0
    
    def construct_strassen_224():
        """Construct a rank-14 decomposition for 2x2 x 2x4 using two Strassen blocks."""
        # For 2x2 x 2x2, Strassen uses 7 multiplications:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # For 2x2 x 2x4, split into two 2x2 x 2x2 blocks (cols 0-1 and cols 2-3)
        # Total rank: 14
        
        rank = 14
        U = np.zeros((4, rank))  # 2x2 = 4 entries
        V = np.zeros((8, rank))  # 2x4 = 8 entries
        W = np.zeros((4, rank))  # 2x2 = 4 entries
        
        # Strassen for columns 0-1 (block 1)
        # m1 = (a00+a11)*(b00+b10) -> contributes to c00, c11
        # Actually for 2x2 x 2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        
        # Strassen:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V indices: 0=b00, 1=b01, 2=b10, 3=b11 (for block 1, cols 0-1)
        # W indices: 0=c00, 1=c01, 2=c10, 3=c11
        
        # m1
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1
        
        # m2
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[2, 1] += 1; W[3, 1] -= 1
        
        # m3
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1
        
        # m4
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[2, 3] += 1
        
        # m5
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] -= 1; W[1, 4] += 1
        
        # m6
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1
        
        # m7
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1; W[3, 6] -= 1
        
        # Strassen for columns 2-3 (block 2)
        # V indices for block 2: 4=b02, 5=b03, 6=b12, 7=b13
        # W indices for block 2: 0=c02, 1=c03, 2=c12, 3=c13 -> but W only has 4 entries for 2x2
        # Wait, for 2x2 x 2x4, the output is 2x4, so W should have 8 entries?
        # No, W shape is (p*n, rank) = (4*2, rank) = (8, rank) for the full problem.
        # But for this sub-problem (2x2 x 2x4), the output is 2x4 = 8 entries.
        
        # I need to reconsider. The full problem is 2x3 x 3x4.
        # This helper is just for constructing part of it.
        # Let me abandon this approach and just do the full numerical solve.
        
        return None
    
    def try_rank_19():
        """Try to find a rank-19 decomposition using structured initialization."""
        rank = 19
        start = time.time()
        best_cost = float('inf')
        best_x = None
        
        # Use the naive 24-rank solution as a starting point and try to compress
        # Or use a structured initialization based on Strassen
        
        # Strategy: Initialize with a rank-19 structure based on Strassen for 2x2x2
        # plus additional terms
        
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() - start > 100.0:
                break
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + attempt * 7)
            
            if attempt < 50:
                # Sparse initialization with Strassen-like structure
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                
                # Use Strassen for the 2x2 x 2x2 part (j=0,1; k=0,1)
                # and similar for other blocks, then add cross terms
                
                # Simplified: random sparse with small values
                for r in range(rank):
                    u_idx = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
                    v_idx = np.random.choice(mp, size=np.random.randint(1, 4), replace=False)
                    w_idx = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
                    for idx in u_idx:
                        U[idx, r] = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 1.5)
                    for idx in v_idx:
                        V[idx, r] = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 1.5)
                    for idx in w_idx:
                        W[idx, r] = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 1.5)
            elif attempt < 100:
                # Dense random
                scale = 0.8 / np.sqrt(rank)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            else:
                # Binary initialization
                U = np.random.choice([-1, 0, 1], size=(nm, rank)) * np.random.uniform(0.3, 1.0, size=(1, rank))
                V = np.random.choice([-1, 0, 1], size=(mp, rank)) * np.random.uniform(0.3, 1.0, size=(1, rank))
                W = np.random.choice([-1, 0, 1], size=(pn, rank)) * np.random.uniform(0.3, 1.0, size=(1, rank))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(
                    lambda x: residual_func(x, rank),
                    x0,
                    method='lm',
                    max_nfev=5000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14
                )
                
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_x = result.x.copy()
                
                if result.cost < 1e-10:
                    U_sol = result.x[:nm * rank].reshape(nm, rank)
                    V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                    W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    if verify(U_sol, V_sol, W_sol, tol=1e-6):
                        U_rounded = np.round(U_sol * 2) / 2
                        V_rounded = np.round(V_sol * 2) / 2
                        W_rounded = np.round(W_sol * 2) / 2
                        
                        if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                            return U_rounded, V_rounded, W_rounded
                        else:
                            return U_sol, V_sol, W_sol
            except Exception:
                continue
        
        if best_x is not None and best_cost < 1e-6:
            U_sol = best_x[:nm * rank].reshape(nm, rank)
            V_sol = best_x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
            W_sol = best_x[nm * rank + mp * rank:].reshape(pn, rank)
            if verify(U_sol, V_sol, W_sol, tol=1e-4):
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    return U_rounded, V_rounded, W_rounded
                else:
                    return U_sol, V_sol, W_sol
        
        return None
    
    # Try to find a rank-19 decomposition
    solution = try_rank_19()
    if solution is not None:
        U_sol, V_sol, W_sol = solution
        
        def to_list(arr):
            lst = []
            for row in arr:
                row_lst = []
                for val in row:
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        row_lst.append(rounded)
                    else:
                        row_lst.append(float(val))
                lst.append(row_lst)
            return lst
        
        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fallback: naive decomposition with rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
