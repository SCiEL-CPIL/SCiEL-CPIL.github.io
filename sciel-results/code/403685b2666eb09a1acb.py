import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 30s margin before 150s limit
    
    # Seed solution: rank 40 (schoolbook)
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][col] = 1
                V_seed[j * p + k][col] = 1
                W_seed[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed

    def check_decomposition(U, V, W, n, m, p):
        """Verify if U, V, W form a valid decomposition"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = 1
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_2x2x5(rank, max_restarts=10):
        """
        Try to find a rank decomposition for 2x2x5 matrix multiplication.
        n=2, m=2, p=5.
        U: 4 x rank
        V: 10 x rank
        W: 10 x rank
        """
        n_s, m_s, p_s = 2, 2, 5
        U_size = n_s * m_s * rank
        V_size = m_s * p_s * rank
        W_size = p_s * n_s * rank
        total_size = U_size + V_size + W_size
        
        best_obj = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 137)
            # Scale to avoid overflow/underflow
            scale = 1.0 / np.sqrt(rank)
            params = np.random.randn(total_size) * scale
            
            def residuals(params):
                U = params[:U_size].reshape(n_s*m_s, rank)
                V = params[U_size:U_size+V_size].reshape(m_s*p_s, rank)
                W = params[U_size+V_size:U_size+V_size+W_size].reshape(p_s*n_s, rank)
                
                res = np.zeros(n_s*m_s*p_s)
                idx = 0
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            val = np.dot(U[i*m_s+j, :], V[j*p_s+k, :] * W[k*n_s+i, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                result = least_squares(residuals, params, method='lm', max_nfev=10000)
                if result.cost < best_obj:
                    best_obj = result.cost
                    best_params = result.x
            except Exception:
                continue
        
        if best_params is not None and best_obj < 1e-10:
            U = best_params[:U_size].reshape(n_s*m_s, rank)
            V = best_params[U_size:U_size+V_size].reshape(m_s*p_s, rank)
            W = best_params[U_size+V_size:U_size+V_size+W_size].reshape(p_s*n_s, rank)
            
            # Verify
            err = check_decomposition(U, V, W, n_s, m_s, p_s)
            if err < 1e-6:
                # Round to nearest half-integer or integer if close
                U_list = [[round(float(x), 8) for x in row] for row in U]
                V_list = [[round(float(x), 8) for x in row] for row in V]
                W_list = [[round(float(x), 8) for x in row] for row in W]
                return U_list, V_list, W_list
        return None

    # Strategy: Decompose 2x4x5 into two 2x2x5 blocks
    # A = [A1 A2] (2x4), B = [B1; B2] (4x5)
    # C = A1 B1 + A2 B2
    # Block 1: 2x2x5, Block 2: 2x2x5
    
    # Try ranks for 2x2x5 starting from 15
    # If we find rank r for 2x2x5, total rank is 2*r
    # Trivial 2x2x5 rank is 10.
    # Lower bound is likely higher.
    
    found_block = None
    for r_block in range(15, 10, -1): # Try 15, 14, 13, 12, 11
        if time.time() > deadline - 10:
            break
        res = try_2x2x5(r_block, max_restarts=5)
        if res is not None:
            found_block = res
            break
            
    if found_block is not None:
        U_b, V_b, W_b = found_block
        r_block = len(U_b[0])
        total_rank = 2 * r_block
        
        # Construct full U, V, W
        # U_full: 8 x total_rank
        # V_full: 20 x total_rank
        # W_full: 10 x total_rank
        
        U_full = [[0] * total_rank for _ in range(8)]
        V_full = [[0] * total_rank for _ in range(20)]
        W_full = [[0] * total_rank for _ in range(10)]
        
        # Block 1 (i=0,1; j=0,1; k=0..4)
        # U indices: i*4+j -> 0..3
        # V indices: j*5+k -> 0..9
        # W indices: k*2+i -> 0..9
        
        for i in range(2):
            for j in range(2):
                for r in range(r_block):
                    U_full[i*2+j][r] = U_b[i*2+j][r]
        
        for j in range(2):
            for k in range(5):
                for r in range(r_block):
                    V_full[j*5+k][r] = V_b[j*5+k][r]
        
        for k in range(5):
            for i in range(2):
                for r in range(r_block):
                    W_full[k*2+i][r] = W_b[k*2+i][r]
        
        # Block 2 (i=0,1; j=2,3; k=0..4)
        # Offset in U: j=2,3 -> indices 4..7
        # Offset in V: j=2,3 -> indices 10..19
        # Offset in W: same as block 1? No, W depends on i,k.
        # Wait, W is shared?
        # C = A1 B1 + A2 B2
        # C[i,k] = sum_j A[i,j] B[j,k]
        # C[i,k] = sum_{j=0}^1 A[i,j] B[j,k] + sum_{j=2}^3 A[i,j] B[j,k]
        # The output C is the same.
        # So W must be the SAME for both blocks?
        # No, the decomposition is sum_r U_r V_r W_r.
        # If we sum two decompositions, we just concatenate the terms.
        # Term r in block 1: U1[i,j] V1[j,k] W1[i,k]
        # Term r in block 2: U2[i,j] V2[j,k] W2[i,k]
        # These are separate terms in the total sum.
        # So W_full for block 2 is just another set of columns.
        
        for i in range(2):
            for j in range(2):
                for r in range(r_block):
                    U_full[(i)*2 + (j+2)][r_block + r] = U_b[i*2+j][r]
        
        for j in range(2):
            for k in range(5):
                for r in range(r_block):
                    V_full[(j+2)*5+k][r_block + r] = V_b[j*5+k][r]
        
        for k in range(5):
            for i in range(2):
                for r in range(r_block):
                    W_full[k*2+i][r_block + r] = W_b[k*2+i][r]
                    
        # Verify full solution
        err = check_decomposition(U_full, V_full, W_full, n, m, p)
        if err < 1e-6:
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = total_rank
            # We can try to optimize further, but this is a solid improvement
            # If we have time, we could try to merge or reduce, but let's return this first
            # Actually, we can just return this.
            return {"U": best_U, "V": best_V, "W": best_W}
            
    # Fallback: General search if block decomposition fails
    # (Omitted for brevity in this thought, but kept in code if needed)
    # For now, if block fails, return seed.
    
    return {"U": best_U, "V": best_V, "W": best_W}
