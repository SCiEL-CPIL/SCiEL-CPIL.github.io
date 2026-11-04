import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS

def _validate(U, V, W):
    """Check if U, V, W form a valid rank-R tensor decomposition."""
    if not U or not V or not W:
        return False
    
    rank = len(V[0]) if V else 0
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if s != 1:
                    return False
    return True

def _build_block_55():
    """
    Construct the Rank 55 decomposition using 5 blocks of (2x3x2).
    Each block uses Rank 11.
    """
    n, m, p = DIMENSIONS
    R = 55
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for g in range(5):
        base_rank = 11 * g
        
        # --- Part 1: A1 * B1 (2x2 * 2x2) -> Rank 7 (Strassen) ---
        # Indices in U (A): 0:A00, 1:A01, 2:A02, 3:A10, 4:A11, 5:A12
        # Indices in V (B slice 2 cols): 
        #   The code maps B columns 2g, 2g+1 to local indices.
        #   V is flattened (3x10).
        #   For block g, B cols are 2g, 2g+1.
        #   V row 0, col 2g -> index 2g
        #   V row 1, col 2g -> index 10+2g
        #   V row 2, col 2g -> index 20+2g
        #   Similarly for 2g+1.
        
        # The previous code used specific indices. Let's replicate the logic exactly.
        c = base_rank
        U[0][c] += 1   # A00
        U[4][c] += 1   # A11
        V[2*g][c] += 1   # B[0, 2g]
        V[10+2*g+1][c] += 1 # B[1, 2g+1]
        W[4*g][c] += 1   # C[2g, 0] (W index k*n+i. k=2g, i=0 -> 4g)
        W[4*g+3][c] += 1 # C[2g+1, 1] (k=2g+1, i=1 -> 4g+1+3=4g+4? No. W index k*n+i. n=2.
                          # k=2g, i=0 -> 4g.
                          # k=2g+1, i=1 -> 2(2g+1)+1 = 4g+3. Correct.)
        
        c = base_rank + 1
        U[3][c] += 1   # A10
        U[4][c] += 1   # A11
        V[2*g][c] += 1   # B[0, 2g]
        W[4*g+2][c] += 1 # C[2g+1, 0] (k=2g+1, i=0 -> 4g+2)
        
        c = base_rank + 2
        U[0][c] += 1   # A00
        V[10+2*g][c] += 1   # B[1, 2g]
        V[10+2*g+1][c] -= 1 # B[1, 2g+1]
        W[4*g+1][c] += 1 # C[2g, 1] (k=2g, i=1 -> 4g+1)
        W[4*g+3][c] += 1 # C[2g+1, 1]
        
        c = base_rank + 3
        U[4][c] += 1   # A11
        V[2*g+1][c] += 1   # B[0, 2g+1]
        V[2*g][c] -= 1   # B[0, 2g]
        W[4*g][c] += 1   # C[2g, 0]
        W[4*g+2][c] += 1 # C[2g+1, 0]
        
        c = base_rank + 4
        U[0][c] += 1   # A00
        U[1][c] += 1   # A01
        V[10+2*g+1][c] += 1 # B[1, 2g+1]
        W[4*g][c] -= 1   # C[2g, 0]
        W[4*g+1][c] += 1 # C[2g, 1]
        
        c = base_rank + 5
        U[3][c] += 1   # A10
        U[0][c] -= 1   # A00
        V[2*g][c] += 1   # B[0, 2g]
        V[2*g+1][c] += 1 # B[0, 2g+1]
        W[4*g+3][c] += 1 # C[2g+1, 1]
        
        c = base_rank + 6
        U[1][c] += 1   # A01
        U[4][c] -= 1   # A11
        V[10+2*g][c] += 1 # B[1, 2g]
        V[10+2*g+1][c] += 1 # B[1, 2g+1]
        W[4*g][c] += 1   # C[2g, 0]
        
        # --- Part 2: A2 * B2 (2x1 * 1x2) -> Rank 4 ---
        # Uses A[0,2] (index 2) and A[1,2] (index 5)
        # And B[2, 2g], B[2, 2g+1]
        
        c = base_rank + 7
        U[2][c] += 1 # A02
        V[20+2*g][c] += 1 # B[2, 2g]
        W[4*g][c] += 1 # C[2g, 0]
        
        c = base_rank + 8
        U[2][c] += 1 # A02
        V[20+2*g+1][c] += 1 # B[2, 2g+1]
        W[4*g+1][c] += 1 # C[2g, 1]
        
        c = base_rank + 9
        U[5][c] += 1 # A12
        V[20+2*g][c] += 1 # B[2, 2g]
        W[4*g+2][c] += 1 # C[2g+1, 0]
        
        c = base_rank + 10
        U[5][c] += 1 # A12
        V[20+2*g+1][c] += 1 # B[2, 2g+1]
        W[4*g+3][c] += 1 # C[2g+1, 1]

    return U, V, W

def _try_find_rank_20_for_234(deadline):
    """
    Attempt to find a rank 20 decomposition for 2x3x4 matrix multiplication.
    This would allow coupling two 2x3x2 blocks into one 2x3x4 block with rank 20 (vs 22).
    """
    # This is a heuristic search. 
    # We parameterize U (2x3x20), V (3x4x20), W (4x2x20).
    # Target tensor T_{i,j,k} = 1 for all i,j,k.
    # Dimensions: n=2, m=3, p=4.
    # We use random initialization with half-integers and local search.
    
    # For simplicity and speed, we will try a small number of random restarts
    # with a gradient-free method (Nelder-Mead) on the squared error,
    # then snap to half-integers and validate.
    
    from scipy.optimize import minimize
    
    n_sub, m_sub, p_sub = 2, 3, 4
    R_sub = 20
    
    size_U = n_sub * m_sub * R_sub
    size_V = m_sub * p_sub * R_sub
    size_W = p_sub * n_sub * R_sub
    total_size = size_U + size_V + size_W
    
    def objective(params):
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U_mat = U_flat.reshape(n_sub*m_sub, R_sub)
        V_mat = V_flat.reshape(m_sub*p_sub, R_sub)
        W_mat = W_flat.reshape(p_sub*n_sub, R_sub)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        error = np.sum((T - 1.0)**2)
        return error
    
    best_result = None
    best_error = float('inf')
    
    num_restarts = 10 # Keep low due to time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5: break
        
        params = np.random.randn(total_size) * 0.5
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 200, 'maxfev': 200})
            if res.fun < best_error:
                best_error = res.fun
                if res.fun < 1e-4:
                    best_result = res.x
                    break
        except Exception:
            pass
            
    if best_result is None:
        return None
        
    U_flat = best_result[:size_U]
    V_flat = best_result[size_U:size_U+size_V]
    W_flat = best_result[size_U+size_V:]
    
    U_mat = U_flat.reshape(n_sub*m_sub, R_sub)
    V_mat = V_flat.reshape(m_sub*p_sub, R_sub)
    W_mat = W_flat.reshape(p_sub*n_sub, R_sub)
    
    # Snap to half-integers
    U_snap = np.round(U_mat * 2) / 2
    V_snap = np.round(V_mat * 2) / 2
    W_snap = np.round(W_mat * 2) / 2
    
    U_list = [[Fraction(int(x*2), 2) for x in row] for row in U_snap]
    V_list = [[Fraction(int(x*2), 2) for x in row] for row in V_snap]
    W_list = [[Fraction(int(x*2), 2) for x in row] for row in W_snap]
    
    # Validate for 2x3x4
    # Reuse _validate logic but with local dimensions
    def _validate_sub(U, V, W, n, m, p):
        rank = len(V[0])
        U_arr = np.array(U, dtype=object)
        V_arr = np.array(V, dtype=object)
        W_arr = np.array(W, dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if s != 1:
                        return False
        return True

    if _validate_sub(U_list, V_list, W_list, n_sub, m_sub, p_sub):
        return U_list, V_list, W_list
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Get the baseline Rank 55 solution
    U_base, V_base, W_base = _build_block_55()
    
    if _validate(U_base, V_base, W_base):
        # 2. Try to improve by coupling blocks
        # We have 5 blocks. We can couple 4 of them into 2 pairs of 2x3x4.
        # If we find a rank 20 solution for 2x3x4, we can save 2 ranks per pair.
        # Total rank = 2 * 20 + 1 * 11 = 51.
        
        coupled_solution = _try_find_rank_20_for_234(deadline)
        
        if coupled_solution:
            U_c, V_c, W_c = coupled_solution
            # Construct the full 2x3x10 decomposition
            # Blocks 0 and 1 (cols 0,1 and 2,3 of B) use the coupled 2x3x4 solution.
            # Blocks 2 and 3 (cols 4,5 and 6,7) use another instance? 
            # The coupled solution is for specific indices. 
            # We need to map the indices correctly.
            
            # The coupled solution U_c, V_c, W_c is for n=2, m=3, p=4.
            # U_c shape: 6x20. V_c shape: 12x20. W_c shape: 8x20.
            
            # Full U: 6x51.
            # Full V: 30x51.
            # Full W: 20x51.
            
            R_full = 51
            U_full = [[0] * R_full for _ in range(6)]
            V_full = [[0] * R_full for _ in range(30)]
            W_full = [[0] * R_full for _ in range(20)]
            
            # Pair 1: B cols 0,1,2,3. C cols 0,1,2,3.
            # Map U_c to U_full cols 0..19.
            for i in range(6):
                for c in range(20):
                    U_full[i][c] = U_c[i][c]
                    
            # Map V_c to V_full.
            # V_c is for B[0..2, 0..3].
            # V_full is for B[0..2, 0..9].
            # V_c index j*4 + k. V_full index j*10 + k.
            for j in range(3):
                for k in range(4):
                    idx_c = j * 4 + k
                    idx_f = j * 10 + k
                    for c in range(20):
                        V_full[idx_f][c] = V_c[idx_c][c]
                        
            # Map W_c to W_full.
            # W_c is for C[0..3, 0..1].
            # W_full is for C[0..9, 0..1].
            # W_c index k*2 + i. W_full index k*2 + i.
            for k in range(4):
                for i in range(2):
                    idx_c = k * 2 + i
                    idx_f = k * 2 + i
                    for c in range(20):
                        W_full[idx_f][c] = W_c[idx_c][c]
                        
            # Pair 2: B cols 4,5,6,7. C cols 4,5,6,7.
            # We need to shift the indices for the second instance.
            # Rank offset: 20.
            for i in range(6):
                for c in range(20):
                    U_full[i][20 + c] = U_c[i][c]
                    
            for j in range(3):
                for k in range(4):
                    idx_c = j * 4 + k
                    # B cols 4,5,6,7. So k_local in 0..3 maps to k_global = 4 + k_local.
                    idx_f = j * 10 + (4 + k)
                    for c in range(20):
                        V_full[idx_f][20 + c] = V_c[idx_c][c]
                        
            for k in range(4):
                for i in range(2):
                    idx_c = k * 2 + i
                    # C cols 4,5,6,7. So k_local in 0..3 maps to k_global = 4 + k_local.
                    idx_f = (4 + k) * 2 + i
                    for c in range(20):
                        W_full[idx_f][20 + c] = W_c[idx_c][c]
                        
            # Block 3: B cols 8,9. C cols 8,9.
            # Use the standard Rank 11 block from _build_block_55, but extracted.
            # We can just reuse the logic for g=4 from _build_block_55 but write to offset 40.
            # Or simpler: call _build_block_55 and take the last 11 columns.
            # But _build_block_55 returns the whole thing.
            # Let's just construct the last block manually or extract from base.
            
            # Extract last 11 columns from base solution (which is for g=4)
            for i in range(6):
                for c in range(11):
                    U_full[i][40 + c] = U_base[i][44 + c] # Base g=4 starts at 44
            for j in range(3):
                for k in range(10):
                    for c in range(11):
                        V_full[j*10+k][40+c] = V_base[j*10+k][44+c]
            for k in range(10):
                for i in range(2):
                    for c in range(11):
                        W_full[k*2+i][40+c] = W_base[k*2+i][44+c]
                        
            if _validate(U_full, V_full, W_full):
                return {"U": U_full, "V": V_full, "W": W_full}
        
        # Fallback to Rank 55
        return {"U": U_base, "V": V_base, "W": W_base}
        
    # Fallback
    return {"U": U_base, "V": V_base, "W": W_base}
