import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = 2, 3, 9

    # Build the rank-50 decomposition (same as current)
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Verify rank-50 is correct
    decomp_full = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    if not np.allclose(decomp_full, T, atol=1e-10):
        return {"U": U, "V": V, "W": W}

    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()

    # Strategy 1: Null space search for column removal
    try:
        # Build support matrix M
        T_support = []
        for i in range(n*m):
            ai = i // m
            am = i % m
            for j in range(m*p):
                bj = j // p
                bm = j % p
                if am == bj:
                    k = bm * n + ai
                    T_support.append((i, j, k))
        
        n_support = len(T_support)
        M = np.zeros((n_support, rank))
        for t, (i, j, k) in enumerate(T_support):
            for r in range(rank):
                M[t, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
        
        # SVD for null space
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = s[0] * rank * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            null_dim = rank - num_rank
            null_vecs = Vt[num_rank:]
            
            # Try removing single columns
            for null_idx in range(null_dim):
                if time.time() > deadline - 30:
                    break
                
                null_vec = null_vecs[null_idx]
                abs_null = np.abs(null_vec)
                candidates = np.argsort(abs_null)[::-1]
                
                for drop_col in candidates[:5]:
                    if time.time() > deadline - 20:
                        break
                    
                    mask = np.ones(rank, dtype=bool)
                    mask[drop_col] = False
                    U_new = U_np[:, mask]
                    V_new = V_np[:, mask]
                    W_new = W_np[:, mask]
                    
                    decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                    residual = np.sum((decomp_new - T)**2)
                    
                    if residual < 1e-8 and mask.sum() < best_rank:
                        best_rank = int(mask.sum())
                        best_U = U_new.copy()
                        best_V = V_new.copy()
                        best_W = W_new.copy()
                    
                    if residual < 1e-8:
                        break
                
                if best_rank < rank:
                    break
            
            # Try pairs
            if best_rank == rank:
                for null_idx in range(null_dim):
                    if time.time() > deadline - 15:
                        break
                    
                    null_vec = null_vecs[null_idx]
                    abs_null = np.abs(null_vec)
                    candidates = np.argsort(abs_null)[::-1]
                    
                    for i in range(min(4, len(candidates))):
                        for j in range(i+1, min(4, len(candidates))):
                            if time.time() > deadline - 10:
                                break
                            
                            drop1 = candidates[i]
                            drop2 = candidates[j]
                            
                            mask = np.ones(rank, dtype=bool)
                            mask[drop1] = False
                            mask[drop2] = False
                            U_new = U_np[:, mask]
                            V_new = V_np[:, mask]
                            W_new = W_np[:, mask]
                            
                            decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                            residual = np.sum((decomp_new - T)**2)
                            
                            if residual < 1e-8 and mask.sum() < best_rank:
                                best_rank = int(mask.sum())
                                best_U = U_new.copy()
                                best_V = V_new.copy()
                                best_W = W_new.copy()
                            
                            if residual < 1e-8:
                                break
                        
                        if best_rank < rank:
                            break
                    
                    if best_rank < rank:
                        break
    except Exception:
        pass

    # Strategy 2: Try a simpler block-based decomposition
    # Split inner dimension m=3 into (2+1)
    # For the 2x2x3 blocks, use trivial rank-12
    # For the 1x1x9 part, use rank 9
    # Total: 3 * 12 + 9 = 45? Or maybe we can share?
    
    try:
        if time.time() < deadline - 20:
            # Let's try a different approach: 
            # The tensor is T[i,j,k] where i in 0..5, j in 0..26, k in 0..17
            # T[i,j,k] = 1 iff i%3 == j%3 and k = (j%3)*2 + (i//3)
            
            # Let's try to find a rank-40 decomposition by hand or search
            # Actually, let's try the approach from the inspiration:
            # Use Strassen on 2x2 blocks with (2+1) split
            
            # The current rank-50 uses 7 terms per 2-column block (4 blocks = 28)
            # + 4 for the last column + 18 for the tail = 50
            
            # Can we do better?
            # For 2x3x3, the rank is known to be 7 (Strassen-like)
            # For 2x3x9, we have 3 such blocks, so 3*7 = 21?
            # But there might be sharing.
            
            # Let's try to construct a rank-21 decomposition
            # Each block handles 3 output columns
            
            rank21 = 21
            U21 = [[0] * rank21 for _ in range(n * m)]
            V21 = [[0] * rank21 for _ in range(m * p)]
            W21 = [[0] * rank21 for _ in range(p * n)]
            
            # For each block b (0, 1, 2), handle columns 3*b to 3*b+2
            # Use a 2x3x3 decomposition of rank 7
            
            # Standard 2x3x3 rank-7 decomposition:
            # A = [a11 a12 a13; a21 a22 a23]
            # B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
            # C = A*B (2x3)
            
            # Let's use a known rank-7 algorithm for 2x3x3
            # This is equivalent to multiplying a 2x3 by a 3x3
            
            # One approach: embed into 3x3x3 and use Strassen rank 7
            # But we only need the top 2 rows of the result
            
            # Let's try a direct construction
            # For block b, the relevant B rows are 0,1,2 (all rows)
            # and columns are 3*b to 3*b+2
            
            # Actually, the structure is:
            # C[i, 3*b + k] = sum_m A[i,m] * B[m, 3*b+k]
            
            # For fixed b, this is a 2x3x3 tensor
            # Let's use the trivial rank-9 (3 terms per output column)
            # Or find a rank-7
            
            # Let's just use rank-9 per block for now: 3*9 = 27
            # That's better than 50
            
            rank27 = 27
            U27 = [[0] * rank27 for _ in range(n * m)]
            V27 = [[0] * rank27 for _ in range(m * p)]
            W27 = [[0] * rank27 for _ in range(p * n)]
            
            col27 = 0
            for b in range(3):
                for k in range(3):
                    # Output column 3*b + k
                    # C[i, 3*b+k] = A[i,0]*B[0,3*b+k] + A[i,1]*B[1,3*b+k] + A[i,2]*B[2,3*b+k]
                    
                    # Term 1: A[i,0] * B[0,3*b+k]
                    for i in range(2):
                        U27[i*3 + 0][col27] += 1
                    V27[0 * p + (3*b+k)][col27] += 1
                    W27[(3*b+k)*2 + 0][col27] += 1
                    W27[(3*b+k)*2 + 1][col27] += 1
                    col27 += 1
                    
                    # Term 2: A[i,1] * B[1,3*b+k]
                    for i in range(2):
                        U27[i*3 + 1][col27] += 1
                    V27[1 * p + (3*b+k)][col27] += 1
                    W27[(3*b+k)*2 + 0][col27] += 1
                    W27[(3*b+k)*2 + 1][col27] += 1
                    col27 += 1
                    
                    # Term 3: A[i,2] * B[2,3*b+k]
                    for i in range(2):
                        U27[i*3 + 2][col27] += 1
                    V27[2 * p + (3*b+k)][col27] += 1
                    W27[(3*b+k)*2 + 0][col27] += 1
                    W27[(3*b+k)*2 + 1][col27] += 1
                    col27 += 1
            
            U27_np = np.array(U27, dtype=float)
            V27_np = np.array(V27, dtype=float)
            W27_np = np.array(W27, dtype=float)
            
            decomp27 = np.einsum('ir,jr,kr->ijk', U27_np, V27_np, W27_np)
            if np.allclose(decomp27, T, atol=1e-10):
                best_rank = 27
                best_U = U27_np.copy()
                best_V = V27_np.copy()
                best_W = W27_np.copy()
    except Exception:
        pass

    # Strategy 3: Try even lower rank with search
    try:
        if time.time() < deadline - 10 and best_rank > 20:
            # Try to find a rank-20 or lower decomposition
            # Use a greedy search with random perturbations
            
            # Start from the rank-27 solution
            if best_rank == 27:
                U_cur = best_U.copy()
                V_cur = best_V.copy()
                W_cur = best_W.copy()
                cur_rank = 27
                
                # Try removing columns
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Try each column
                    for r in range(cur_rank):
                        mask = np.ones(cur_rank, dtype=bool)
                        mask[r] = False
                        U_new = U_cur[:, mask]
                        V_new = V_cur[:, mask]
                        W_new = W_cur[:, mask]
                        
                        decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                        residual = np.sum((decomp_new - T)**2)
                        
                        if residual < 1e-8:
                            U_cur = U_new
                            V_cur = V_new
                            W_cur = W_new
                            cur_rank -= 1
                            break
                        
                        if time.time() > deadline - 5:
                            break
                    else:
                        break
                    continue
                    break
                
                if cur_rank < best_rank:
                    best_rank = cur_rank
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
    except Exception:
        pass

    if best_rank < rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    return {"U": U, "V": V, "W": W}
