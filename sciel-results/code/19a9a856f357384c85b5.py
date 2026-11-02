import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = 2, 3, 9
    
    # First, build the known rank-50 decomposition as fallback
    rank50 = 50
    U50 = [[0] * rank50 for _ in range(n * m)]
    V50 = [[0] * rank50 for _ in range(m * p)]
    W50 = [[0] * rank50 for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U50[u_a][col] += 1
        U50[u_d][col] += 1
        V50[v_e][col] += 1
        V50[v_h][col] += 1
        
        U50[u_c][col+1] += 1
        U50[u_d][col+1] += 1
        V50[v_e][col+1] += 1
        
        U50[u_a][col+2] += 1
        V50[v_f][col+2] += 1
        V50[v_h][col+2] -= 1
        
        U50[u_d][col+3] += 1
        V50[v_g][col+3] += 1
        V50[v_e][col+3] -= 1
        
        U50[u_a][col+4] += 1
        U50[u_b][col+4] += 1
        V50[v_h][col+4] += 1
        
        U50[u_c][col+5] += 1
        U50[u_a][col+5] -= 1
        V50[v_e][col+5] += 1
        V50[v_f][col+5] += 1
        
        U50[u_b][col+6] += 1
        U50[u_d][col+6] -= 1
        V50[v_g][col+6] += 1
        V50[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W50[w_00][col] += 1
        W50[w_00][col+3] += 1
        W50[w_00][col+4] -= 1
        W50[w_00][col+6] += 1
        
        W50[w_01][col+2] += 1
        W50[w_01][col+4] += 1
        
        W50[w_10][col+1] += 1
        W50[w_10][col+3] += 1
        
        W50[w_11][col] += 1
        W50[w_11][col+1] -= 1
        W50[w_11][col+2] += 1
        W50[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U50[u_a][col] += 1
    V50[v_e][col] += 1
    U50[u_b][col+1] += 1
    V50[v_g][col+1] += 1
    U50[u_c][col+2] += 1
    V50[v_e][col+2] += 1
    U50[u_d][col+3] += 1
    V50[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W50[w_08][col] += 1
    W50[w_08][col+1] += 1
    W50[w_18][col+2] += 1
    W50[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U50[u_ar0][col] += 1
        V50[v_b][col] += 1
        W50[k * n + 0][col] += 1
        U50[u_ar1][col+1] += 1
        V50[v_b][col+1] += 1
        W50[k * n + 1][col+1] += 1
        col += 2
    
    # Now try a structural approach: embed into 3x3x3 and use rank-5
    # The key idea: A is 2x3, embed as 3x3 with row3 = row1+row2
    # B is 3x9, split into three 3x3 blocks B0, B1, B2
    # C = A*B is 2x9, split into three 2x3 blocks C0, C1, C2
    # C_i = A * B_i for i=0,1,2
    # With A_ext (3x3) and B_i (3x3), we compute A_ext * B_i (3x3)
    # Then C_i[0,:] = (A_ext*B_i)[0,:], C_i[1,:] = (A_ext*B_i)[1,:]
    # The third row of A_ext*B_i is redundant (sum of first two rows)
    
    # Rank-5 decomposition for 3x3x3 (known to exist):
    # This uses a specific set of 5 rank-1 tensors
    # Let me try to construct one based on known results
    
    # Actually, let me try a different approach: use the structure more directly
    # The (2,3,9) tensor can be seen as 3 copies of (2,3,3) tensors
    # Each (2,3,3) tensor: multiply 2x3 by 3x3 to get 2x3
    # The rank of (2,3,3) is at most 6 (trivial), but can be lower
    
    # For (2,3,3): A is 2x3, B is 3x3, C = A*B is 2x3
    # T[i,j,k] where i in 0..5 (A indices), j in 0..8 (B indices), k in 0..5 (C indices)
    # T[i,j,k] = 1 if i%3 == j%3 and k = (j%3)*2 + (i//3)
    
    # Let me try to find a low-rank decomposition for (2,3,3) first
    # Then replicate for each of the 3 blocks
    
    # Actually, let me try a completely different structural approach:
    # Use the fact that the (2,3,9) tensor has a Kronecker product structure
    # (2,3,9) = (2,3,3) x 3 (three independent blocks)
    # So rank(2,3,9) = 3 * rank(2,3,3)
    
    # If rank(2,3,3) = 5, then rank(2,3,9) = 15
    # If rank(2,3,3) = 6, then rank(2,3,9) = 18
    # The trivial rank of (2,3,3) is 3 (one per column of B)
    # Wait, that's not right. Let me think again.
    
    # For (2,3,3): we need to compute C[i,k] = sum_m A[i,m] * B[m,k]
    # for i in {0,1}, m in {0,1,2}, k in {0,1,2}
    # The trivial decomposition has rank 3 (sum over m)
    # But we can potentially do better with bilinear algorithms
    
    # Actually, the rank of the (2,3,3) tensor is exactly 3 (it's just a sum of 3 products)
    # So the trivial rank is 3, and the tensor rank is at most 3
    # But wait, that means rank(2,3,9) = 3*3 = 9? That can't be right because
    # the trivial rank of (2,3,9) is 9 (sum over m=0,1,2 of A[i,m]*B[m,k])
    
    # I think I'm confusing myself. Let me be more careful.
    # The tensor T for (n,m,p) has T[i,j,k] = 1 iff i%m == j%p and k = (j%p)*n + (i//m)
    # For (2,3,9): i in 0..5, j in 0..26, k in 0..17
    # T[i,j,k] = 1 if i%3 == j%9 and k = (j%9)*2 + (i//3)
    # Wait, that's not right either. Let me re-read the convention.
    
    # The convention is W[k*n+i, r], so k is the row of C (0..p-1=0..8), i is the col of C (0..n-1=0..1)
    # So C[k,i] = sum_m A[i,m] * B[m,k]... no wait.
    # Standard: C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_m A[i,m] * B[m,k]
    # The tensor: T[A_idx, B_idx, C_idx]
    # A_idx = i*m + a where i is row of A (0..n-1), a is col of A (0..m-1)
    # B_idx = b*p + k where b is row of B (0..m-1), k is col of B (0..p-1)
    # C_idx = k*n + i where k is col of C (0..p-1), i is row of C (0..n-1)
    # T[A_idx, B_idx, C_idx] = 1 if a == b (the inner dimension matches)
    
    # So for (2,3,9): A is 2x3, B is 3x9, C is 2x9
    # A_idx in 0..5: A_idx = i*3 + a, i in {0,1}, a in {0,1,2}
    # B_idx in 0..26: B_idx = b*9 + k, b in {0,1,2}, k in {0,..,8}
    # C_idx in 0..17: C_idx = k*2 + i, k in {0,..,8}, i in {0,1}
    # T[A_idx, B_idx, C_idx] = 1 if a == b
    
    # The trivial decomposition: for each b in {0,1,2} and each k in {0,..,8}:
    # T has 1 at (i*3+b, b*9+k, k*2+i) for all i
    # This gives rank 2*3*9/2 = 27? No, the trivial rank is m*p = 3*9 = 27? 
    # Actually the trivial rank is m = 3 if we sum over m... no.
    # The trivial bilinear algorithm: C[i,k] = sum_m A[i,m]*B[m,k]
    # This is a sum of m=3 terms, so rank at most 3? No, each term A[i,m]*B[m,k] 
    # is a rank-1 tensor only if we fix m. But A[i,m] varies with i and B[m,k] varies with k.
    # So for fixed m, the tensor T_m[A_idx, B_idx, C_idx] = 1 if a=m and b=m and C_idx=k*n+i
    # This is a rank-1 tensor? No, it's a product of three indicator functions.
    # Actually, for fixed m, T_m is nonzero only when A_idx has a=m, B_idx has b=m, 
    # and C_idx = k*n+i for some k,i. The support is {(i*3+m, m*9+k, k*2+i) for all i,k}.
    # This is not a rank-1 tensor in general.
    
    # I think I need to just try the null-space approach more aggressively.
    # Let me build the coefficient matrix and find the minimum rank subset.
    
    U_np = np.array(U50, dtype=float)
    V_np = np.array(V50, dtype=float)
    W_np = np.array(W50, dtype=float)
    
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
    decomp50 = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    if not np.allclose(decomp50, T, atol=1e-10):
        return {"U": U50, "V": V50, "W": W50}
    
    # Build the support of T
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
    
    # Build matrix M where M[t, r] = U[i_t, r] * V[j_t, r] * W[k_t, r]
    M = np.zeros((n_support, rank50))
    for t, (i, j, k) in enumerate(T_support):
        for r in range(rank50):
            M[t, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Also build the target vector on support
    T_vec = np.zeros(n_support)
    for t, (i, j, k) in enumerate(T_support):
        T_vec[t] = T[i, j, k]
    
    # Find the null space of M to identify dependencies
    try:
        if time.time() < deadline - 20:
            _, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = s[0] * rank50 * 1e-10
            num_rank = np.sum(s > tol)
            
            if num_rank < rank50:
                null_dim = rank50 - num_rank
                null_vecs = Vt[num_rank:]
                
                # Greedy column removal
                best_mask = np.ones(rank50, dtype=bool)
                best_rank = rank50
                
                # Try removing columns one at a time, starting with those most in null space
                removed = set()
                for _ in range(rank50 - 1):
                    if time.time() > deadline - 10:
                        break
                    
                    best_drop = None
                    best_res = float('inf')
                    
                    for r in range(rank50):
                        if r in removed:
                            continue
                        
                        mask = np.ones(rank50, dtype=bool)
                        mask[removed] = False
                        mask[r] = False
                        
                        U_new = U_np[:, mask]
                        V_new = V_np[:, mask]
                        W_new = W_np[:, mask]
                        
                        decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                        residual = np.sum((decomp_new - T)**2)
                        
                        if residual < best_res:
                            best_res = residual
                            best_drop = r
                    
                    if best_drop is not None and best_res < 1e-8:
                        removed.add(best_drop)
                        best_rank -= 1
                    else:
                        break
                
                if best_rank < rank50:
                    final_mask = np.ones(rank50, dtype=bool)
                    final_mask[removed] = False
                    U_best = U_np[:, final_mask]
                    V_best = V_np[:, final_mask]
                    W_best = W_np[:, final_mask]
                    
                    # Verify
                    decomp_best = np.einsum('ir,jr,kr->ijk', U_best, V_best, W_best)
                    if np.allclose(decomp_best, T, atol=1e-10):
                        return {
                            "U": U_best.tolist(),
                            "V": V_best.tolist(),
                            "W": W_best.tolist()
                        }
    except Exception:
        pass
    
    return {"U": U50, "V": V50, "W": W50}
