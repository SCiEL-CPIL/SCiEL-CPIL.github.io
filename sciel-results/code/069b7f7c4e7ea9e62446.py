"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
    
    # Strassen-style block decomposition
    # Split 2x4 x 4x7 into blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 1x2
    # B = [[B00, B01, B02, B03, B04, B05, B06],
    #      [B10, B11, B12, B13, B14, B15, B16]]
    # Actually, let's think of it differently.
    # A is 2x4, B is 4x7.
    # Split A into 2 blocks of 1x2: A = [A_top, A_bottom]
    # Split B into 2 blocks of 2x7: B = [B_top; B_bottom]
    # Then C = A_top @ B_top + A_bottom @ B_bottom
    # Each term is 1x2 @ 2x7 = 1x7, which needs rank 2 (schoolbook for 1x2x7)
    # But we can use Strassen for the 2x2 core if we pad.
    
    # Better approach: Use the known result that R(2,4,7) can be done with
    # Strassen on 2x2x2 blocks.
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 1x2
    # Wait, A is 2x4. Let's split into 2 blocks of 1x4 or 4 blocks of 1x2.
    # Let's split into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 1x2
    # B is 4x7. Split into 2x2 blocks? No, B is 4x7.
    # Let's split B into 2 blocks of 2x7: B = [B_top; B_bottom]
    # And A into 2 blocks of 1x4: A = [A_top; A_bottom]
    # C = A_top @ B_top + A_bottom @ B_bottom
    # A_top is 1x4, B_top is 2x7. This doesn't directly help.
    
    # Let me try a different block structure.
    # A is 2x4. Split columns into 2 groups of 2: A = [A_L | A_R] where each is 2x2
    # B is 4x7. Split rows into 2 groups of 2: B = [B_T; B_B] where each is 2x7
    # C = A_L @ B_T + A_R @ B_B
    # Each term is 2x2 @ 2x7.
    # For 2x2 @ 2x7: we can use Strassen for the 2x2 part if we extend to 2x2x2.
    # Actually, 2x2 @ 2x7: think of it as 7 independent 2x2 @ 2x1 multiplications.
    # R(2,2,1) = 4 (known result)
    # So 2x2 @ 2x7 needs 7*4 = 28? No, that's not right either.
    
    # Let me use the measured finding: R(2,2,2)=7 and R(2,2,1)=4.
    # Split A (2x4) into 2 blocks of 2x2: A = [A0 | A1]
    # Split B (4x7) into 2 blocks of 2x7: B = [B0; B1]
    # C = A0 @ B0 + A1 @ B1
    # For A0 @ B0: A0 is 2x2, B0 is 2x7.
    # We can view B0 as 7 columns, each 2x1.
    # So A0 @ B0 = [A0@b0, A0@b1, ..., A0@b6] where each bi is 2x1.
    # R(2,2,1) = 4, so each column needs rank 4.
    # Total for A0@B0: 7 * 4 = 28? No, the rank is shared.
    
    # Actually, let me just implement the Strassen-based approach directly.
    # Use the fact that we can compute 2x2 @ 2x2 with rank 7.
    # Pad B to 2x8 (add a zero column) or handle 2x7 specially.
    
    # For now, let's use a simpler approach: try to find a rank-50 decomposition
    # using the block structure.
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to construct a rank-50 decomposition using Strassen blocks
    # A = [A0 | A1] where A0, A1 are 2x2
    # B = [B0; B1] where B0, B1 are 2x7
    # C = A0 @ B0 + A1 @ B1
    
    # For each 2x2 @ 2x7 multiplication, use Strassen-like approach
    # 2x2 @ 2x2 has rank 7 (Strassen)
    # 2x2 @ 2x7: treat as 7 separate 2x2 @ 2x1, each with rank 4
    # But we can share the structure.
    
    # Let me just try alternating minimization with better initialization
    # based on the block structure.
    
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    T_vec_u = T.reshape(nm, mp * pn)
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    
    def try_rank_alt(r, n_trials=20, max_iter=600):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Structured initialization based on Strassen blocks
            if trial % 3 == 0:
                # Block-structured initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                
                # For each rank element, assign it to a specific (i,j,k) block
                for d in range(r):
                    # Determine which block this rank element belongs to
                    block = d % 4  # 4 blocks: (0,0), (0,1), (1,0), (1,1)
                    i_block = block // 2
                    j_block = block % 2
                    
                    # Within each block, use Strassen-like patterns
                    # For simplicity, just assign to specific positions
                    U[i_block * m + j_block, d] = 1.0
                    V[j_block * 2 + d % 2, d] = 1.0  # Approximate
                    W[(d % p) * n + i_block, d] = 1.0
            elif trial % 3 == 1:
                # Random initialization
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            else:
                # Sparse initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    idx_u = np.random.choice(nm, size=max(1, nm//4), replace=False)
                    idx_v = np.random.choice(mp, size=max(1, mp//4), replace=False)
                    idx_w = np.random.choice(pn, size=max(1, pn//4), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                X = np.einsum('ad,bd->abd', U, V).reshape(nm * mp, r)
                XTX = X.T @ X + 1e-10 * np.eye(r)
                XTT = X.T @ T_vec_w.T
                try:
                    W_new = np.linalg.solve(XTX, XTT).T
                except np.linalg.LinAlgError:
                    W_new = W
                
                Y = np.einsum('bd,cd->bcd', V, W_new).reshape(mp * pn, r)
                YTY = Y.T @ Y + 1e-10 * np.eye(r)
                YTT = Y.T @ T_vec_u.T
                try:
                    U_new = np.linalg.solve(YTY, YTT).T
                except np.linalg.LinAlgError:
                    U_new = U
                
                Z = np.einsum('ad,cd->acd', U_new, W_new).reshape(nm * pn, r)
                ZTZ = Z.T @ Z + 1e-10 * np.eye(r)
                ZTT = Z.T @ T_vec_v.T
                try:
                    V_new = np.linalg.solve(ZTZ, ZTT).T
                except np.linalg.LinAlgError:
                    V_new = V
                
                U, V, W = U_new, V_new, W_new
                
                T_rec = tensor_from_factors(U, V, W)
                err = np.linalg.norm(T_rec - T)
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-12 and it > 20:
                    break
                prev_err = err
            
            T_rec = tensor_from_factors(U, V, W)
            if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec_r = tensor_from_factors(U_r, V_r, W_r)
                if np.allclose(T_rec_r, T, atol=1e-8, rtol=1e-8):
                    best_U, best_V, best_W = U_r, V_r, W_r
                else:
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    T_rec_i = tensor_from_factors(U_i, V_i, W_i)
                    if np.allclose(T_rec_i, T, atol=1e-8, rtol=1e-8):
                        best_U, best_V, best_W = U_i, V_i, W_i
                    else:
                        best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        return False
    
    # Try ranks from a reasonable lower bound
    # Based on the finding, rank 50 should be achievable
    for r in range(45, best_rank):
        if time.time() > deadline:
            break
        n_trials = 15 if r < 50 else 10
        success = try_rank_alt(r, n_trials=n_trials)
        if success:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
