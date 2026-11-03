import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:, t], V[:, t])
            T_recon += np.einsum('ab,c->abc', uv, W[:, t])
        return np.max(np.abs(T_recon - T))

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)

    # Try to find a lower-rank decomposition using alternating least squares
    # with careful initialization and convergence checking.
    # 
    # Strategy: Start from schoolbook, then try to merge terms by finding
    # linear dependencies among the rank-1 tensors.
    # 
    # A better approach: Use the fact that matrix multiplication tensor has
    # structure. We can partition the computation and use Strassen-like
    # decompositions on sub-blocks.
    #
    # For 4x5 * 5x6:
    # Partition A (4x5) = [A1(4x4) | A2(4x1)]
    # Partition B (5x6) = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: (4x4)*(4x6) - can be viewed as 4x4 times 4x6
    # A2*B2: (4x1)*(1x6) - rank 4 (schoolbook)
    #
    # For A1*B1 where A1 is 4x4 and B1 is 4x6:
    # We can further partition B1 = [B1a(4x4) | B1b(4x2)]
    # A1*B1a: 4x4 * 4x4 - use Strassen (rank 28)
    # A1*B1b: 4x4 * 4x2 - schoolbook rank 32
    #
    # Total: 28 + 32 + 4 = 64
    #
    # But wait, let me reconsider. The rank of 4x4*4x4 via Strassen is 28.
    # Let me verify this is correct and implement it.
    
    def build_strassen_4x4():
        """Build rank-28 Strassen decomposition for 4x4 * 4x4.
        Returns U (16, 28), V (16, 28), W (16, 28) such that
        sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] = delta_{ik} for the 4x4*4x4 tensor.
        """
        r = 28
        U = np.zeros((16, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        
        # 4 sub-multiplications: (row_block, col_block) in {(0,0), (0,1), (1,0), (1,1)}
        # For each pair (rb, cb), we multiply X[2rb:2rb+2, 2cb:2cb+2] * Y[2cb:2cb+2, 2cb:2cb+2]
        # and add to C[2rb:2rb+2, 2cb:2cb+2]
        
        for rb in range(2):
            for cb in range(2):
                base = (rb * 2 + cb) * 7
                
                # X block indices: rows 2rb, 2rb+1; cols 2cb, 2cb+1
                # Y block indices: rows 2cb, 2cb+1; cols 2cb, 2cb+1
                # C block indices: rows 2rb, 2rb+1; cols 2cb, 2cb+1
                
                # Strassen for 2x2 * 2x2:
                # X = [[x00, x01], [x10, x11]]
                # Y = [[y00, y01], [y10, y11]]
                # C = [[c00, c01], [c10, c11]]
                #
                # M1 = (x11 + x10)(y00 + y11)
                # M2 = x10(y11 - y00)
                # M3 = (x00 + x01)y11
                # M4 = x00(y10 - y11)
                # M5 = (x00 + x11)y00
                # M6 = (x10 - x00)(y00 + y01)
                # M7 = (x01 - x11)(y10 + y11)
                #
                # c00 = M1 + M4 - M5 + M7
                # c01 = M3 + M7
                # c10 = M2 + M5
                # c11 = M1 - M2 + M3 - M4
                
                # Map to global indices:
                # x00 = X[2rb, 2cb], x01 = X[2rb, 2cb+1]
                # x10 = X[2rb+1, 2cb], x11 = X[2rb+1, 2cb+1]
                # y00 = Y[2cb, 2cb], y01 = Y[2cb, 2cb+1]
                # y10 = Y[2cb+1, 2cb], y11 = Y[2cb+1, 2cb+1]
                # c00 = C[2rb, 2cb], c01 = C[2rb, 2cb+1]
                # c10 = C[2rb+1, 2cb], c11 = C[2rb+1, 2cb+1]
                
                # U indices: i*4+j where i is row of X, j is col of X
                # V indices: j*4+k where j is row of Y, k is col of Y
                # W indices: k*4+i where k is col of C, i is row of C
                
                x00_u = (2*rb) * 4 + (2*cb)
                x01_u = (2*rb) * 4 + (2*cb + 1)
                x10_u = (2*rb + 1) * 4 + (2*cb)
                x11_u = (2*rb + 1) * 4 + (2*cb + 1)
                
                y00_v = (2*cb) * 4 + (2*cb)
                y01_v = (2*cb) * 4 + (2*cb + 1)
                y10_v = (2*cb + 1) * 4 + (2*cb)
                y11_v = (2*cb + 1) * 4 + (2*cb + 1)
                
                c00_w = (2*cb) * 4 + (2*rb)
                c01_w = (2*cb + 1) * 4 + (2*rb)
                c10_w = (2*cb) * 4 + (2*rb + 1)
                c11_w = (2*cb + 1) * 4 + (2*rb + 1)
                
                # M1 = (x11 + x10)(y00 + y11)
                t = base + 0
                U[x11_u, t] += 1
                U[x10_u, t] += 1
                V[y00_v, t] += 1
                V[y11_v, t] += 1
                W[c00_w, t] += 1
                W[c11_w, t] += 1
                
                # M2 = x10(y11 - y00)
                t = base + 1
                U[x10_u, t] += 1
                V[y11_v, t] += 1
                V[y00_v, t] += -1
                W[c10_w, t] += 1
                W[c11_w, t] += -1
                
                # M3 = (x00 + x01)y11
                t = base + 2
                U[x00_u, t] += 1
                U[x01_u, t] += 1
                V[y11_v, t] += 1
                W[c01_w, t] += 1
                W[c11_w, t] += 1
                
                # M4 = x00(y10 - y11)
                t = base + 3
                U[x00_u, t] += 1
                V[y10_v, t] += 1
                V[y11_v, t] += -1
                W[c00_w, t] += 1
                W[c11_w, t] += -1
                
                # M5 = (x00 + x11)y00
                t = base + 4
                U[x00_u, t] += 1
                U[x11_u, t] += 1
                V[y00_v, t] += 1
                W[c00_w, t] += -1
                W[c10_w, t] += 1
                
                # M6 = (x10 - x00)(y00 + y01)
                t = base + 5
                U[x10_u, t] += 1
                U[x00_u, t] += -1
                V[y00_v, t] += 1
                V[y01_v, t] += 1
                # M6 is not used in the standard Strassen output formulas!
                # Let me recheck... Actually, in standard Strassen, all 7 M's are used.
                # Let me re-examine the formulas.
                # 
                # From standard Strassen:
                # c00 = M1 + M4 - M5 + M7
                # c01 = M3 + M7  
                # c10 = M2 + M5
                # c11 = M1 - M2 + M3 - M4
                #
                # M6 is not used! That can't be right for a rank-7 decomposition.
                # Let me look up the correct Strassen formulas.
                #
                # Actually, I think I have the wrong formulas. Let me use a verified set.
                # From Wikipedia (Strassen algorithm):
                # m1 = (a11 + a21)(b11 + b22)
                # m2 = a21(b12 - b11)
                # m3 = (a11 + a12)b22
                # m4 = a11(b21 - b22)
                # m5 = (a11 + a21)b11
                # m6 = (a21 - a11)(b11 + b12)
                # m7 = (a12 - a22)(b21 + b22)
                #
                # c11 = m1 + m4 - m5 + m7
                # c12 = m3 + m7
                # c21 = m2 + m5
                # c22 = m1 - m2 + m3 - m4
                #
                # So m6 is indeed not used in the output! This means the rank is actually 7
                # but one of the multiplications is redundant? No, that doesn't make sense.
                # 
                # Wait, I think the issue is that I'm using a non-standard Strassen variant.
                # Let me just verify numerically that my construction gives the right answer.
                pass
        
        return U, V, W
    
    # Let me verify the Strassen construction numerically
    U_s, V_s, W_s = build_strassen_4x4()
    
    # Build 4x4*4x4 tensor
    T_44 = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T_44[i*4+j, j*4+k, k*4+i] = 1.0
    
    # Verify
    T_recon = np.zeros((16, 16, 16))
    for t in range(28):
        uv = np.outer(U_s[:, t], V_s[:, t])
        T_recon += np.einsum('ab,c->abc', uv, W_s[:, t])
    
    err_44 = np.max(np.abs(T_recon - T_44))
    
    if err_44 > 1e-10:
        # Strassen construction has errors, fall back to schoolbook
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Good, Strassen works. Now build the full 4x5*5x6 decomposition.
    # 
    # A (4x5) = [A1(4x4) | A2(4x1)]
    # B (5x6) = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: (4x4)*(4x6)
    # Further partition: B1 = [B1a(4x4) | B1b(4x2)]
    # A1*B1a: 4x4 * 4x4 -> Strassen rank 28
    # A1*B1b: 4x4 * 4x2 -> schoolbook rank 32
    #
    # A2*B2: (4x1)*(1x6) -> schoolbook rank 4
    #
    # Total rank: 28 + 32 + 4 = 64
    
    r_total = 28 + 32 + 4
    U_full = np.zeros((20, r_total))
    V_full = np.zeros((30, r_total))
    W_full = np.zeros((24, r_total))
    
    col = 0
    
    # Part 1: A1*B1a where A1 is rows 0-3, cols 0-3 of A
    #         B1a is rows 0-3, cols 0-3 of B
    #         C contribution is rows 0-3, cols 0-3 of C
    #
    # In the global indexing:
    # A index: i*m+j = i*5+j for i in 0..3, j in 0..3
    # B index: j*p+k = j*6+k for j in 0..3, k in 0..3
    # C index: k*n+i = k*4+i for k in 0..3, i in 0..3
    
    for t in range(28):
        for a_idx in range(16):
            if U_s[a_idx, t] != 0:
                # a_idx = i*4+j where i,j in 0..3
                i = a_idx // 4
                j = a_idx % 4
                U_full[i*5 + j, col] += U_s[a_idx, t]
        
        for b_idx in range(16):
            if V_s[b_idx, t] != 0:
                # b_idx = j*4+k where j,k in 0..3
                j = b_idx // 4
                k = b_idx % 4
                V_full[j*6 + k, col] += V_s[b_idx, t]
        
        for c_idx in range(16):
            if W_s[c_idx, t] != 0:
                # c_idx = k*4+i where k,i in 0..3
                k = c_idx // 4
                i = c_idx % 4
                W_full[k*4 + i, col] += W_s[c_idx, t]
        
        col += 1
    
    # Part 2: A1*B1b where A1 is rows 0-3, cols 0-3 of A
    #         B1b is rows 0-3, cols 4-5 of B
    #         C contribution is rows 0-3, cols 4-5 of C
    #
    # Schoolbook: for each i in 0..3, j in 0..3, k in 4..5
    # A[i,j] * B[j,k] contributes to C[i,k]
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_full[i*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + i, col] = 1
                col += 1
    
    # Part 3: A2*B2 where A2 is rows 0-3, col 4 of A
    #         B2 is row 4, cols 0-5 of B
    #         C contribution is rows 0-3, cols 0-5 of C
    #
    # Schoolbook: for each i in 0..3, k in 0..5
    # A[i,4] * B[4,k] contributes to C[i,k]
    
    for i in range(4):
        for k in range(6):
            U_full[i*5 + 4, col] = 1
            V_full[4*6 + k, col] = 1
            W_full[k*4 + i, col] = 1
            col += 1
    
    # Verify the full decomposition
    err_full = verify_decomposition(U_full, V_full, W_full)
    
    if err_full < 1e-10:
        best_rank = r_total
        best_decomp = (U_full.copy(), V_full.copy(), W_full.copy())
    
    # Now try to find an even lower-rank decomposition using ALS
    # Start from our rank-64 solution and try to reduce rank
    
    np.random.seed(42)
    
    for target_rank in [50, 55, 60]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random perturbation of the best solution
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                
                # Fix V, W: solve for U
                X = np.einsum('bt,ct->bct', V, W).reshape(target_rank, mp*pn).T
                for a in range(nm):
                    U[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
                
                # Fix U, W: solve for V
                X = np.einsum('at,ct->act', U, W).reshape(target_rank, nm*pn).T
                for b in range(mp):
                    V[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
                
                # Fix U, V: solve for W
                X = np.einsum('at,bt->abt', U, V).reshape(target_rank, nm*mp).T
                for c in range(pn):
                    W[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
                
                # Check convergence
                err = verify_decomposition(U, V, W)
                if err < 1e-12:
                    break
            
            err = verify_decomposition(U, V, W)
            if err < 1e-10:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                break
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
