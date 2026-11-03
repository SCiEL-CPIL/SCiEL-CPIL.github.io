"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.reshape(nm * mp, pn)  # (216, 24)
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def build_strassen_decomp():
        """Build the Strassen-based rank-68 decomposition."""
        U_t = []
        V_t = []
        W_t = []
        
        def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
            a = (row_off_a, col_off_a)
            b = (row_off_a, col_off_a + 1)
            c = (row_off_a + 1, col_off_a)
            d = (row_off_a + 1, col_off_a + 1)
            e = (row_off_b, col_off_b)
            f = (row_off_b, col_off_b + 1)
            g = (row_off_b + 1, col_off_b)
            h = (row_off_b + 1, col_off_b + 1)
            
            C00 = (row_off_a, col_off_b)
            C01 = (row_off_a, col_off_b + 1)
            C10 = (row_off_a + 1, col_off_b)
            C11 = (row_off_a + 1, col_off_b + 1)
            
            # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
            U_t.append([(a, 1), (d, 1)])
            V_t.append([(e, 1), (h, 1)])
            W_t.append([(C00, 1), (C11, 1)])
            
            # m2 = (c+d)*e -> C10 += 1, C11 -= 1
            U_t.append([(c, 1), (d, 1)])
            V_t.append([(e, 1)])
            W_t.append([(C10, 1), (C11, -1)])
            
            # m3 = a*(f-h) -> C01 += 1, C11 += 1
            U_t.append([(a, 1)])
            V_t.append([(f, 1), (h, -1)])
            W_t.append([(C01, 1), (C11, 1)])
            
            # m4 = d*(g-e) -> C00 += 1, C10 += 1
            U_t.append([(d, 1)])
            V_t.append([(g, 1), (e, -1)])
            W_t.append([(C00, 1), (C10, 1)])
            
            # m5 = (a+b)*h -> C00 -= 1, C01 += 1
            U_t.append([(a, 1), (b, 1)])
            V_t.append([(h, 1)])
            W_t.append([(C00, -1), (C01, 1)])
            
            # m6 = (c-a)*(e+f) -> C11 += 1
            U_t.append([(c, 1), (a, -1)])
            V_t.append([(e, 1), (f, 1)])
            W_t.append([(C11, 1)])
            
            # m7 = (b-d)*(g+h) -> C00 += 1
            U_t.append([(b, 1), (d, -1)])
            V_t.append([(g, 1), (h, 1)])
            W_t.append([(C00, 1)])
        
        def direct_term(i, j, k):
            U_t.append([((i, j), 1)])
            V_t.append([((j, k), 1)])
            W_t.append([((i, k), 1)])
        
        # Apply Strassen to all 2x2 B-blocks for A[0:2, 0:2]
        for b_col in range(0, 8, 2):
            strassen_decomp(0, 0, 0, b_col)
        
        # Direct terms for A[0:2, 2] * B[2, :]
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
        
        # Direct terms for A[2, 0:2] * B[0:2, :]
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
        
        # Direct terms for A[2, 2] * B[2, :]
        for k in range(8):
            direct_term(2, 2, k)
        
        rank = len(W_t)
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
        
        return U, V, W, rank
    
    def verify(Ua, Va, Wa):
        """Verify decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with Strassen rank-68 decomposition
    U_sb, V_sb, W_sb, sb_rank = build_strassen_decomp()
    
    if not verify(U_sb, V_sb, W_sb):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_fb = np.zeros((nm, schoolbook_rank))
        V_fb = np.zeros((mp, schoolbook_rank))
        W_fb = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1.0
                    V_fb[j*p+k, col] = 1.0
                    W_fb[k*n+i, col] = 1.0
                    col += 1
        return {"U": [[float(x) for x in row] for row in U_fb],
                "V": [[float(x) for x in row] for row in V_fb],
                "W": [[float(x) for x in row] for row in W_fb]}
    
    best_rank = sb_rank
    best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    # Try to find lower rank using random search
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 15:
            break
        
        max_attempts = min(15, max(3, 120 // target_rank))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + target_rank * 1000 + attempt)
            
            # Random initialization with small values
            U_rand = np.zeros((nm, target_rank))
            V_rand = np.zeros((mp, target_rank))
            
            for c in range(target_rank):
                u_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nm)
                v_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=mp)
                u_mask = np.random.random(nm) < 0.5
                v_mask = np.random.random(mp) < 0.5
                U_rand[:, c] = np.where(u_mask, u_vals, 0.0)
                V_rand[:, c] = np.where(v_mask, v_vals, 0.0)
            
            # Form Khatri-Rao matrix
            K = np.zeros((nm * mp, target_rank))
            for c in range(target_rank):
                K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
            
            # Solve K * W^T = T_flat
            try:
                W_sol, residuals, rank_K, sv = np.linalg.lstsq(K, T_flat, rcond=1e-12)
            except np.linalg.LinAlgError:
                continue
            
            residual = np.abs(K @ W_sol - T_flat).max()
            
            if residual < 1e-9:
                W_check = W_sol * 2
                if np.allclose(W_check, np.round(W_check), atol=1e-6):
                    W_final = np.round(W_sol.T * 2) / 2
                    
                    recon = np.zeros((nm, mp, pn))
                    for c in range(target_rank):
                        recon += np.einsum('a,b,c->abc', U_rand[:,c], V_rand[:,c], W_final[:,c])
                    if np.abs(recon - T).max() < 1e-8:
                        best_rank = target_rank
                        best_decomp = (U_rand.copy(), V_rand.copy(), W_final)
                        break
        
        if best_rank < sb_rank:
            break
    
    # Tensor-train style search: exploit block structure with 3x3x4 sub-tensors
    if time.time() < deadline - 20:
        # Try to find a 3x3x4 decomposition of rank <= 19
        for r4 in range(19, 12, -1):
            if time.time() > deadline - 15:
                break
            
            # Build 3x3x4 target tensor
            T4 = np.zeros((nm, m*4, 4*n))
            for i in range(n):
                for j in range(m):
                    for k in range(4):
                        T4[i*m+j, j*4+k, k*n+i] = 1.0
            T4_flat = T4.reshape(nm * (m*4), 4*n)
            
            for attempt in range(10):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(777 + r4 * 100 + attempt)
                U4 = np.zeros((nm, r4))
                V4 = np.zeros((m*4, r4))
                
                for c in range(r4):
                    u_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nm)
                    v_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=m*4)
                    u_mask = np.random.random(nm) < 0.5
                    v_mask = np.random.random(m*4) < 0.5
                    U4[:, c] = np.where(u_mask, u_vals, 0.0)
                    V4[:, c] = np.where(v_mask, v_vals, 0.0)
                
                K4 = np.zeros((nm * (m*4), r4))
                for c in range(r4):
                    K4[:, c] = np.outer(U4[:, c], V4[:, c]).flatten()
                
                try:
                    W4_sol, _, _, _ = np.linalg.lstsq(K4, T4_flat, rcond=1e-12)
                except:
                    continue
                
                res4 = np.abs(K4 @ W4_sol - T4_flat).max()
                if res4 < 1e-9:
                    W4_check = W4_sol * 2
                    if np.allclose(W4_check, np.round(W4_check), atol=1e-6):
                        W4_final = np.round(W4_sol.T * 2) / 2
                        # Verify 3x3x4
                        recon4 = np.zeros((nm, m*4, 4*n))
                        for c in range(r4):
                            recon4 += np.einsum('a,b,c->abc', U4[:,c], V4[:,c], W4_final[:,c])
                        if np.abs(recon4 - T4).max() < 1e-8:
                            # Build 3x3x8 from 3x3x4 x 1x1x2
                            rank8 = r4 * 2
                            U8 = np.zeros((nm, rank8))
                            V8 = np.zeros((mp, rank8))
                            W8 = np.zeros((pn, rank8))
                            
                            for c4 in range(r4):
                                for c2 in range(2):
                                    c8 = c4 * 2 + c2
                                    U8[:, c8] = U4[:, c4]
                                    for j in range(m):
                                        for k4 in range(4):
                                            k = k4 + 4 * c2
                                            V8[j*p + k, c8] = V4[j*4 + k4, c4]
                                    for k4 in range(4):
                                        for i in range(n):
                                            k = k4 + 4 * c2
                                            W8[k*n + i, c8] = W4_final[k4*n + i, c4]
                            
                            # Verify
                            recon8 = np.zeros((nm, mp, pn))
                            for c8 in range(rank8):
                                recon8 += np.einsum('a,b,c->abc', U8[:,c8], V8[:,c8], W8[:,c8])
                            if np.abs(recon8 - T).max() < 1e-8:
                                if rank8 < best_rank:
                                    best_rank = rank8
                                    best_decomp = (U8.copy(), V8.copy(), W8.copy())
                            break
            if best_rank < sb_rank:
                break
    
    # Additional: try structured decomposition using 2x3 blocks with Strassen on overlapping
    if time.time() < deadline - 10 and best_rank >= 60:
        # Try a different partitioning: A as 2x3, B as 3x8
        # Use Strassen on 2x2 sub-blocks of A with full width of B
        U_t2 = []
        V_t2 = []
        W_t2 = []
        
        def strassen_full_width(row_off_a, col_off_a, row_off_b):
            """Strassen for 2x2 A block times 2x8 B block (two 2x4 sub-blocks)."""
            # This is equivalent to two separate Strassen decompositions
            # So we just call the standard one twice
            pass
        
        # Try 3x3x8 as three 1x3x8 strips
        # Each 1x3x8 has rank 3 (since it's just outer product of A[i,:] with B)
        # But we can do better by sharing structure
        
        # Alternative: use the fact that 3x3x8 = 3x3x4 + 3x3x4 (shifted)
        # If we find rank r for 3x3x4, we get rank 2r for 3x3x8
        # But we might do better with a single decomposition
        
        # Try random search with structured initialization
        for target_rank in range(min(best_rank - 1, 67), 50, -1):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(5):
                if time.time() > deadline - 3:
                    break
                
                np.random.seed(999 + target_rank * 100 + attempt)
                
                # Initialize with Strassen-like structure
                U_rand = np.zeros((nm, target_rank))
                V_rand = np.zeros((mp, target_rank))
                
                # First 28 entries: Strassen for A[0:2,0:2] x B[0:2,:]
                for b_col in range(0, 8, 2):
                    base = b_col // 2
                    # m1
                    U_rand[0, base*7+0] = 1; U_rand[3, base*7+0] = 1
                    V_rand[b_col, base*7+0] = 1; V_rand[b_col+1, base*7+0] = 1
                    W_rand_val = 1
                    # ... this is getting complex, let's just use random
                    
                # Pure random with denser initialization
                for c in range(target_rank):
                    u_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nm)
                    v_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=mp)
                    u_mask = np.random.random(nm) < 0.6
                    v_mask = np.random.random(mp) < 0.6
                    U_rand[:, c] = np.where(u_mask, u_vals, 0.0)
                    V_rand[:, c] = np.where(v_mask, v_vals, 0.0)
                
                K = np.zeros((nm * mp, target_rank))
                for c in range(target_rank):
                    K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
                
                try:
                    W_sol, residuals, rank_K, sv = np.linalg.lstsq(K, T_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K @ W_sol - T_flat).max()
                
                if residual < 1e-9:
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-6):
                        W_final = np.round(W_sol.T * 2) / 2
                        
                        recon = np.zeros((nm, mp, pn))
                        for c in range(target_rank):
                            recon += np.einsum('a,b,c->abc', U_rand[:,c], V_rand[:,c], W_final[:,c])
                        if np.abs(recon - T).max() < 1e-8:
                            best_rank = target_rank
                            best_decomp = (U_rand.copy(), V_rand.copy(), W_final)
                            break
            
            if best_rank < sb_rank - 5:
                break
    
    U_out, V_out, W_out = best_decomp
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
