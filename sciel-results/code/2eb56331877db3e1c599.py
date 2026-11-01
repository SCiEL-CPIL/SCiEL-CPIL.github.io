"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Build rank-42 Strassen decomposition
    def build_strassen_rank42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_products = [
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1)]},
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1)]},
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, -1), (3, 1)]},
        ]

        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                for idx, coeff in strassen_products[s]['U']:
                    U[idx, col] = coeff
                for idx, coeff in strassen_products[s]['V']:
                    j = idx // 2
                    k_local = idx % 2
                    k = 2 * t + k_local
                    V[j * p + k, col] = coeff
                for idx, coeff in strassen_products[s]['W']:
                    k_local = idx // 2
                    i = idx % 2
                    k = 2 * t + k_local
                    W[k * n + i, col] = coeff

        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
        err = np.linalg.norm(T_rec - T)
        return err, U, V, W

    err42, U42, V42, W42 = build_strassen_rank42()
    if err42 >= 1e-9:
        # Fallback schoolbook
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = 42
    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()

    # Strategy: iterative rank reduction via term merging
    # Try to replace pairs of rank-1 terms with a single rank-1 term
    # that reproduces their combined contribution.
    
    def verify(U, V, W):
        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
        return np.linalg.norm(T_rec - T)

    def try_merge_terms(U, V, W, rank, max_time=30):
        """Try to reduce rank by merging pairs of terms.
        For each pair (r1, r2), check if there exists a single rank-1 term
        that equals the sum of the two terms' contributions.
        A rank-1 term is u*v*w. The sum of two rank-1 terms is u1*v1*w1 + u2*v2*w2.
        This can be a rank-1 term only in special cases.
        
        More practically: try to find a new set of (rank-1) terms that
        reproduces the same tensor by solving a small linear system.
        """
        if time.time() > deadline - 10:
            return None, None, None, rank

        # Represent current decomposition as list of (U_col, V_col, W_col)
        terms = [(U[:, r].copy(), V[:, r].copy(), W[:, r].copy()) for r in range(rank)]

        # Try all pairs of terms that share the same block structure
        # (i.e., same t block in the 6-block partition)
        # Terms in block t are at indices 7*t .. 7*t+6
        
        # First, try within-block merges
        for t in range(6):
            if time.time() > deadline - 15:
                break
            block_start = 7 * t
            block_terms = list(range(block_start, block_start + 7))
            
            for i in range(len(block_terms)):
                for j in range(i + 1, len(block_terms)):
                    if time.time() > deadline - 15:
                        break
                    r1 = block_terms[i]
                    r2 = block_terms[j]
                    
                    # Combined tensor contribution
                    T_comb = np.einsum('i,j,k->ijk', terms[r1][0], terms[r1][1], terms[r1][2]) + \
                             np.einsum('i,j,k->ijk', terms[r2][0], terms[r2][1], terms[r2][2])
                    
                    # Check if T_comb is rank-1
                    # Flatten and check if it's a tensor product
                    # A 3D array is rank-1 if it can be written as a*b*c
                    # Check: for any two non-zero entries in each mode, ratios must be consistent
                    Tc = T_comb
                    nz = np.nonzero(Tc)
                    if len(nz[0]) < 2:
                        continue
                    
                    # Pick a reference element
                    ref_idx = (nz[0][0], nz[1][0], nz[2][0])
                    ref_val = Tc[ref_idx]
                    if ref_val == 0:
                        continue
                    
                    # Check rank-1: Tc[a,b,c] = Tc[ref] * (Tc[a,ref_b,ref_c]/Tc[ref]) * ...
                    # Simpler: check if all 2x2 minors of matricizations are zero
                    # Matricize along first mode: shape (n*m, m*p * p*n)
                    mat = Tc.reshape(n * m, (m * p) * (p * n))
                    if mat.shape[0] >= 2 and mat.shape[1] >= 2:
                        # Check if rank <= 1 by looking at 2x2 minors
                        # Sample a few minors for speed
                        is_rank1 = True
                        checked = 0
                        for a1 in range(min(2, mat.shape[0])):
                            for a2 in range(a1 + 1, min(3, mat.shape[0])):
                                if checked > 20:
                                    break
                                # Find non-zero columns
                                nz_cols = np.nonzero(np.abs(mat[a1] + mat[a2]) > 1e-12)[0]
                                if len(nz_cols) >= 2:
                                    for c1_idx in range(min(2, len(nz_cols))):
                                        for c2_idx in range(c1_idx + 1, min(3, len(nz_cols))):
                                            c1, c2 = nz_cols[c1_idx], nz_cols[c2_idx]
                                            minor = mat[a1, c1] * mat[a2, c2] - mat[a1, c2] * mat[a2, c1]
                                            if abs(minor) > 1e-10:
                                                is_rank1 = False
                                            checked += 1
                                            if checked > 20:
                                                break
                                if not is_rank1:
                                    break
                            if not is_rank1:
                                break
                        if not is_rank1:
                            continue
                    
                    # If rank-1, extract the vectors
                    # Find non-zero entries
                    a_nz = np.unique(nz[0])
                    b_nz = np.unique(nz[1])
                    c_nz = np.unique(nz[2])
                    
                    if len(a_nz) == 0 or len(b_nz) == 0 or len(c_nz) == 0:
                        continue
                    
                    # u[a] = Tc[a, b0, c0] / (v[b0]*w[c0]) for some reference
                    # Use: Tc[a,b,c] = u[a]*v[b]*w[c]
                    # u[a] = Tc[a, b_ref, c_ref] / (v[b_ref]*w[c_ref])
                    # We need to find consistent u, v, w
                    # Set u[a_ref] = 1, then v[b] = Tc[a_ref, b, c_ref] / w[c_ref]
                    # This gets circular. Use SVD-like approach:
                    # Tc = u * v * w  =>  mat(Tc) has rank 1
                    # u = first column of mat, v*w = remaining factor
                    
                    # Actually, let's just try: fix u from first mode
                    # u[a] = Tc[a, b0, c0] for a fixed (b0, c0) where Tc != 0
                    # Then v[b]*w[c] = Tc[a0, b, c] / u[a0]
                    a0 = a_nz[0]
                    # Find b0, c0 where Tc[a0, b0, c0] != 0
                    sub = Tc[a0]
                    nz_sub = np.nonzero(sub)
                    if len(nz_sub[0]) == 0:
                        continue
                    b0, c0 = nz_sub[0][0], nz_sub[1][0]
                    u_vec = Tc[:, b0, c0]
                    # v[b]*w[c] = Tc[a0, b, c] / u_vec[a0]
                    vw = sub / u_vec[a0]
                    # Now factor vw = v * w
                    # This is a rank-1 matrix factorization
                    # v[b] = vw[b, c0_ref] for fixed c, w[c] = vw[b_ref, c] / v[b_ref]
                    nz_vw = np.nonzero(vw)
                    if len(nz_vw[0]) == 0:
                        continue
                    b0v = nz_vw[0][0]
                    c0w = nz_vw[1][0]
                    v_vec = vw[:, c0w]
                    w_vec = vw[b0v, :] / v_vec[b0v]
                    
                    # Verify
                    recon = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
                    if np.linalg.norm(recon - Tc) > 1e-8:
                        continue
                    
                    # Check if values are integer or half-integer
                    all_vals = np.concatenate([u_vec, v_vec, w_vec])
                    scaled = np.round(all_vals * 2)
                    if np.allclose(all_vals, scaled / 2.0, atol=1e-8):
                        u_new = scaled[:len(u_vec)] / 2.0
                        v_new = scaled[len(u_vec):len(u_vec)+len(v_vec)] / 2.0
                        w_new = scaled[len(u_vec)+len(v_vec):] / 2.0
                        
                        # Build new decomposition with this pair merged
                        new_rank = rank - 1
                        new_terms = []
                        for r in range(rank):
                            if r == r1 or r == r2:
                                continue
                            new_terms.append((terms[r][0], terms[r][1], terms[r][2]))
                        new_terms.append((u_new, v_new, w_new))
                        
                        new_U = np.zeros((n * m, new_rank))
                        new_V = np.zeros((m * p, new_rank))
                        new_W = np.zeros((p * n, new_rank))
                        for idx, (uu, vv, ww) in enumerate(new_terms):
                            new_U[:, idx] = uu
                            new_V[:, idx] = vv
                            new_W[:, idx] = ww
                        
                        err = verify(new_U, new_V, new_W)
                        if err < 1e-8:
                            return new_U, new_V, new_W, new_rank

        # Try cross-block merges: terms from different blocks that might combine
        # This is more complex but let's try a few random pairs
        np.random.seed(123)
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            r1 = np.random.randint(0, rank)
            r2 = np.random.randint(0, rank)
            if r1 == r2:
                continue
            
            T_comb = np.einsum('i,j,k->ijk', terms[r1][0], terms[r1][1], terms[r1][2]) + \
                     np.einsum('i,j,k->ijk', terms[r2][0], terms[r2][1], terms[r2][2])
            
            # Quick rank check via matricization
            mat = T_comb.reshape(n * m, (m * p) * (p * n))
            if mat.shape[0] < 2:
                continue
            # Check a few 2x2 minors
            is_rank1 = True
            for _ in range(5):
                a1 = np.random.randint(0, mat.shape[0])
                a2 = np.random.randint(0, mat.shape[0])
                if a1 == a2:
                    continue
                row1 = mat[a1]
                row2 = mat[a2]
                nz = np.nonzero(np.abs(row1) + np.abs(row2) > 1e-12)[0]
                if len(nz) < 2:
                    continue
                c1 = nz[np.random.randint(0, len(nz))]
                c2 = nz[np.random.randint(0, len(nz))]
                if c1 == c2:
                    continue
                minor = row1[c1] * row2[c2] - row1[c2] * row2[c1]
                if abs(minor) > 1e-10:
                    is_rank1 = False
                    break
            
            if not is_rank1:
                continue
            
            # Extract vectors
            a_nz = np.unique(np.nonzero(T_comb)[0])
            if len(a_nz) == 0:
                continue
            a0 = a_nz[0]
            sub = T_comb[a0]
            nz_sub = np.nonzero(sub)
            if len(nz_sub[0]) == 0:
                continue
            b0, c0 = nz_sub[0][0], nz_sub[1][0]
            u_vec = T_comb[:, b0, c0]
            vw = sub / u_vec[a0]
            nz_vw = np.nonzero(vw)
            if len(nz_vw[0]) == 0:
                continue
            b0v = nz_vw[0][0]
            c0w = nz_vw[1][0]
            v_vec = vw[:, c0w]
            w_vec = vw[b0v, :] / v_vec[b0v]
            
            recon = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
            if np.linalg.norm(recon - T_comb) > 1e-8:
                continue
            
            all_vals = np.concatenate([u_vec, v_vec, w_vec])
            scaled = np.round(all_vals * 2)
            if not np.allclose(all_vals, scaled / 2.0, atol=1e-8):
                continue
            
            u_new = scaled[:len(u_vec)] / 2.0
            v_new = scaled[len(u_vec):len(u_vec)+len(v_vec)] / 2.0
            w_new = scaled[len(u_vec)+len(v_vec):] / 2.0
            
            new_rank = rank - 1
            new_terms = []
            for r in range(rank):
                if r == r1 or r == r2:
                    continue
                new_terms.append((terms[r][0], terms[r][1], terms[r][2]))
            new_terms.append((u_new, v_new, w_new))
            
            new_U = np.zeros((n * m, new_rank))
            new_V = np.zeros((m * p, new_rank))
            new_W = np.zeros((p * n, new_rank))
            for idx, (uu, vv, ww) in enumerate(new_terms):
                new_U[:, idx] = uu
                new_V[:, idx] = vv
                new_W[:, idx] = ww
            
            err = verify(new_U, new_V, new_W)
            if err < 1e-8:
                return new_U, new_V, new_W, new_rank

        return None, None, None, rank

    # Iteratively try to reduce rank
    current_U, current_V, current_W = U42.copy(), V42.copy(), W42.copy()
    current_rank = 42
    
    for iteration in range(10):
        if time.time() > deadline - 20:
            break
        if current_rank <= 35:
            break
        
        new_U, new_V, new_W, new_rank = try_merge_terms(current_U, current_V, current_W, current_rank)
        
        if new_U is not None and new_rank < current_rank:
            current_U, current_V, current_W = new_U, new_V, new_W
            current_rank = new_rank
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        else:
            break

    # Alternative strategy: try a completely different construction
    # Use the fact that 2x2x12 can be seen as 2x2x(6*2) and try
    # a rank-14 per-pair construction with shared structure
    
    if time.time() < deadline - 30:
        # Try: can we do better than 7 per 2x2x2 block by sharing across blocks?
        # Idea: use a 2x2x6 block with rank < 21 (less than 3*7)
        # Actually, let's try a direct approach: assume the decomposition has
        # a Kronecker-like structure where the 12 columns are handled by
        # a smaller set of "template" terms
        
        # Try rank-36: 6 blocks of rank-6 each
        # For 2x2x2, the minimum rank is 7 (Strassen), so rank-6 per block is impossible.
        # But maybe we can do 2 blocks of rank-10 + 4 blocks of rank-4? No, rank-4 < 7.
        
        # Let's try: 3 blocks of 2x2x4. Each 2x2x4 has some minimum rank.
        # The tensor for 2x2x4 multiplication:
        # T[a,b,c] = 1 if a=i*m+j, b=j*4+k, c=k*2+i for i in 0,1; j in 0,1; k in 0,1,2,3
        # This is a 4x8x8 tensor.
        
        # Known: 2x2x4 multiplication. The rank is at least 8 (since output is 2x4=8 entries
        # and each rank-1 term can contribute to at most... actually no, a rank-1 term 
        # u*v*w where u is 4-dim, v is 8-dim, w is 8-dim can have many non-zeros.
        
        # Let me try to construct a rank-10 solution for 2x2x4 by hand:
        # The 2x2x4 tensor has entries T[i*2+j, j*4+k, k*2+i] = 1
        # For i=0,j=0: k=0,1,2,3 -> T[0, k, 2k] = 1
        # For i=0,j=1: k=0,1,2,3 -> T[1, 4+k, 2k] = 1
        # For i=1,j=0: k=0,1,2,3 -> T[2, k, 2k+1] = 1
        # For i=1,j=1: k=0,1,2,3 -> T[3, 4+k, 2k+1] = 1
        
        # So the non-zero entries are:
        # (0,0,0), (0,1,2), (0,2,4), (0,3,6)
        # (1,4,0), (1,5,2), (1,6,4), (1,7,6)
        # (2,0,1), (2,1,3), (2,2,5), (2,3,7)
        # (3,4,1), (3,5,3), (3,6,5), (3,7,7)
        
        # Total 16 non-zero entries.
        # A rank-1 term u*v*w can cover at most |supp(u)|*|supp(v)|*|supp(w)| entries.
        # To cover all 16 entries with rank r, we need r terms.
        
        # Let's try to find a rank-8 solution (matching the output dimension).
        # Actually, the lower bound on rank is max of matricization ranks.
        # Matricize T as (4, 64): rank is at most 4.
        # Matricize as (32, 8): rank at most 8.
        # Matricize as (4, 8, 8) -> (4*8, 8) = (32, 8): rank at most 8.
        # So rank >= 4 and rank >= 8 from different matricizations? Let me check.
        
        # T reshaped to (4, 8*8=64): each row i*2+j has non-zeros at specific positions.
        # Row 0: cols 0, 2, 4, 6 (positions k*8+2k for k=0,1,2,3 in the 64-dim)
        # Wait, let me be more careful. T[a,b,c] with a in 0..3, b in 0..7, c in 0..7.
        # Reshape to (4, 64): index (a, b*8+c)
        # Row a=0: non-zero at (b,c) = (0,0),(1,2),(2,4),(3,6) -> cols 0, 1*8+2=10, 2*8+4=20, 3*8+6=30
        # Row a=1: non-zero at (4,0),(5,2),(6,4),(7,6) -> cols 32, 42, 52, 62
        # Row a=2: non-zero at (0,1),(1,3),(2,5),(3,7) -> cols 1, 11, 21, 31
        # Row a=3: non-zero at (4,1),(5,3),(6,5),(7,7) -> cols 33, 43, 53, 63
        
        # These 4 rows are linearly independent (each has non-zeros in disjoint column sets),
        # so rank of this matricization is 4.
        
        # Reshape to (4*8=32, 8): index (a*8+b, c)
        # For c=0: non-zero at a*8+b where (a,b) in {(0,0),(1,4),(2,0),(3,4)} -> rows 0, 12, 16, 28
        # For c=1: (2,0),(3,4) -> rows 16, 28
        # For c=2: (0,1),(1,5) -> rows 1, 13
        # For c=3: (2,1),(3,5) -> rows 17, 29
        # For c=4: (0,2),(1,6) -> rows 2, 14
        # For c=5: (2,2),(3,6) -> rows 18, 30
        # For c=6: (0,3),(1,7) -> rows 3, 15
        # For c=7: (2,3),(3,7) -> rows 19, 31
        
        # The matrix is 32x8. Let's see its rank.
        # Columns: c=0 has entries at rows 0,12,16,28
        # c=1 at 16,28
        # c=2 at 1,13
        # c=3 at 17,29
        # c=4 at 2,14
        # c=5 at 18,30
        # c=6 at 3,15
        # c=7 at 19,31
        
        # Are these 8 columns linearly independent?
        # col0 = e0+e12+e16+e28
        # col1 = e16+e28
        # col2 = e1+e13
        # col3 = e17+e29
        # col4 = e2+e14
        # col5 = e18+e30
        # col6 = e3+e15
        # col7 = e19+e31
        
        # col0 - col1 = e0+e12
        # So we have e0+e12, e16+e28, e1+e13, e17+e29, e2+e14, e18+e30, e3+e15, e19+e31
        # These 8 vectors are linearly independent (each involves disjoint pairs of basis vectors).
        # So rank is 8.
        
        # Therefore rank(2x2x4) >= 8.
        # Can we achieve rank 8? Let's try.
        
        # We need 8 rank-1 terms whose sum gives T.
        # Each term r has U_r (4-dim), V_r (8-dim), W_r (8-dim).
        # The constraint is: for each (a,b,c) in the support, sum_r U_r[a]*V_r[b]*W_r[c] = 1
        # and for (a,b,c) not in support, sum_r U_r[a]*V_r[b]*W_r[c] = 0.
        
        # This is a system of 4*8*8 = 256 equations (but only 16 are non-zero constraints,
        # the other 240 are zero constraints) in 8*(4+8+8) = 160 unknowns.
        # Highly overdetermined, but the structure might allow a solution.
        
        # Let me try a structured approach:
        # Notice the pattern: T[a,b,c] depends on a, b, c through:
        # a = i*2+j, b = j*4+k, c = k*2+i
        # So i = a mod 2, j = a div 2, k = b - j*4, and we need c = k*2+i.
        
        # Let's try: for each k in 0,1,2,3, define a term that handles column k.
        # Term for column k: U_k[i*2+j] = 1 for all i,j (constant in a), 
        # V_k[j*4+k'] = 1 if k'=k else 0, W_k[k'*2+i] = 1 if k'=k else 0.
        # But this gives U_k as all-ones, V_k as one-hot, W_k as one-hot.
        # Then sum_k U_k[a]*V_k[b]*W_k[c] = sum_k 1 * [b=j*4+k] * [c=k*2+i]
        # For a = i*2+j: = [c = ((b-j*4))*2 + i] where b-j*4 = k.
        # This equals 1 iff c = k*2+i where k = b-j*4. Which is exactly our condition!
        # So this gives rank 4, not 8!
        
        # Wait, let me recheck. U_k is 4-dim (a index 0..3). If U_k[a] = 1 for all a,
        # V_k[b] = 1 if b = j*4+k for some j (i.e., b mod 4 = k), else 0.
        # W_k[c] = 1 if c mod 2 = i for the right i... 
        # Hmm, W_k[c] needs to depend on both k and i, but i is determined by a.
        
        # Actually, the issue is that U_k can't be constant because different i values
        # need different W patterns. Let me reconsider.
        
        # For a fixed k, the non-zero entries in T with b = j*4+k are:
        # (a=0*2+0=0, b=k, c=k*2+0=k*2)
        # (a=0*2+1=1, b=4+k, c=k*2+0=k*2)  -- wait, c = k*2 + i, i=0 for a=0
        # (a=1*2+0=2, b=k, c=k*2+1)
        # (a=1*2+1=3, b=4+k, c=k*2+1)
        
        # So for fixed k:
        # T[0, k, 2k] = 1, T[1, 4+k, 2k] = 1
        # T[2, k, 2k+1] = 1, T[3, 4+k, 2k+1] = 1
        
        # A single rank-1 term can't cover all 4 of these unless U, V, W are specifically structured.
        # Term: U[a]*V[b]*W[c]
        # For (0,k,2k): U[0]*V[k]*W[2k] = 1
        # For (1,4+k,2k): U[1]*V[4+k]*W[2k] = 1
        # For (2,k,2k+1): U[2]*V[k]*W[2k+1] = 1
        # For (3,4+k,2k+1): U[3]*V[4+k]*W[2k+1] = 1
        
        # From first two: U[0]*V[k]*W[2k] = U[1]*V[4+k]*W[2k] => U[0]*V[k] = U[1]*V[4+k]
        # From last two: U[2]*V[k]*W[2k+1] = U[3]*V[4+k]*W[2k+1] => U[2]*V[k] = U[3]*V[4+k]
        # From first and third: U[0]*V[k]*W[2k] = U[2]*V[k]*W[2k+1] => U[0]*W[2k] = U[2]*W[2k+1]
        
        # Let U[0]=U[1]=U[2]=U[3]=1, V[k]=V[4+k]=1, W[2k]=W[2k+1]=1.
        # Then all four are 1. But this term also contributes to other (a,b,c) entries!
        # U[a]=1 for all a, V[b]=1 if b mod 4 == k, W[c]=1 if c in {2k, 2k+1}.
        # So this term contributes 1 to ALL entries where b mod 4 = k and c in {2k, 2k+1}.
        # That includes entries that should be 0 in T.
        
        # For example, (a=0, b=k, c=2k+1): should be 0 (since c should be 2k for i=0),
        # but our term gives 1. So we need to cancel this.
        
        # This is getting complex. Let me just try to solve it numerically with a small rank.
        
        def solve_2x2x4(rank_sub):
            """Try to find rank_sub decomposition for 2x2x4."""
            if time.time() > deadline - 20:
                return None
            
            n_s, m_s, p_s = 2, 2, 4
            T_s = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        a = i * m_s + j
                        b = j * p_s + k
                        c = k * n_s + i
                        T_s[a, b, c] = 1.0
            
            # Parameters: U (4, rank), V (8, rank), W (8, rank)
            n_params = 4 * rank_sub + 8 * rank_sub + 8 * rank_sub
            
            def make_objective(x):
                U_s = x[:4*rank_sub].reshape(4, rank_sub)
                V_s = x[4*rank_sub:12*rank_sub].reshape(8, rank_sub)
                W_s = x[12*rank_sub:].reshape(8, rank_sub)
                T_rec = np.zeros_like(T_s)
                for r in range(rank_sub):
                    T_rec += np.einsum('i,j,k->ijk', U_s[:, r], V_s[:, r], W_s[:, r])
                return (T_rec - T_s).flatten()
            
            # Try several random initializations
            np.random.seed(777)
            best_err = np.inf
            best_x = None
            
            for attempt in range(20):
                if time.time() > deadline - 20:
                    break
                
                # Sparse random init
                x0 = np.zeros(n_params)
                for r in range(rank_sub):
                    u_nz = np.random.choice(4, size=np.random.randint(1, 4), replace=False)
                    v_nz = np.random.choice(8, size=np.random.randint(1, 6), replace=False)
                    w_nz = np.random.choice(8, size=np.random.randint(1, 6), replace=False)
                    x0[u_nz * rank_sub + r] = np.random.choice([-1, 1], size=len(u_nz))
                    x0[4*rank_sub + v_nz * rank_sub + r] = np.random.choice([-1, 1], size=len(v_nz))
                    x0[12*rank_sub + w_nz * rank_sub + r] = np.random.choice([-1, 1], size=len(w_nz))
                
                try:
                    import scipy.optimize
                    result = scipy.optimize.least_squares(
                        lambda z: make_objective(z), x0, method='trf',
                        max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14
                    )
                    err = np.linalg.norm(result.fun)
                    if err < best_err:
                        best_err = err
                        best_x = result.x.copy()
                        if err < 1e-10:
                            break
                except Exception:
                    continue
            
            if best_x is None or best_err > 1e-8:
                return None
            
            U_s = best_x[:4*rank_sub].reshape(4, rank_sub)
            V_s = best_x[4*rank_sub:12*rank_sub].reshape(8, rank_sub)
            W_s = best_x[12*rank_sub:].reshape(8, rank_sub)
            
            # Snap to half-integers
            U_snap = np.round(U_s * 2) / 2
            V_snap = np.round(V_s * 2) / 2
            W_snap = np.round(W_s * 2) / 2
            
            T_rec = np.zeros_like(T_s)
            for r in range(rank_sub):
                T_rec += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            
            if np.linalg.norm(T_rec - T_s) < 1e-9:
                return U_snap, V_snap, W_snap
            
            # If snap failed, check if original is close enough
            if best_err < 1e-6:
                return U_s, V_s, W_s
            
            return None
        
        # Try rank 8, 9, 10 for 2x2x4
        for rk in [8, 9, 10]:
            if time.time() > deadline - 25:
                break
            res = solve_2x2x4(rk)
            if res is not None:
                U_s, V_s, W_s = res
                # Build full 2x2x12 by replicating 3 times
                rank_total = 3 * rk
                U_total = np.zeros((n * m, rank_total))
                V_total = np.zeros((m * p, rank_total))
                W_total = np.zeros((p * n, rank_total))
                
                for g in range(3):
                    for r in range(rk):
                        col = rk * g + r
                        U_total[:, col] = U_s[:, r]
                        for b_idx in range(8):
                            j = b_idx // 4
                            k_local = b_idx % 4
                            k = 4 * g + k_local
                            V_total[j * p + k, col] = V_s[b_idx, r]
                        for c_idx in range(8):
                            k_local = c_idx // 2
                            i = c_idx % 2
                            k = 4 * g + k_local
                            W_total[k * n + i, col] = W_s[c_idx, r]
                
                err = verify(U_total, V_total, W_total)
                if err < 1e-8 and rank_total < best_rank:
                    best_rank = rank_total
                    best_U, best_V, best_W = U_total.copy(), V_total.copy(), W_total.copy()
                break  # Only try the first successful rank
        
        # Also try 2x2x3 blocks (4 blocks of 3)
        if time.time() < deadline - 30:
            def solve_2x2x3(rank_sub):
                if time.time() > deadline - 20:
                    return None
                
                n_s, m_s, p_s = 2, 2, 3
                T_s = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            a = i * m_s + j
                            b = j * p_s + k
                            c = k * n_s + i
                            T_s[a, b, c] = 1.0
                
                n_params = 4 * rank_sub + 6 * rank_sub + 6 * rank_sub
                
                def make_objective(x):
                    U_s = x[:4*rank_sub].reshape(4, rank_sub)
                    V_s = x[4*rank_sub:10*rank_sub].reshape(6, rank_sub)
                    W_s = x[10*rank_sub:].reshape(6, rank_sub)
                    T_rec = np.zeros_like(T_s)
                    for r in range(rank_sub):
                        T_rec += np.einsum('i,j,k->ijk', U_s[:, r], V_s[:, r], W_s[:, r])
                    return (T_rec - T_s).flatten()
                
                np.random.seed(999)
                best_err = np.inf
                best_x = None
                
                for attempt in range(20):
                    if time.time() > deadline - 20:
                        break
                    
                    x0 = np.zeros(n_params)
                    for r in range(rank_sub):
                        u_nz = np.random.choice(4, size=np.random.randint(1, 4), replace=False)
                        v_nz = np.random.choice(6, size=np.random.randint(1, 5), replace=False)
                        w_nz = np.random.choice(6, size=np.random.randint(1, 5), replace=False)
                        x0[u_nz * rank_sub + r] = np.random.choice([-1, 1], size=len(u_nz))
                        x0[4*rank_sub + v_nz * rank_sub + r] = np.random.choice([-1, 1], size=len(v_nz))
                        x0[10*rank_sub + w_nz * rank_sub + r] = np.random.choice([-1, 1], size=len(w_nz))
                    
                    try:
                        import scipy.optimize
                        result = scipy.optimize.least_squares(
                            lambda z: make_objective(z), x0, method='trf',
                            max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14
                        )
                        err = np.linalg.norm(result.fun)
                        if err < best_err:
                            best_err = err
                            best_x = result.x.copy()
                            if err < 1e-10:
                                break
                    except Exception:
                        continue
                
                if best_x is None or best_err > 1e-8:
                    return None
                
                U_s = best_x[:4*rank_sub].reshape(4, rank_sub)
                V_s = best_x[4*rank_sub:10*rank_sub].reshape(6, rank_sub)
                W_s = best_x[10*rank_sub:].reshape(6, rank_sub)
                
                U_snap = np.round(U_s * 2) / 2
                V_snap = np.round(V_s * 2) / 2
                W_snap = np.round(W_s * 2) / 2
                
                T_rec = np.zeros_like(T_s)
                for r in range(rank_sub):
                    T_rec += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                
                if np.linalg.norm(T_rec - T_s) < 1e-9:
                    return U_snap, V_snap, W_snap
                
                if best_err < 1e-6:
                    return U_s, V_s, W_s
                
                return None
            
            # 2x2x3: known rank is 5 (from literature: rank(2,2,3)=5)
            for rk in [5, 6, 7]:
                if time.time() > deadline - 25:
                    break
                res = solve_2x2x3(rk)
                if res is not None:
                    U_s, V_s, W_s = res
                    rank_total = 4 * rk
                    U_total = np.zeros((n * m, rank_total))
                    V_total = np.zeros((m * p, rank_total))
                    W_total = np.zeros((p * n, rank_total))
                    
                    for g in range(4):
                        for r in range(rk):
                            col = rk * g + r
                            U_total[:, col] = U_s[:, r]
                            for b_idx in range(6):
                                j = b_idx // 3
                                k_local = b_idx % 3
                                k = 3 * g + k_local
                                V_total[j * p + k, col] = V_s[b_idx, r]
                            for c_idx in range(6):
                                k_local = c_idx // 2
                                i = c_idx % 2
                                k = 3 * g + k_local
                                W_total[k * n + i, col] = W_s[c_idx, r]
                    
                    err = verify(U_total, V_total, W_total)
                    if err < 1e-8 and rank_total < best_rank:
                        best_rank = rank_total
                        best_U, best_V, best_W = U_total.copy(), V_total.copy(), W_total.copy()
                    break

    # Final verification
    if time.time() < deadline - 5:
        final_err = verify(best_U, best_V, best_W)
        if final_err < 1e-8:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    
    # Ultimate fallback
    return {
        "U": U42.tolist(),
        "V": V42.tolist(),
        "W": W42.tolist()
    }
# EVOLVE-BLOCK-END
