"""Strassen block-wise matrix multiplication for 4x5 * 5x6."""
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    # We'll use block decomposition with Strassen for 2x2x2 sub-blocks
    # and simple multiplication for other sizes.
    
    # Strategy: split into blocks where we can use Strassen (2x2x2)
    # and simple schoolbook for the rest.
    
    # Let's think about a clean block decomposition.
    # We can group rows of A, columns of A = rows of B, columns of B.
    
    # A is 4x5, B is 5x6, C is 4x6.
    
    # Split A into two 2x5 blocks (top and bottom rows)
    # Split B into two 5x3 blocks (left and right columns)
    # This gives 4 sub-products, each needing rank...
    
    # Actually, let's use a more systematic approach.
    # We'll enumerate all (i,j,k) triples and use Strassen to compress
    # groups of 8 products (2x2x2) into 7 terms.
    
    # For 4x5x6, total schoolbook rank is 120.
    # We can group into 2x2x2 blocks where possible.
    # 
    # i ranges 0-3, j ranges 0-4, k ranges 0-5.
    # 
    # A 2x2x2 block needs even ranges for i, j, k.
    # i: can form blocks [0,1] and [2,3] -> 2 blocks
    # j: can form blocks [0,1] and [2,3], plus [4] leftover -> 2 blocks + 1
    # k: can form blocks [0,1], [2,3], [4,5] -> 3 blocks
    #
    # So 2 x 2 x 3 = 12 full 2x2x2 blocks, each Strassen rank 7 -> 84
    # Plus 2 x 1 x 3 = 6 blocks of size 2x1x2, rank 4 each -> 24
    # Total: 84 + 24 = 108
    
    # Let's implement this.
    
    rank_total = 0
    U = []
    V = []
    W = []
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen factors for C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2]"""
        # Strassen for 2x2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        global rank_total
        # Each term: (u_factor, v_factor, w_factor)
        # u_factor is a linear combination of a,b,c,d
        # v_factor is a linear combination of e,f,g,h
        # w_factor is a linear combination of C entries
        
        a_idx = (i0) * m + j0       # A[0,0] local
        b_idx = (i0) * m + (j0+1)   # A[0,1] local
        c_idx = (i0+1) * m + j0     # A[1,0] local
        d_idx = (i0+1) * m + (j0+1) # A[1,1] local
        
        e_idx = j0 * p + k0         # B[0,0] local
        f_idx = j0 * p + (k0+1)     # B[0,1] local
        g_idx = (j0+1) * p + k0     # B[1,0] local
        h_idx = (j0+1) * p + (k0+1) # B[1,1] local
        
        # C indices (using W[k*n+i] convention)
        C00 = k0 * n + i0           # C[0,0]
        C01 = (k0+1) * n + i0       # C[0,1]
        C10 = k0 * n + (i0+1)       # C[1,0]
        C11 = (k0+1) * n + (i0+1)   # C[1,1]
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        terms = []
        
        # M1 = (a+d)(e+h): u={a:1,d:1}, v={e:1,h:1}, w={C00:1,C11:1}
        terms.append(([(a_idx,1),(d_idx,1)], [(e_idx,1),(h_idx,1)], [(C00,1),(C11,1)]))
        # M2 = (c+d)e: u={c:1,d:1}, v={e:1}, w={C10:1}
        terms.append(([(c_idx,1),(d_idx,1)], [(e_idx,1)], [(C10,1)]))
        # M3 = a(f-h): u={a:1}, v={f:1,h:-1}, w={C01:1}
        terms.append(([(a_idx,1)], [(f_idx,1),(h_idx,-1)], [(C01,1)]))
        # M4 = d(g-e): u={d:1}, v={g:1,e:-1}, w={C00:1,C10:1}
        terms.append(([(d_idx,1)], [(g_idx,1),(e_idx,-1)], [(C00,1),(C10,1)]))
        # M5 = (a+b)h: u={a:1,b:1}, v={h:1}, w={C00:-1}
        terms.append(([(a_idx,1),(b_idx,1)], [(h_idx,1)], [(C00,-1)]))
        # M6 = (c-a)(e+f): u={c:1,a:-1}, v={e:1,f:1}, w={C11:1}
        terms.append(([(c_idx,1),(a_idx,-1)], [(e_idx,1),(f_idx,1)], [(C11,1)]))
        # M7 = (b-d)(g+h): u={b:1,d:-1}, v={g:1,h:1}, w={C00:1}
        terms.append(([(b_idx,1),(d_idx,-1)], [(g_idx,1),(h_idx,1)], [(C00,1)]))
        
        for u_terms, v_terms, w_terms in terms:
            rank_total += 1
            u_row = [0] * rank_total
            v_row = [0] * rank_total
            w_row = [0] * rank_total
            for idx, val in u_terms:
                u_row[idx] = val
            for idx, val in v_terms:
                v_row[idx] = val
            for idx, val in w_terms:
                w_row[idx] = val
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
    
    def add_simple_2x1x2(i0, j0, k0):
        """Add schoolbook factors for C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+1] @ B[j0:j0+1, k0:k0+2]"""
        # A is 2x1: [[a],[c]], B is 1x2: [[e,f]]
        # C = [[a*e, a*f], [c*e, c*f]]
        global rank_total
        a_idx = (i0) * m + j0
        c_idx = (i0+1) * m + j0
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0+1)
        
        C00 = k0 * n + i0
        C01 = (k0+1) * n + i0
        C10 = k0 * n + (i0+1)
        C11 = (k0+1) * n + (i0+1)
        
        terms = [
            ([(a_idx,1)], [(e_idx,1)], [(C00,1)]),
            ([(a_idx,1)], [(f_idx,1)], [(C01,1)]),
            ([(c_idx,1)], [(e_idx,1)], [(C10,1)]),
            ([(c_idx,1)], [(f_idx,1)], [(C11,1)]),
        ]
        
        for u_terms, v_terms, w_terms in terms:
            rank_total += 1
            u_row = [0] * rank_total
            v_row = [0] * rank_total
            w_row = [0] * rank_total
            for idx, val in u_terms:
                u_row[idx] = val
            for idx, val in v_terms:
                v_row[idx] = val
            for idx, val in w_terms:
                w_row[idx] = val
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
    
    # i blocks: [0,1], [2,3]
    # j blocks: [0,1], [2,3], [4] (single)
    # k blocks: [0,1], [2,3], [4,5]
    
    for i0 in [0, 2]:
        for j0 in [0, 2, 4]:
            for k0 in [0, 2, 4]:
                if j0 == 4:
                    add_simple_2x1x2(i0, j0, k0)
                else:
                    add_strassen_2x2x2(i0, j0, k0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
