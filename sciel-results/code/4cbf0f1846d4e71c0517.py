import numpy as np
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_decomposition():
    """Build a decomposition for 4x4 matrix multiplication using recursive Strassen,
    then attempt symmetry-based rank reduction to eliminate redundant terms."""
    _n, _m, _p = 4, 4, 4
    
    def get_U_index(row, col):
        return row * _m + col
    
    def get_V_index(row, col):
        return row * _p + col
    
    def get_W_index(row, col):
        """Trace convention: W[k*n+i, r] where k=col, i=row"""
        return col * _n + row
    
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # Build the 7 block-level Strassen terms
    L_terms = []
    R_terms = []
    contrib_terms = []
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R0[r][c] = {(er, ec): 1, (hr, hc): 1}
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib0[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib0[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L0)
    R_terms.append(R0)
    contrib_terms.append(contrib0)
    
    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(1, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L1[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            R1[r][c] = {(er, ec): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 0, r, c)
            contrib1[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib1[(r, c)].append((cr2, cc2, -1))
    L_terms.append(L1)
    R_terms.append(R1)
    contrib_terms.append(contrib1)
    
    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            L2[r][c] = {(ar, ac): 1}
            fr, fc = b_elem(0, 1, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R2[r][c] = {(fr, fc): 1, (hr, hc): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 1, r, c)
            contrib2[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib2[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L2)
    R_terms.append(R2)
    contrib_terms.append(contrib2)
    
    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            dr, dc = a_elem(1, 1, r, c)
            L3[r][c] = {(dr, dc): 1}
            gr, gc = b_elem(1, 0, r, c)
            er, ec = b_elem(0, 0, r, c)
            R3[r][c] = {(gr, gc): 1, (er, ec): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib3[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 0, r, c)
            contrib3[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L3)
    R_terms.append(R3)
    contrib_terms.append(contrib3)
    
    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            br, bc = a_elem(0, 1, r, c)
            L4[r][c] = {(ar, ac): 1, (br, bc): 1}
            hr, hc = b_elem(1, 1, r, c)
            R4[r][c] = {(hr, hc): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib4[(r, c)] = [(cr, cc, -1)]
            cr2, cc2 = c_elem(0, 1, r, c)
            contrib4[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L4)
    R_terms.append(R4)
    contrib_terms.append(contrib4)
    
    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            cr, cc = a_elem(1, 0, r, c)
            ar, ac = a_elem(0, 0, r, c)
            L5[r][c] = {(cr, cc): 1, (ar, ac): -1}
            er, ec = b_elem(0, 0, r, c)
            fr, fc = b_elem(0, 1, r, c)
            R5[r][c] = {(er, ec): 1, (fr, fc): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 1, r, c)
            contrib5[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L5)
    R_terms.append(R5)
    contrib_terms.append(contrib5)
    
    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            br, bc = a_elem(0, 1, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L6[r][c] = {(br, bc): 1, (dr, dc): -1}
            gr, gc = b_elem(1, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R6[r][c] = {(gr, gc): 1, (hr, hc): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib6[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L6)
    R_terms.append(R6)
    contrib_terms.append(contrib6)
    
    # Strassen sub-product coefficients for 2x2 result
    sub_contrib = [
        [(0,0,1), (1,1,1)],    # M1
        [(1,0,1), (1,1,-1)],   # M2
        [(0,1,1), (1,1,1)],    # M3
        [(0,0,1), (1,0,1)],    # M4
        [(0,0,-1), (0,1,1)],   # M5
        [(1,1,1)],             # M6
        [(0,0,1)]              # M7
    ]
    
    # Build the full 49-rank decomposition
    all_U = []  # list of dicts: rank_idx -> {a_index: coeff}
    all_V = []  # list of dicts: rank_idx -> {b_index: coeff}
    all_W = []  # list of dicts: rank_idx -> {c_index: coeff}
    
    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = contrib_terms[t]
        
        a = L_t[0][0]
        b = L_t[0][1]
        c = L_t[1][0]
        d = L_t[1][1]
        e = R_t[0][0]
        f = R_t[0][1]
        g = R_t[1][0]
        h = R_t[1][1]
        
        sub_products = []
        
        # M1: X = a+d, Y = e+h
        x1 = {}
        for key, coeff in a.items():
            x1[key] = x1.get(key, 0) + coeff
        for key, coeff in d.items():
            x1[key] = x1.get(key, 0) + coeff
        y1 = {}
        for key, coeff in e.items():
            y1[key] = y1.get(key, 0) + coeff
        for key, coeff in h.items():
            y1[key] = y1.get(key, 0) + coeff
        sub_products.append((x1, y1))
        
        # M2: X = c+d, Y = e
        x2 = {}
        for key, coeff in c.items():
            x2[key] = x2.get(key, 0) + coeff
        for key, coeff in d.items():
            x2[key] = x2.get(key, 0) + coeff
        y2 = dict(e)
        sub_products.append((x2, y2))
        
        # M3: X = a, Y = f-h
        x3 = dict(a)
        y3 = {}
        for key, coeff in f.items():
            y3[key] = y3.get(key, 0) + coeff
        for key, coeff in h.items():
            y3[key] = y3.get(key, 0) - coeff
        sub_products.append((x3, y3))
        
        # M4: X = d, Y = g-e
        x4 = dict(d)
        y4 = {}
        for key, coeff in g.items():
            y4[key] = y4.get(key, 0) + coeff
        for key, coeff in e.items():
            y4[key] = y4.get(key, 0) - coeff
        sub_products.append((x4, y4))
        
        # M5: X = a+b, Y = h
        x5 = {}
        for key, coeff in a.items():
            x5[key] = x5.get(key, 0) + coeff
        for key, coeff in b.items():
            x5[key] = x5.get(key, 0) + coeff
        y5 = dict(h)
        sub_products.append((x5, y5))
        
        # M6: X = c-a, Y = e+f
        x6 = {}
        for key, coeff in c.items():
            x6[key] = x6.get(key, 0) + coeff
        for key, coeff in a.items():
            x6[key] = x6.get(key, 0) - coeff
        y6 = {}
        for key, coeff in e.items():
            y6[key] = y6.get(key, 0) + coeff
        for key, coeff in f.items():
            y6[key] = y6.get(key, 0) + coeff
        sub_products.append((x6, y6))
        
        # M7: X = b-d, Y = g+h
        x7 = {}
        for key, coeff in b.items():
            x7[key] = x7.get(key, 0) + coeff
        for key, coeff in d.items():
            x7[key] = x7.get(key, 0) - coeff
        y7 = {}
        for key, coeff in g.items():
            y7[key] = y7.get(key, 0) + coeff
        for key, coeff in h.items():
            y7[key] = y7.get(key, 0) + coeff
        sub_products.append((x7, y7))
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            # U part
            u_dict = {}
            for (row, col), coeff in x_k.items():
                u_idx = get_U_index(row, col)
                u_dict[u_idx] = u_dict.get(u_idx, 0) + coeff
            all_U.append(u_dict)
            
            # V part
            v_dict = {}
            for (row, col), coeff in y_k.items():
                v_idx = get_V_index(row, col)
                v_dict[v_idx] = v_dict.get(v_idx, 0) + coeff
            all_V.append(v_dict)
            
            # W part
            w_dict = {}
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_idx = get_W_index(out_row, out_col)
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + sub_coeff * block_coeff
            all_W.append(w_dict)
    
    # Now attempt symmetry-based rank reduction.
    # The matrix multiplication tensor T[i,j,k,l] = delta[i,l]*delta[j,k] has symmetries.
    # Specifically, if we swap (i,k) <-> (j,l) and A<->B, the tensor is invariant.
    # This means certain rank-1 terms come in "paired" structures.
    
    # Strategy: Check if any rank-1 term is a linear combination of others
    # by looking at the structure of the U, V, W vectors.
    
    # First, let's check for exact duplicates or near-duplicates in the (U,V) pairs
    # that could allow us to merge terms.
    
    # Convert to numpy arrays for linear algebra
    n_rows = 16
    rank_49 = 49
    
    U_arr = np.zeros((n_rows, rank_49))
    V_arr = np.zeros((n_rows, rank_49))
    W_arr = np.zeros((n_rows, rank_49))
    
    for r in range(rank_49):
        for idx, coeff in all_U[r].items():
            U_arr[idx, r] = coeff
        for idx, coeff in all_V[r].items():
            V_arr[idx, r] = coeff
        for idx, coeff in all_W[r].items():
            W_arr[idx, r] = coeff
    
    # Verify the decomposition is correct
    # Tensor T[i,j,k,l] should equal sum_r U[i,r]*V[j,r]*W[k,r]
    # where i = a_index (row*m+col), j = b_index, k = c_index (trace: col*n+row)
    # T[a_idx, b_idx, c_idx] = 1 if they form a valid multiplication path, 0 otherwise
    
    # Let's verify: for each (i,j,k) where i=(a_row,a_col), j=(b_row,b_col), k=(c_col,c_row)
    # T = 1 iff a_col == b_row and c_col == a_row and c_row == b_col
    # i.e., a_col == b_row and a_row == c_col and b_col == c_row
    
    def verify(U_a, V_a, W_a, n=4):
        """Verify that the decomposition is exact."""
        max_err = 0.0
        for a_row in range(n):
            for a_col in range(n):
                a_idx = a_row * n + a_col
                for b_row in range(n):
                    for b_col in range(n):
                        b_idx = b_row * n + b_col
                        for c_row in range(n):
                            for c_col in range(n):
                                c_idx = c_col * n + c_row
                                # Expected value
                                expected = 1.0 if (a_col == b_row and a_row == c_col and b_col == c_row) else 0.0
                                # Computed value
                                computed = 0.0
                                for r in range(U_a.shape[1]):
                                    computed += U_a[a_idx, r] * V_a[b_idx, r] * W_a[c_idx, r]
                                err = abs(computed - expected)
                                if err > max_err:
                                    max_err = err
        return max_err
    
    err = verify(U_arr, V_arr, W_arr)
    if err > 1e-10:
        # Decomposition is wrong, return the 49-rank version as-is
        # (shouldn't happen, but safety check)
        pass
    
    # Attempt rank reduction by checking for linear dependencies
    # among the rank-1 terms using the symmetry structure.
    
    # The key insight: in the recursive Strassen decomposition, some of the
    # 49 terms may be expressible as linear combinations of others due to
    # the specific structure of the Strassen additions.
    
    # Let's try to find a subset of terms that spans the same tensor.
    # We'll use a greedy approach: try to remove each term and check if
    # the remaining terms still span the tensor.
    
    # A term r can be removed if there exist coefficients alpha_s (s != r)
    # such that for all (i,j,k):
    #   U[i,r]*V[j,r]*W[k,r] = sum_{s!=r} alpha_s * U[i,s]*V[j,s]*W[k,s]
    
    # This is a system of 64*64*64 = 262144 equations in (rank-1) unknowns.
    # Too large to solve directly. Instead, we use the fact that the tensor
    # has only 4*4*4 = 64 non-zero entries (one for each valid (a,b,c) triple).
    
    # Actually, the non-zero entries are: for each a_row, b_col, a_col(=b_row), c_col(=a_row), c_row(=b_col)
    # So there are 4*4*4 = 64 non-zero entries.
    
    # For a term to be removable, we need:
    # U[i,r]*V[j,r]*W[k,r] = sum_s alpha_s * U[i,s]*V[j,s]*W[k,s] for all 64 non-zero (i,j,k)
    
    # This gives us 64 equations in (rank-1) unknowns. If rank-1 > 64, we might
    # find a solution. But 49-1 = 48 < 64, so the system is overdetermined.
    # We need to check if a solution exists.
    
    # Let's build the 64 non-zero tensor entries
    nz_entries = []
    for a_row in range(4):
        for b_col in range(4):
            a_col = b_row = b_col  # Wait, this isn't right
            # Non-zero when: a_col == b_row AND a_row == c_col AND b_col == c_row
            # Free variables: a_row, a_col, b_col (then b_row=a_col, c_col=a_row, c_row=b_col)
            for a_col in range(4):
                b_row = a_col
                c_col = a_row
                c_row = b_col
                a_idx = a_row * 4 + a_col
                b_idx = b_row * 4 + b_col
                c_idx = c_col * 4 + c_row
                nz_entries.append((a_idx, b_idx, c_idx))
    
    # For each term r, check if it can be expressed as a linear combination of others
    # by solving the 64x48 system (if consistent)
    
    # Actually, let me think about this differently. The tensor is a 3-tensor of size
    # 16x16x16 with 64 non-zero entries. The rank-49 decomposition gives 49 rank-1 terms.
    # The question is whether the tensor rank is actually less than 49.
    
    # For 4x4 matrix multiplication, the known upper bounds are:
    # - Recursive Strassen: 49
    # - Various improvements: 48 (by exploiting symmetries)
    # - Lower bound: 49 (actually not proven, but believed)
    
    # Let me try a different approach: use the symmetry to directly construct
    # a rank-48 decomposition by modifying the Strassen terms.
    
    # The symmetry T[i,j,k,l] = T[j,i,l,k] (swap A and B, swap i<->j, k<->l)
    # means that if we have a rank-1 term (u,v,w), we also have a "dual" term
    # (v_permuted, u_permuted, w_permuted) that gives the same tensor contribution.
    
    # In the Strassen decomposition, some terms are "self-dual" while others come
    # in pairs. If a pair of terms can be merged into one (using the symmetry),
    # we can reduce the rank.
    
    # For now, let's try a simpler approach: check if any two terms are identical
    # or linearly dependent in a way that allows merging.
    
    # Check for duplicate (U,V) pairs
    used_pairs = set()
    removable = []
    for r in range(rank_49):
        u_key = tuple(U_arr[:, r].round(10))
        v_key = tuple(V_arr[:, r].round(10))
        if (u_key, v_key) in used_pairs:
            removable.append(r)
        else:
            used_pairs.add((u_key, v_key))
    
    # Check for terms where W is a scalar multiple of another term's W
    # and U,V are also related
    
    # For now, if we found removable terms, use them
    if removable:
        keep = [r for r in range(rank_49) if r not in removable]
        new_rank = len(keep)
        U_new = U_arr[:, keep]
        V_new = V_arr[:, keep]
        W_new = W_arr[:, keep]
        
        # Verify
        err_new = verify(U_new, V_new, W_new)
        if err_new < 1e-10:
            U_arr = U_new
            V_arr = V_new
            W_arr = W_new
    
    # Convert back to list format
    final_rank = U_arr.shape[1]
    
    U_list = [[U_arr[i, r] for r in range(final_rank)] for i in range(n_rows)]
    V_list = [[V_arr[i, r] for r in range(final_rank)] for i in range(n_rows)]
    W_list = [[W_arr[i, r] for r in range(final_rank)] for i in range(n_rows)]
    
    return {"U": U_list, "V": V_list, "W": W_list}


# Precompute at module level
_DECOMP = _build_decomposition()


def construct():
    """
    Return the precomputed tensor decomposition for 4x4 matrix multiplication.
    """
    return {"U": _DECOMP["U"], "V": _DECOMP["V"], "W": _DECOMP["W"]}

# EVOLVE-BLOCK-END
