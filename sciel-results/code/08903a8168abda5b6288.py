"""Program for matrix multiplication 3x5x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 120

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+h)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C11, 1=C12, 2=C21, 3=C22
    
    # M1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 +=, C22 +=
    
    # M2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21 +=
    
    # M3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1 # C12 +=, C22 -=
    
    # M4: d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1 # C11 -=, C21 +=
    
    # M5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1 # C11 +=, C12 +=
    
    # M6: (c-d)(f+g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1 # C22 +=
    
    # M7: (b-d)(e+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1 # C11 +=, C22 +=
    
    return U, V, W

def _outer_product_2x1x2():
    """Returns U, V, W for 2x1x2 outer product (rank 1)."""
    # A is 2x1 [a, c], B is 1x2 [e, f]
    # C is 2x2 [[ae, af], [ce, cf]]
    U = np.zeros((2, 1))
    V = np.zeros((2, 1))
    W = np.zeros((4, 1))
    
    U[0, 0] = 1 # a
    U[1, 0] = 1 # c
    
    V[0, 0] = 1 # e
    V[1, 0] = 1 # f
    
    W[0, 0] = 1 # C11 = a*e
    W[1, 0] = 1 # C12 = a*f
    W[2, 0] = 1 # C21 = c*e
    W[3, 0] = 1 # C22 = c*f
    
    return U, V, W

def _rank2_1x2x2():
    """Returns U, V, W for 1x2x2 (rank 2)."""
    # A is 1x2 [a, b], B is 2x2 [[e, f], [g, h]]
    # C is 1x2 [ae+bg, af+bh]
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    # Term 1: a * [e, f]
    U[0, 0] = 1
    V[0, 0] = 1 # e
    V[1, 0] = 1 # f
    W[0, 0] = 1 # C1
    W[1, 0] = 1 # C2
    
    # Term 2: b * [g, h]
    U[1, 1] = 1
    V[2, 1] = 1 # g
    V[3, 1] = 1 # h
    W[0, 1] = 1 # C1
    W[1, 1] = 1 # C2
    
    return U, V, W

def _rank1_1x1x2():
    """Returns U, V, W for 1x1x2 (rank 1)."""
    # A is 1x1 [a], B is 1x2 [e, f]
    # C is 1x2 [ae, af]
    U = np.zeros((1, 1))
    V = np.zeros((2, 1))
    W = np.zeros((2, 1))
    
    U[0, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    W[0, 0] = 1
    W[1, 0] = 1
    
    return U, V, W

def _combine_decompositions(n, m, p, splits_i, splits_j, splits_k, get_decomp):
    """
    Combines decompositions of sub-blocks.
    splits_i, splits_j, splits_k are lists of lists of indices.
    get_decomp(n_sub, m_sub, p_sub) returns (U_sub, V_sub, W_sub, rank_sub)
    """
    rank_total = 0
    # Pre-calculate total rank
    for i_start, i_end in splits_i:
        for j_start, j_end in splits_j:
            for k_start, k_end in splits_k:
                n_sub = i_end - i_start
                m_sub = j_end - j_start
                p_sub = k_end - k_start
                if n_sub == 0 or m_sub == 0 or p_sub == 0: continue
                _, _, _, r = get_decomp(n_sub, m_sub, p_sub)
                rank_total += r
                
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col_offset = 0
    for i_start, i_end in splits_i:
        for j_start, j_end in splits_j:
            for k_start, k_end in splits_k:
                n_sub = i_end - i_start
                m_sub = j_end - j_start
                p_sub = k_end - k_start
                if n_sub == 0 or m_sub == 0 or p_sub == 0: continue
                
                U_sub, V_sub, W_sub, r = get_decomp(n_sub, m_sub, p_sub)
                
                # Map indices
                # U is (n*m, rank). Index is i*m + j.
                # Sub U is (n_sub*m_sub, r). Index is i_sub*m_sub + j_sub.
                # Global i = i_start + i_sub
                # Global j = j_start + j_sub
                # Global index = (i_start + i_sub)*m + (j_start + j_sub)
                
                # V is (m*p, rank). Index is j*p + k.
                # Sub V is (m_sub*p_sub, r). Index is j_sub*p_sub + k_sub.
                # Global j = j_start + j_sub
                # Global k = k_start + k_sub
                # Global index = (j_start + j_sub)*p + (k_start + k_sub)
                
                # W is (p*n, rank). Index is k*n + i.
                # Sub W is (p_sub*n_sub, r). Index is k_sub*n_sub + i_sub.
                # Global k = k_start + k_sub
                # Global i = i_start + i_sub
                # Global index = (k_start + k_sub)*n + (i_start + i_sub)
                
                for c in range(r):
                    for i_sub in range(n_sub):
                        for j_sub in range(m_sub):
                            u_val = U_sub[i_sub*m_sub + j_sub, c]
                            if u_val != 0:
                                global_idx_u = (i_start + i_sub)*m + (j_start + j_sub)
                                U[global_idx_u, col_offset + c] += u_val
                                
                    for j_sub in range(m_sub):
                        for k_sub in range(p_sub):
                            v_val = V_sub[j_sub*p_sub + k_sub, c]
                            if v_val != 0:
                                global_idx_v = (j_start + j_sub)*p + (k_start + k_sub)
                                V[global_idx_v, col_offset + c] += v_val
                                
                    for k_sub in range(p_sub):
                        for i_sub in range(n_sub):
                            w_val = W_sub[k_sub*n_sub + i_sub, c]
                            if w_val != 0:
                                global_idx_w = (k_start + k_sub)*n + (i_start + i_sub)
                                W[global_idx_w, col_offset + c] += w_val
                                
                col_offset += r
                
    return U, V, W, rank_total

def _get_decomp(n, m, p):
    """Returns decomposition for n x m x p."""
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    elif n == 2 and m == 1 and p == 2:
        U, V, W = _outer_product_2x1x2()
        return U, V, W, 1
    elif n == 1 and m == 2 and p == 2:
        U, V, W = _rank2_1x2x2()
        return U, V, W, 2
    elif n == 1 and m == 1 and p == 2:
        U, V, W = _rank1_1x1x2()
        return U, V, W, 1
    else:
        # Fallback to schoolbook for unexpected sizes
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Splits for 3x5x6
    # i: 0-2 -> [0,2), [2,3)
    splits_i = [(0, 2), (2, 3)]
    # j: 0-4 -> [0,2), [2,4), [4,5)
    splits_j = [(0, 2), (2, 4), (4, 5)]
    # k: 0-5 -> [0,2), [2,4), [4,6)
    splits_k = [(0, 2), (2, 4), (4, 6)]
    
    U, V, W, rank = _combine_decompositions(n, m, p, splits_i, splits_j, splits_k, _get_decomp)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
