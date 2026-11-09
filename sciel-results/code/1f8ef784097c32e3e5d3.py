"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return list of (u_vec, v_vec, w_vec) for Strassen 2x2x2.
    n=m=p=2. U index: i*m+j = i*2+j. V index: j*p+k = j*2+k. W index: k*n+i = k*2+i.
    A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [k*n+i: (k,i)]
    """
    # Each term: (list of (u_idx, coeff), list of (v_idx, coeff), list of (w_idx, coeff))
    terms = []
    # M1 = (a+d)(e+h) -> C11 += M1, C22 += M1
    terms.append(([ (0,1),(3,1) ], [ (0,1),(3,1) ], [ (0,1),(3,1) ]))  # W: (k=0,i=0)->0, (k=1,i=1)->3
    # M2 = (c+d)e -> C21 += M2, C22 -= M2
    terms.append(([ (2,1),(3,1) ], [ (0,1) ], [ (2,1), (3,-1) ]))
    # M3 = a(f-h) -> C12 += M3, C22 += M3
    terms.append(([ (0,1) ], [ (1,1),(3,-1) ], [ (1,1),(3,1) ]))
    # M4 = d(g-e) -> C11 += M4, C21 += M4
    terms.append(([ (3,1) ], [ (2,1),(0,-1) ], [ (0,1),(2,1) ]))
    # M5 = (a+b)h -> C11 -= M5, C12 += M5
    terms.append(([ (0,1),(1,1) ], [ (3,1) ], [ (0,-1),(1,1) ]))
    # M6 = (c-a)(e+f) -> C22 += M6
    terms.append(([ (2,1),(0,-1) ], [ (0,1),(1,1) ], [ (3,1) ]))
    # M7 = (b-d)(g+h) -> C11 += M7
    terms.append(([ (1,1),(3,-1) ], [ (2,1),(3,1) ], [ (0,1) ]))
    return terms

def _build_decomposition(terms, n, m, p):
    """Build U, V, W matrices from list of terms."""
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = len(terms)
    U = [[0.0] * rank for _ in range(u_dim)]
    V = [[0.0] * rank for _ in range(v_dim)]
    W = [[0.0] * rank for _ in range(w_dim)]
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for idx, coeff in u_list:
            U[idx][r] += coeff
        for idx, coeff in v_list:
            V[idx][r] += coeff
        for idx, coeff in w_list:
            W[idx][r] += coeff
    return U, V, W

def _strassen_4x4x4():
    """Return U, V, W for 4x4 x 4x4 -> 4x4, rank 49."""
    n = m = p = 4
    base = _strassen_2x2_terms()
    all_terms = []
    for outer in base:
        for inner in base:
            # Combine: U is 16-dim, index = (outer_block_row * 2 + inner_row) * 4 + (outer_block_col * 2 + inner_col)
            # Actually: for 4x4, split into 2x2 blocks of 2x2 each.
            # U index for 4x4: i*4+j where i in [0,3], j in [0,3]
            # Outer block: i1 (0 or 1), inner: i0 (0 or 1) -> i = i1*2 + i0
            # Similarly for j, k
            # Outer U: index i1*2+j1 (2x2), inner U: index i0*2+j0
            # Combined: (i1*2+i0)*4 + (j1*2+j0)
            u_comb = []
            for (oi, oc) in outer[0]:
                for (ii, ic) in inner[0]:
                    combined_idx = (oi*2 + ii) * 4 + (oc*2 + ic)
                    coeff = oc * ic
                    u_comb.append((combined_idx, coeff))
            v_comb = []
            for (oj, ok) in outer[1]:
                for (ij, ik) in inner[1]:
                    combined_idx = (oj*2 + ij) * 4 + (ok*2 + ik)
                    coeff = ok * ik
                    v_comb.append((combined_idx, coeff))
            w_comb = []
            for (ok, oi) in outer[2]:
                for (ik, ii) in inner[2]:
                    combined_idx = (ok*2 + ik) * 4 + (oi*2 + ii)
                    coeff = ok * oi
                    w_comb.append((combined_idx, coeff))
            all_terms.append((u_comb, v_comb, w_comb))
    # Simplify: merge duplicate indices
    simplified = []
    for (u_list, v_list, w_list) in all_terms:
        u_dict = {}
        for idx, c in u_list:
            u_dict[idx] = u_dict.get(idx, 0) + c
        v_dict = {}
        for idx, c in v_list:
            v_dict[idx] = v_dict.get(idx, 0) + c
        w_dict = {}
        for idx, c in w_list:
            w_dict[idx] = w_dict.get(idx, 0) + c
        u_list2 = [(i, c) for i, c in u_dict.items() if c != 0]
        v_list2 = [(i, c) for i, c in v_dict.items() if c != 0]
        w_list2 = [(i, c) for i, c in w_dict.items() if c != 0]
        simplified.append((u_list2, v_list2, w_list2))
    return _build_decomposition(simplified, 4, 4, 4)

def construct():
    """Construct rank-minimal decomposition for 4x4 * 4x9 -> 4x9."""
    import numpy as np
    import time
    t0 = time.time()
    DEADLINE = 100  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Split columns of B (4x9) into 4+4+1
    # C = A*B = [A*B1 | A*B2 | A*B3] where B1,B2 are 4x4, B3 is 4x1
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: A(4x4) * B1(4x4) -> C1(4x4), columns 0-3 of output
    U1, V1, W1 = _strassen_4x4x4()
    # V1: index j*p1+k where p1=4, so V1[j*4+k, r]. But in final V: index j*9+k
    # W1: index k*n+i where k in [0,3] (for B1's columns), i in [0,3]
    # In final W: k is global column index, so for B1 (cols 0-3), k stays 0-3
    U_parts.append(np.array(U1))
    
    V1_final = np.zeros((4*9, len(V1)), dtype=float)
    for j in range(4):
        for k in range(4):
            V1_final[j*9 + k] = V1[j*4 + k]
    V_parts.append(V1_final)
    
    W_parts.append(np.array(W1))  # k in [0,3] for B1
    
    # Part 2: A(4x4) * B2(4x4) -> C2(4x4), columns 4-7 of output
    U2, V2, W2 = _strassen_4x4x4()
    U_parts.append(np.array(U2))
    
    V2_final = np.zeros((4*9, len(V2)), dtype=float)
    for j in range(4):
        for k in range(4):
            V2_final[j*9 + (4+k)] = V2[j*4 + k]
    V_parts.append(V2_final)
    
    W2_final = np.zeros((9*4, len(W2)), dtype=float)
    for k in range(4):
        for i in range(4):
            W2_final[(4+k)*4 + i] = W2[k*4 + i]
    W_parts.append(W2_final)
    
    # Part 3: A(4x4) * B3(4x1) -> C3(4x1), column 8 of output
    # Rank 16: each (i,j) pair gives one term
    rank3 = 16
    U3 = np.zeros((16, rank3))
    V3 = np.zeros((36, rank3))
    W3 = np.zeros((36, rank3))
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U3[i * 4 + j, r] = 1
            V3[j * 9 + 8, r] = 1
            W3[8 * 4 + i, r] = 1
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Combine all parts
    U_all = np.hstack(U_parts)
    V_all = np.hstack(V_parts)
    W_all = np.hstack(W_parts)
    
    # Verify correctness
    def verify():
        A_rand = np.random.randn(4, 4)
        B_rand = np.random.randn(4, 9)
        C_expected = A_rand @ B_rand
        C_computed = np.zeros((4, 9))
        for r in range(U_all.shape[1]):
            a_vals = A_rand.reshape(-1) @ U_all[:, r]  # no, this isn't right
            # Actually: C[i,k] = sum_r U[i*4+j... no
            # The tensor: C = sum_r u_r ⊗ v_r ⊗ w_r
            # C[i,k] = sum_r (sum_j U[i*4+j,r]*A[i,j]) * (sum_l V[j*9+k,r]*B[j,l])... 
            # No, that's not right either. Let me think.
            # U[i*m+j, r] where m=4, so U[i*4+j, r]
            # V[j*p+k, r] where p=9, so V[j*9+k, r]
            # W[k*n+i, r] where n=4, so W[k*4+i, r]
            # C[i,k] = sum_r U_flat * A * V_flat * W... 
            # Actually the standard tensor: 
            # C[i,k] = sum_{j,l} A[i,j] B[j,l] * delta_{k,l}  (since B is 4x9, j in [0,3], k in [0,8])
            # Wait B is 4x9 so B[j,l] where j in [0,3], l in [0,8]. C[i,l] = sum_j A[i,j]*B[j,l].
            # The rank-1 tensor: U[i*4+j, r] * V[j*9+l, r] * W[l*4+i, r]
            # sum_r U[i*4+j,r] V[j*9+l,r] W[l*4+i,r] = delta_{i,i}delta_{j,j}delta_{l,l} = 1
            # So C[i,l] = sum_j A[i,j] B[j,l] sum_r U[i*4+j,r] V[j*9+l,r] W[l*4+i,r]
            # We need sum_r U[i*4+j,r] V[j*9+l,r] W[l*4+i,r] = 1 for all i,j,l
            pass
    
    # Simple verification via random test
    for trial in range(3):
        A_rand = np.random.randn(4, 4)
        B_rand = np.random.randn(4, 9)
        C_expected = A_rand @ B_rand
        C_computed = np.zeros((4, 9))
        for r in range(U_all.shape[1]):
            # For each r, compute contribution
            # C_contrib[i,l] = sum_j U[i*4+j,r] * A[i,j]... no
            # The bilinear form: for fixed r, the term is:
            # sum_{j} U[i*4+j, r] * A[i,j] * V[j*9+l, r] * B[j,l] * W[l*4+i, r]
            # But this isn't separable in the simple way. Let me just compute directly.
            for i in range(4):
                for l in range(9):
                    for j in range(4):
                        for lj in range(9):  # this is O(n*m*p*rank) which is too slow
                            pass
        # Faster: C[i,l] = sum_r (A_row_i @ U_col_r_i) * (something)
        # Let's use: C[i,l] = sum_r (sum_j A[i,j] U[i*4+j,r]) * (sum... 
        # Actually: define for each r:
        #   a_r[i,j] = U[i*4+j, r]  (4x4 matrix)
        #   b_r[j,l] = V[j*9+l, r]  (4x9 matrix)  
        #   w_r[l,i] = W[l*4+i, r]   (9x4 matrix)
        # Then C[i,l] = sum_r (A @ a_r)[i,j]... no
        # C[i,l] = sum_{j} A[i,j] B[j,l] * sum_r U[i*4+j,r] V[j*9+l,r] W[l*4+i,r]
        # = sum_{j,l'} A[i,j] B[j,l'] sum_r U[i*4+j,r] V[j*9+l',r] W[l'*4+i,r] * delta(l',l)
        # The key is: for fixed (i,j,l), T[i,j,l] = sum_r U[i*4+j,r] V[j*9+l,r] W[l*4+i,r] should equal 1.
        # Let's just verify T.
        T = np.zeros((4,4,9))
        for r in range(U_all.shape[1]):
            u_col = U_all[:, r]  # 16 dim
            v_col = V_all[:, r]  # 36 dim
            w_col = W_all[:, r]  # 36 dim
            U_mat = u_col.reshape(4, 4)  # [i,j]
            V_mat = v_col.reshape(4, 9)  # [j,l]
            W_mat = w_col.reshape(9, 4)  # [l,i]
            T += np.einsum('ij,jl,li->ijl', U_mat, V_mat, W_mat)
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
    return True
    
    if not verify():
        # Fallback to schoolbook
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to integer/half-integer if possible
    def to_clean(mat):
        mat = np.round(mat * 2) / 2
        return mat.tolist()
    
    return {"U": to_clean(U_all), "V": to_clean(V_all), "W": to_clean(W_all)}
# EVOLVE-BLOCK-END
