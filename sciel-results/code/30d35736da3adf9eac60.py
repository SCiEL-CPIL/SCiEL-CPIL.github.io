"""Matrix multiplication 6x6x6 using recursive Strassen block decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_terms(size):
    """Return list of (coeff_A_dict, coeff_B_dict, coeff_C_dict) for Strassen on size x size.
    Each coeff dict maps (r,c) -> coefficient (1, -1, or 2 for size>=4).
    """
    if size == 2:
        return [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ]
    half = size // 2
    base = _strassen_terms(half)
    terms = []
    for s in range(7):
        ca, cb, cc = base[s]
        new_ca = {}
        new_cb = {}
        new_cc = {}
        if s == 0:
            blocks_a = [(0,0),(1,1)]
            blocks_b = [(0,0),(1,1)]
            blocks_c = [(0,0),(1,1)]
            coeff = 1
        elif s == 1:
            blocks_a = [(1,0),(1,1)]
            blocks_b = [(0,0)]
            blocks_c = [(1,0),(1,1)]
            coeff = 1
        elif s == 2:
            blocks_a = [(0,0)]
            blocks_b = [(0,1),(1,1)]
            blocks_c = [(0,1),(1,1)]
            coeff = 1
        elif s == 3:
            blocks_a = [(1,1)]
            blocks_b = [(1,0),(0,0)]
            blocks_c = [(0,0),(1,0)]
            coeff = 1
        elif s == 4:
            blocks_a = [(0,0),(0,1)]
            blocks_b = [(1,1)]
            blocks_c = [(0,0),(0,1)]
            coeff = 1
        elif s == 5:
            blocks_a = [(1,0),(0,0)]
            blocks_b = [(0,0),(0,1)]
            blocks_c = [(1,1)]
            coeff = 1
        elif s == 6:
            blocks_a = [(0,1),(1,1)]
            blocks_b = [(1,0),(1,1)]
            blocks_c = [(0,0)]
            coeff = 1
        
        for (br, bc) in blocks_a:
            for (r, c) in ca:
                key = (br * half + r, bc * half + c)
                new_ca[key] = new_ca.get(key, 0) + coeff * ca[(r, c)]
        for (br, bc) in blocks_b:
            for (r, c) in cb:
                key = (br * half + r, bc * half + c)
                new_cb[key] = new_cb.get(key, 0) + coeff * cb[(r, c)]
        for (br, bc) in blocks_c:
            for (r, c) in cc:
                key = (br * half + r, bc * half + c)
                new_cc[key] = new_cc.get(key, 0) + coeff * cc[(r, c)]
        
        # Remove zero coefficients
        new_ca = {k: v for k, v in new_ca.items() if v != 0}
        new_cb = {k: v for k, v in new_cb.items() if v != 0}
        new_cc = {k: v for k, v in new_cc.items() if v != 0}
        terms.append((new_ca, new_cb, new_cc))
    return terms

def _make_vector(n, m, coeff_dict):
    """Create a list of length n*m from a coefficient dict mapping (r,c)->val."""
    vec = [0] * (n * m)
    for (r, c), val in coeff_dict.items():
        vec[r * m + c] = val
    return vec

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # Recursive Strassen: 6 = 2*3, so terms at level 2 (3x3 blocks), then level 3 (6x6)
    terms_3 = _strassen_terms(3)   # 7 terms for 3x3 using 2x2 Strassen on 3? No, 3 is odd.
    
    # Actually, let me use a direct approach for 6x6:
    # Treat 6x6 as 2x2 blocks of 3x3. Use Strassen 7 terms.
    # Each term multiplies two 3x3 matrices.
    # For 3x3, we can use a known rank-19 decomposition or schoolbook.
    # To keep it simple and correct, use schoolbook for the 3x3 base case.
    # Total rank = 7 * (3*3*3) = 7 * 27 = 189.
    
    n_half = n // 2  # 3
    m_half = m // 2  # 3
    p_half = p // 2  # 3
    
    # Base: schoolbook for n_half x m_half x p_half = 3x3x3
    base_rank = n_half * m_half * p_half  # 27
    base_U = []
    base_V = []
    base_W = []
    for i in range(n_half):
        for j in range(m_half):
            for k in range(p_half):
                u = [0] * base_rank
                v = [0] * base_rank
                w = [0] * base_rank
                col = i * m_half + j
                u[col] = 1
                v[j * p_half + k] = 1
                w[k * n_half + i] = 1
                base_U.append(u)
                base_V.append(v)
                base_W.append(w)
    
    # Strassen 7-term structure for 2x2 blocks
    # For each Strassen term s, we have:
    #   A_s = sum over block_a entries
    #   B_s = sum over block_b entries  
    #   C_ij += coeff * (A_s * B_s) for (i,j) in blocks_c
    #
    # We need to build the full 6x6 U, V, W.
    # 
    # U row index: (I, i) where I in {0,1} is block row, i in [0,3) is local row
    #   Actually U has shape (n*m, rank) = (36, rank)
    #   Row = global_row * ? ... Let me think about the tensor decomposition.
    #
    # The matrix multiplication tensor T[A,B,C] where A is n x m, B is m x p, C is p x n.
    # T_{i,j,k,l,m,n} ... standard: C[i,j] = sum_k A[i,k] B[k,j]
    # Tensor indices: (i, k) for A, (k, j) for B, (i, j) for C.
    # U has rows indexed by (i, k) -> i*m + k, shape (n*m, rank)
    # V has rows indexed by (k, j) -> k*p + j, shape (m*p, rank)
    # W has rows indexed by (i, j) -> i*n + j? No, W[k*n+i] per the convention.
    # Wait, the task says: W[k*n+i, r]. So W is indexed by (k, i) where k is the "p" index and i is the "n" index?
    # Let me re-read: "W[k*n+i,r]" with shape (p*n, rank). So the first index is k*n+i where k in [0,p), i in [0,n).
    # This means W's row corresponds to (output_row=i, output_col=k) in C[i,k]? 
    # Actually for matrix mult C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # Tensor: sum over j of A[i,j] * B[j,k] * C[i,k]
    # U rows: (i,j) -> i*m + j, shape (n*m, rank)
    # V rows: (j,k) -> j*p + k, shape (m*p, rank)
    # W rows: should be (i,k) -> i*p + k, shape (n*p, rank)
    # But task says W[k*n+i] with shape (p*n, rank). So W index = k*n + i, meaning row index = k*n + i.
    # This means W is indexed with k first, then i. So W[k*n + i] corresponds to C[i,k].
    # OK so W row index = k*n + i where k in [0,p), i in [0,n).
    
    # So for the full 6x6:
    # U rows: (i, j) -> i*6 + j, i in [0,6), j in [0,6)
    # V rows: (j, k) -> j*6 + k, j in [0,6), k in [0,6)
    # W rows: (k, i) -> k*6 + i, k in [0,6), i in [0,6)
    
    # Using 2x2 blocks of 3x3:
    # i = I*3 + i2, j = J*3 + j2, k = K*3 + k2 where I,J,K in {0,1}, i2,j2,k2 in [0,3)
    
    # For each Strassen term s (0..6):
    #   A_s = linear combination of A blocks
    #   B_s = linear combination of B blocks
    #   Contributes to C blocks
    #
    # For a given term s, we create base_rank many rank-1 terms (one per schoolbook basis for 3x3).
    # For each basis element (i2, j2, k2) in the 3x3 schoolbook:
    #   U row (i, j): U[i*6 + j, col] += coeff_A_s_block * (1 if (i2,j2) matches)
    #   V row (j, k): V[j*6 + k, col] += coeff_B_s_block * (1 if (j2,k2) matches)
    #   W row (k, i): W[k*6 + i, col] += coeff_C_s_block * (1 if (i2,k2) matches)
    #
    # But the Strassen coefficients apply at the block level. Let me be more careful.
    
    # Strassen term s:
    #   A_s = sum_{(Ia,Ja) in blocks_a(s)} c_a * A[Ia,Ja]  where A[Ia,Ja] is the 3x3 block
    #   B_s = sum_{(Jb,Kb) in blocks_b(s)} c_b * B[Jb,Kb]
    #   C[Ic,Kc] += c_c * (A_s * B_s) for (Ic,Kc) in blocks_c(s)
    #
    # A_s * B_s = sum_{i2,j2,k2} A_s[i2,j2] * B_s[j2,k2] * C_contrib[i2,k2]
    # where C_contrib is the 3x3 result block.
    
    # For the full tensor decomposition, for each term s and each schoolbook basis (i2,j2,k2):
    #   The U coefficient for row (i,j) = (Ia*3+i2, Ja*3+j2):
    #     = c_a(Ia,Ja) * 1  (if this is the matching (Ia,Ja,i2,j2) from A_s)
    #   Actually, A_s[i2,j2] = sum_{(Ia,Ja)} c_a(Ia,Ja) * A[Ia*3+i2, Ja*3+j2]
    #   So U[Ia*3+i2, Ja*3+j2, col] += c_a(Ia,Ja)
    #
    #   B_s[j2,k2] = sum_{(Jb,Kb)} c_b(Jb,Kb) * B[Jb*3+j2, Kb*3+k2]
    #   So V[Jb*3+j2, Kb*3+k2, col] += c_b(Jb,Kb)
    #
    #   C_contrib[i2,k2] gets added to C[Ic*3+i2, Kc*3+k2] with coefficient c_c(Ic,Kc)
    #   So W[Kc*3+k2, Ic*3+i2, col] += c_c(Ic,Kc)  [since W index = k*n+i = Kc*3+k2 * 6 + Ic*3+i2]
    
    # Let me define the Strassen blocks explicitly for 2x2.
    strassen_def = [
        # (blocks_a, blocks_b, blocks_c)
        # Each block is a list of (block_row, block_col) with coefficient +1
        # s=0: A00+A11, B00+B11 -> C00, C11
        ([(0,0),(1,1)], [(0,0),(1,1)], [(0,0),(1,1)]),
        # s=1: A10+A11, B00 -> C10, -C11
        ([(1,0),(1,1)], [(0,0)], [(1,0),(1,1)]),
        # s=2: A00, B01-B11 -> C01, C11
        ([(0,0)], [(0,1),(1,1)], [(0,1),(1,1)]),
        # s=3: A11, B10-B00 -> C00, C10
        ([(1,1)], [(1,0),(0,0)], [(0,0),(1,0)]),
        # s=4: A00+A01, B11 -> -C00, C01
        ([(0,0),(0,1)], [(1,1)], [(0,0),(0,1)]),
        # s=5: A10-A00, B00+B01 -> C11
        ([(1,0),(0,0)], [(0,0),(0,1)], [(1,1)]),
        # s=6: A01-A11, B10+B11 -> C00
        ([(0,1),(1,1)], [(1,0),(1,1)], [(0,0)]),
    ]
    
    # Coefficients for each block (all +1 or -1)
    def get_coeffs(blocks, sign_list):
        """Return dict (br,bc) -> coeff for blocks with given signs."""
        d = {}
        for (br, bc), sign in zip(blocks, sign_list):
            d[(br, bc)] = sign
        return d
    
    # For each Strassen term, define the signs for a, b, c blocks
    strassen_terms_full = [
        # s=0
        (get_coeffs([(0,0),(1,1)], [1,1]), get_coeffs([(0,0),(1,1)], [1,1]), get_coeffs([(0,0),(1,1)], [1,1])),
        # s=1
        (get_coeffs([(1,0),(1,1)], [1,1]), get_coeffs([(0,0)], [1]), get_coeffs([(1,0),(1,1)], [1,-1])),
        # s=2
        (get_coeffs([(0,0)], [1]), get_coeffs([(0,1),(1,1)], [1,-1]), get_coeffs([(0,1),(1,1)], [1,1])),
        # s=3
        (get_coeffs([(1,1)], [1]), get_coeffs([(1,0),(0,0)], [1,-1]), get_coeffs([(0,0),(1,0)], [1,1])),
        # s=4
        (get_coeffs([(0,0),(0,1)], [1,1]), get_coeffs([(1,1)], [1]), get_coeffs([(0,0),(0,1)], [-1,1])),
        # s=5
        (get_coeffs([(1,0),(0,0)], [1,-1]), get_coeffs([(0,0),(0,1)], [1,1]), get_coeffs([(1,1)], [1])),
        # s=6
        (get_coeffs([(0,1),(1,1)], [1,-1]), get_coeffs([(1,0),(1,1)], [1,1]), get_coeffs([(0,0)], [1])),
    ]
    
    total_rank = 7 * base_rank  # 7 * 27 = 189
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for s in range(7):
        ca, cb, cc = strassen_terms_full[s]
        for i2 in range(n_half):
            for j2 in range(m_half):
                for k2 in range(p_half):
                    # For this (i2, j2, k2) schoolbook basis element:
                    # U contributions
                    for (Ia, Ja), coeff_a in ca.items():
                        i = Ia * n_half + i2
                        j = Ja * m_half + j2
                        U[i * m + j][col] += coeff_a
                    # V contributions
                    for (Jb, Kb), coeff_b in cb.items():
                        j = Jb * m_half + j2
                        k = Kb * p_half + k2
                        V[j * p + k][col] += coeff_b
                    # W contributions
                    for (Ic, Kc), coeff_c in cc.items():
                        i = Ic * n_half + i2
                        k = Kc * p_half + k2
                        W[k * n + i][col] += coeff_c
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
