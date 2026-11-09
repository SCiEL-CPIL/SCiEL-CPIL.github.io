import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0

    N, M, P = 4, 4, 5

    # --- Verification helper ---
    def check_validity(Umat, Vmat, Wmat):
        r = Umat.shape[1]
        if Vmat.shape != (M * P, r) or Wmat.shape != (P * N, r):
            return False
        U3 = Umat.reshape(N, M, r)
        V3 = Vmat.reshape(M, P, r)
        W3 = Wmat.reshape(P, N, r)
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: The problem statement says W[k*n+i, r].
        # My reshape W3 is (P, N, r). So W3[k, i, r] is correct.
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return bool(np.allclose(T, 1.0, atol=1e-10))

    # --- Schoolbook fallback (rank 80) ---
    def schoolbook():
        R = N * M * P
        U = np.zeros((N * M, R))
        V = np.zeros((M * P, R))
        W = np.zeros((P * N, R))
        idx = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, idx] = 1.0
                    V[j * P + k, idx] = 1.0
                    W[k * N + i, idx] = 1.0
                    idx += 1
        return U, V, W

    # --- Rank-65 construction: 49 (Strassen 4x4x4) + 16 (tail) ---
    # This logic is derived from the Inspiration program which is known to work.
    
    U = np.zeros((N * M, 65), dtype=float)
    V = np.zeros((M * P, 65), dtype=float)
    W = np.zeros((P * N, 65), dtype=float)
    
    # Strassen 2x2: 7 multiplications
    # C = A * B where A, B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2_mults():
        """
        Returns list of 7 tuples: (u_dict, v_dict)
        where u_dict maps (r,c) -> coeff for the 2x2 A matrix
        and v_dict maps (r,c) -> coeff for the 2x2 B matrix
        """
        mults = []
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        mults.append((u1, v1))
        
        # M2
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        mults.append((u2, v2))
        
        # M3
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        mults.append((u3, v3))
        
        # M4
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        mults.append((u4, v4))
        
        # M5
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        mults.append((u5, v5))
        
        # M6
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        mults.append((u6, v6))
        
        # M7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        mults.append((u7, v7))
        
        return mults

    def expand_2x2_to_scalars(u_matrix, v_matrix):
        """
        u_matrix: 2x2 array where each entry is a dict {(i,j): coeff} of 4x4 A entries
        v_matrix: 2x2 array where each entry is a dict {(i,j): coeff} of 4x4 B entries
        Returns list of 7 (u_scalar, v_scalar) where each is a dict {(i,j): coeff}
        """
        strassen_mults = strassen_2x2_mults()
        results = []
        for u_dict, v_dict in strassen_mults:
            u_scalar = {}
            v_scalar = {}
            for (ur, uc), coeff_u in u_dict.items():
                for (ai, aj), coeff_ai in u_matrix[ur][uc].items():
                    key = (ai, aj)
                    u_scalar[key] = u_scalar.get(key, 0) + coeff_u * coeff_ai
            for (vr, vc), coeff_v in v_dict.items():
                for (bi, bj), coeff_bi in v_matrix[vr][vc].items():
                    key = (bi, bj)
                    v_scalar[key] = v_scalar.get(key, 0) + coeff_v * coeff_bi
            # Remove zeros
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
            results.append((u_scalar, v_scalar))
        return results

    # Define the 7 outer terms as 2x2 matrices of linear combos
    # A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) in block coordinates
    # Block (br, bc) corresponds to rows 2br..2br+1, cols 2bc..2bc+1
    
    def make_block_matrix(block_defs):
        """
        block_defs: list of 4 dicts, each {(i,j): coeff} for the 2x2 block
        Returns 2x2 array of dicts
        """
        return [[block_defs[r*2+c] for c in range(2)] for r in range(2)]

    # Outer term 1: U1 = A00 + A11, V1 = B00 + B11
    U1_blocks = []
    for r in range(2):
        for c in range(2):
            U1_blocks.append({(r,c): 1, (r+2,c+2): 1})
    V1_blocks = []
    for r in range(2):
        for c in range(2):
            V1_blocks.append({(r,c): 1, (r+2,c+2): 1})
    
    # Outer term 2: U2 = A10 + A11, V2 = B00
    U2_blocks = []
    for r in range(2):
        for c in range(2):
            U2_blocks.append({(r+2,c): 1, (r+2,c+2): 1})
    V2_blocks = []
    for r in range(2):
        for c in range(2):
            V2_blocks.append({(r,c): 1})
            
    # Outer term 3: U3 = A00, V3 = B01 - B11
    U3_blocks = []
    for r in range(2):
        for c in range(2):
            U3_blocks.append({(r,c): 1})
    V3_blocks = []
    for r in range(2):
        for c in range(2):
            V3_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
            
    # Outer term 4: U4 = A11, V4 = B10 - B00
    U4_blocks = []
    for r in range(2):
        for c in range(2):
            U4_blocks.append({(r+2,c+2): 1})
    V4_blocks = []
    for r in range(2):
        for c in range(2):
            V4_blocks.append({(r+2,c): 1, (r,c): -1})
            
    # Outer term 5: U5 = A00 + A01, V5 = B11
    U5_blocks = []
    for r in range(2):
        for c in range(2):
            U5_blocks.append({(r,c): 1, (r,c+2): 1})
    V5_blocks = []
    for r in range(2):
        for c in range(2):
            V5_blocks.append({(r+2,c+2): 1})
            
    # Outer term 6: U6 = A10 - A00, V6 = B00 + B01
    U6_blocks = []
    for r in range(2):
        for c in range(2):
            U6_blocks.append({(r+2,c): 1, (r,c): -1})
    V6_blocks = []
    for r in range(2):
        for c in range(2):
            V6_blocks.append({(r,c): 1, (r,c+2): 1})
            
    # Outer term 7: U7 = A01 - A11, V7 = B10 + B11
    U7_blocks = []
    for r in range(2):
        for c in range(2):
            U7_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
    V7_blocks = []
    for r in range(2):
        for c in range(2):
            V7_blocks.append({(r+2,c): 1, (r+2,c+2): 1})

    # Inner Strassen contribution to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    inner_C_contrib = {
        0: [(0,0, 1), (1,1, 1)],      # M1
        1: [(1,0, 1), (1,1, -1)],     # M2
        2: [(0,1, 1), (1,1, 1)],      # M3
        3: [(0,0, 1), (1,0, 1)],      # M4
        4: [(0,0, -1), (0,1, 1)],     # M5
        5: [(1,1, 1)],                # M6
        6: [(0,0, 1)],                # M7
    }

    # Build the 49 terms
    outer_terms = [
        (U1_blocks, V1_blocks),
        (U2_blocks, V2_blocks),
        (U3_blocks, V3_blocks),
        (U4_blocks, V4_blocks),
        (U5_blocks, V5_blocks),
        (U6_blocks, V6_blocks),
        (U7_blocks, V7_blocks)
    ]
    
    col = 0
    for j in range(7):
        Ub = make_block_matrix(outer_terms[j][0])
        Vb = make_block_matrix(outer_terms[j][1])
        
        inner_terms = expand_2x2_to_scalars(Ub, Vb)
        
        for t in range(7):
            u_scalar, v_scalar = inner_terms[t]
            
            # Add to U and V
            for (ai, aj), coeff in u_scalar.items():
                U[ai * M + aj, col] += coeff
            for (bi, bj), coeff in v_scalar.items():
                V[bi * P + bj, col] += coeff
                
            # Determine W contributions
            # The inner term t contributes to certain C blocks (from inner_C_contrib)
            # Each C block is a 2x2 matrix of scalars.
            # The W index for C[i,j] is j*N + i.
            # The C block at (c_br, c_bc) occupies rows 2*c_br..2*c_br+1, cols 2*c_bc..2*c_bc+1.
            w_coeffs = {}
            for (c_br, c_bc, coeff_c) in inner_C_contrib[t]:
                for dr in range(2):
                    for dc in range(2):
                        c_row = 2 * c_br + dr
                        c_col = 2 * c_bc + dc
                        w_idx = c_col * N + c_row
                        w_coeffs[w_idx] = w_coeffs.get(w_idx, 0) + coeff_c
            
            for w_idx, coeff in w_coeffs.items():
                if coeff != 0:
                    W[w_idx, col] += coeff
                    
            col += 1

    # Add 16 terms for the 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(N):
        for j in range(M):
            U[i * M + j, col] = 1.0
            V[j * P + 4, col] = 1.0
            W[4 * N + i, col] = 1.0
            col += 1

    # Verify
    if check_validity(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    if time.time() < DEADLINE:
        Us, Vs, Ws = schoolbook()
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Last resort
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
