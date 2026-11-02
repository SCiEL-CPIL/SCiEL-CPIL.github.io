"""Matrix multiplication tensor decomposition for 4x5 * 5x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Schoolbook decomposition: rank = n*m*p = 100 (guaranteed valid)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    # Strassen-style decomposition for 4x5*5x5 using Kronecker products
    # The idea: split the 4x5*5x5 multiplication into blocks and use
    # Strassen's 7-multiplication scheme for 2x2 blocks
    #
    # For 4x4*4x4: use Kronecker product of two Strassen 2x2 decompositions
    # This gives rank 7*7 = 49 for the 4x4*4x4 core
    #
    # Full decomposition:
    # A (4x5) = [A11 (4x4) | A12 (4x1)]
    # B (5x5) = [B11 (4x4) B12 (4x1); B21 (1x4) B22 (1x1)]
    # C (4x5) = A11*B11 + A11*B12 + A12*B21 + A12*B22
    #
    # Rank contributions:
    # - A11*B11: 4x4*4x4 using Kronecker Strassen: rank 49
    # - A11*B12: 4x4*4x1 schoolbook: rank 16
    # - A12*B21: 4x1*1x4 schoolbook: rank 4
    # - A12*B22: 4x1*1x1 schoolbook: rank 4
    # Total: 49 + 16 + 4 + 4 = 73

    def strassen_2x2_factors():
        """Return U, V, W factors (7 columns each) for 2x2*2x2 -> 2x2."""
        # Strassen's 7 multiplications for 2x2 matrices
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M2 + M3 + M6
        
        # U factors: for A = [[a,b],[c,d]] (flattened as [a,b,c,d])
        # Each rank-r term picks a linear combination of A entries
        U = [
            [1, 0, 0, 1],   # M1: (a+d)
            [0, 0, 1, 1],   # M2: (c+d)
            [1, 0, 0, 0],   # M3: a
            [0, 0, 0, 1],   # M4: d
            [1, 1, 0, 0],   # M5: (a+b)
            [1, 0, -1, 0],  # M6: (c-a) -> actually (a-c)
            [0, 1, 0, -1],  # M7: (b-d)
        ]
        
        # V factors: for B = [[e,f],[g,h]] (flattened as [e,f,g,h])
        V = [
            [1, 0, 0, 1],   # M1: (e+h)
            [1, 0, 0, 0],   # M2: e
            [0, 1, 0, -1],  # M3: (f-h)
            [0, -1, 1, 0],  # M4: (g-e)
            [0, 0, 0, 1],   # M5: h
            [1, 1, 0, 0],   # M6: (e+f)
            [0, 0, 1, 1],   # M7: (g+h)
        ]
        
        # W factors: for C = [[C11,C12],[C21,C22]] (flattened as [C11,C12,C21,C22])
        # Using W[k*n+i] convention with n=2, p=2
        # C11 at index 0, C12 at index 1, C21 at index 2, C22 at index 3
        # But task says W[k*n+i], so for 2x2 output:
        # W[0*2+0]=C00, W[1*2+0]=C10, W[0*2+1]=C01, W[1*2+1]=C11
        # Wait, let me re-read: "W[k*n+i,r]" where output is n x p
        # For n=2, p=2: W[k*2+i] where k in [0,2), i in [0,2)
        # So W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
        # Hmm, actually "W[k*n+i,r]" suggests row k, column i
        # So W[0*2+0]=W[0]=C[0,0], W[0*2+1]=W[1]=C[0,1]
        # W[1*2+0]=W[2]=C[1,0], W[1*2+1]=W[3]=C[1,1]
        
        # From Strassen:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        W = [
            [1, 0, 0, -1, 1, 0, 1],   # C00
            [0, 0, 1, 1, 0, 0, 0],    # C01
            [0, 1, 0, 0, 1, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0],   # C11
        ]
        
        return U, V, W

    def kron_factors(U1, V1, W1, U2, V2, W2):
        """Compute Kronecker product of two factor sets."""
        R1 = len(U1[0]) if U1 else 0
        R2 = len(U2[0]) if U2 else 0
        R = R1 * R2
        
        nm1 = len(U1)
        mp1 = len(V1)
        pn1 = len(W1)
        nm2 = len(U2)
        mp2 = len(V2)
        pn2 = len(W2)
        
        nm = nm1 * nm2
        mp = mp1 * mp2
        pn = pn1 * pn2
        
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        
        for a1 in range(nm1):
            for a2 in range(nm2):
                a = a1 * nm2 + a2
                for r1 in range(R1):
                    for r2 in range(R2):
                        r = r1 * R2 + r2
                        U[a][r] = U1[a1][r1] * U2[a2][r2]
        
        for b1 in range(mp1):
            for b2 in range(mp2):
                b = b1 * mp2 + b2
                for r1 in range(R1):
                    for r2 in range(R2):
                        r = r1 * R2 + r2
                        V[b][r] = V1[b1][r1] * V2[b2][r2]
        
        for c1 in range(pn1):
            for c2 in range(pn2):
                c = c1 * pn2 + c2
                for r1 in range(R1):
                    for r2 in range(R2):
                        r = r1 * R2 + r2
                        W[c][r] = W1[c1][r1] * W2[c2][r2]
        
        return U, V, W

    def schoolbook_factors(n, m, p):
        """Standard schoolbook decomposition for n x m * m x p -> n x p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    def combine_factors(factor_list):
        """Combine multiple factor sets by concatenating columns."""
        if not factor_list:
            return None, None, None, 0
        
        total_rank = sum(f[3] for f in factor_list)
        
        # Determine max sizes
        max_nm = max(f[0].__len__() for f in factor_list)
        max_mp = max(f[1].__len__() for f in factor_list)
        max_pn = max(f[2].__len__() for f in factor_list)
        
        U = [[0] * total_rank for _ in range(max_nm)]
        V = [[0] * total_rank for _ in range(max_mp)]
        W = [[0] * total_rank for _ in range(max_pn)]
        
        offset = 0
        for U_f, V_f, W_f, rank_f in factor_list:
            for a in range(len(U_f)):
                for r in range(rank_f):
                    U[a][offset + r] = U_f[a][r]
            for b in range(len(V_f)):
                for r in range(rank_f):
                    V[b][offset + r] = V_f[b][r]
            for c in range(len(W_f)):
                for r in range(rank_f):
                    W[c][offset + r] = W_f[c][r]
            offset += rank_f
        
        return U, V, W, total_rank

    # Build Strassen 2x2 factors
    U2, V2, W2 = strassen_2x2_factors()
    
    # Verify Strassen 2x2 is correct
    def verify_2x2(U_f, V_f, W_f):
        U_arr = np.array(U_f, dtype=np.float64)
        V_arr = np.array(V_f, dtype=np.float64)
        W_arr = np.array(W_f, dtype=np.float64)
        # T[a,b,c] where a=0..3 (A entries), b=0..3 (B entries), c=0..3 (C entries)
        # For 2x2: A=[a,b,c,d], B=[e,f,g,h], C=[C00,C01,C10,C11]
        # T should satisfy: C00=sum_j A0j*Bj0, etc.
        # T[a,b,c] = 1 if the product contributes
        # A00*B00 -> C00: T[0,0,0]=1
        # A01*B10 -> C00: T[1,2,0]=1
        # A00*B01 -> C01: T[0,1,1]=1
        # A01*B11 -> C01: T[1,3,1]=1
        # A10*B00 -> C10: T[2,0,2]=1
        # A11*B10 -> C10: T[3,2,2]=1
        # A10*B01 -> C11: T[2,1,3]=1
        # A11*B11 -> C11: T[3,3,3]=1
        T = np.zeros((4, 4, 4), dtype=np.float64)
        T[0,0,0] = 1  # A00*B00->C00
        T[1,2,0] = 1  # A01*B10->C00
        T[0,1,1] = 1  # A00*B01->C01
        T[1,3,1] = 1  # A01*B11->C01
        T[2,0,2] = 1  # A10*B00->C10
        T[3,2,2] = 1  # A11*B10->C10
        T[2,1,3] = 1  # A10*B01->C11
        T[3,3,3] = 1  # A11*B11->C11
        T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(T_rec - T))
        return err
    
    strassen_err = verify_2x2(U2, V2, W2)
    
    # Build 4x4*4x4 using Kronecker product of two Strassen 2x2
    U4, V4, W4 = kron_factors(U2, V2, W2, U2, V2, W2)
    
    # Verify 4x4*4x4
    def verify_4x4(U_f, V_f, W_f):
        nm = 16  # 4*4
        mp = 16
        pn = 16
        U_arr = np.array(U_f, dtype=np.float64)
        V_arr = np.array(V_f, dtype=np.float64)
        W_arr = np.array(W_f, dtype=np.float64)
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    T[a, b, c] = 1
        T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(T_rec - T))
        return err
    
    # Verify the full 4x5*5x5 structure
    # A (4x5): indices 0..19, flattened row-major
    # B (5x5): indices 0..24, flattened row-major  
    # C (4x5): indices 0..19, using W[k*n+i] where n=4, p=5
    # W[k*4+i] for k in [0,5), i in [0,4)
    
    # Build the full decomposition
    factor_list = []
    
    # Block 1: A11 (4x4) * B11 (4x4) -> C11 (4x4)
    # A11 corresponds to A[:, 0:4], indices 0..15
    # B11 corresponds to B[0:4, 0:4], indices 0..15
    # C11 corresponds to C[:, 0:4], using W[k*4+i] for k in [0,4), i in [0,4)
    # We need to embed the 4x4*4x4 factors into the full 4x5*5x5 space
    
    # For embedding, we need to map the local indices to global indices
    # U4 has 16 rows (4x4 A block), V4 has 16 rows (4x4 B block), W4 has 16 rows (4x4 C block)
    
    # Create full-size factors with zeros
    nm_full = n * m  # 4*5 = 20
    mp_full = m * p  # 5*5 = 25
    pn_full = p * n  # 5*4 = 20
    
    # Block 1: A[:, 0:4] * B[0:4, 0:4] -> C[:, 0:4]
    U1_block = [[0] * 49 for _ in range(nm_full)]
    V1_block = [[0] * 49 for _ in range(mp_full)]
    W1_block = [[0] * 49 for _ in range(pn_full)]
    for local_a in range(16):
        global_a = local_a  # A[:, 0:4] is at positions 0..15
        for r in range(49):
            U1_block[global_a][r] = U4[local_a][r]
    for local_b in range(16):
        global_b = local_b  # B[0:4, 0:4] is at positions 0..15
        for r in range(49):
            V1_block[global_b][r] = V4[local_b][r]
    for local_c in range(16):
        # C[:, 0:4]: k in [0,4), i in [0,4), index = k*4+i
        global_c = local_c  # W[k*4+i] for k<4, i<4
        for r in range(49):
            W1_block[global_c][r] = W4[local_c][r]
    factor_list.append((U1_block, V1_block, W1_block, 49))
    
    # Block 2: A[:, 0:4] * B[0:4, 4] -> C[:, 4]
    # A11 (4x4) * B12 (4x1) -> C12 (4x1)
    # Schoolbook: rank = 4*4*1 = 16
    U2_block, V2_block, W2_block, rank2 = schoolbook_factors(4, 4, 1)
    
    U2_full = [[0] * rank2 for _ in range(nm_full)]
    V2_full = [[0] * rank2 for _ in range(mp_full)]
    W2_full = [[0] * rank2 for _ in range(pn_full)]
    for local_a in range(16):
        global_a = local_a
        for r in range(rank2):
            U2_full[global_a][r] = U2_block[local_a][r]
    for local_b in range(4):
        # B[0:4, 4]: j in [0,4), k=4, index = j*5+4
        global_b = local_b * 5 + 4
        for r in range(rank2):
            V2_full[global_b][r] = V2_block[local_b][r]
    for local_c in range(4):
        # C[:, 4]: k=4, i in [0,4), index = 4*4+i = 16+i
        global_c = 16 + local_c
        for r in range(rank2):
            W2_full[global_c][r] = W2_block[local_c][r]
    factor_list.append((U2_full, V2_full, W2_full, rank2))
    
    # Block 3: A[:, 4] * B[4, 0:4] -> C[:, 0:4]
    # A12 (4x1) * B21 (1x4) -> C21 (4x4)
    # Schoolbook: rank = 4*1*4 = 16
    U3_block, V3_block, W3_block, rank3 = schoolbook_factors(4, 1, 4)
    
    U3_full = [[0] * rank3 for _ in range(nm_full)]
    V3_full = [[0] * rank3 for _ in range(mp_full)]
    W3_full = [[0] * rank3 for _ in range(pn_full)]
    for local_a in range(4):
        # A[:, 4]: i in [0,4), j=4, index = i*5+4
        global_a = local_a * 5 + 4
        for r in range(rank3):
            U3_full[global_a][r] = U3_block[local_a][r]
    for local_b in range(4):
        # B[4, 0:4]: j=4, k in [0,4), index = 4*5+k = 20+k
        global_b = 20 + local_b
        for r in range(rank3):
            V3_full[global_b][r] = V3_block[local_b][r]
    for local_c in range(16):
        # C[:, 0:4]: k in [0,4), i in [0,4), index = k*4+i
        global_c = local_c
        for r in range(rank3):
            W3_full[global_c][r] = W3_block[local_c][r]
    factor_list.append((U3_full, V3_full, W3_full, rank3))
    
    # Block 4: A[:, 4] * B[4, 4] -> C[:, 4]
    # A12 (4x1) * B22 (1x1) -> C22 (4x1)
    # Schoolbook: rank = 4*1*1 = 4
    U4_block, V4_block, W4_block, rank4 = schoolbook_factors(4, 1, 1)
    
    U4_full = [[0] * rank4 for _ in range(nm_full)]
    V4_full = [[0] * rank4 for _ in range(mp_full)]
    W4_full = [[0] * rank4 for _ in range(pn_full)]
    for local_a in range(4):
        # A[:, 4]: i in [0,4), j=4, index = i*5+4
        global_a = local_a * 5 + 4
        for r in range(rank4):
            U4_full[global_a][r] = U4_block[local_a][r]
    # B[4, 4]: j=4, k=4, index = 4*5+4 = 24
    V4_full[24][0] = 1
    for r in range(rank4):
        V4_full[24][r] = V4_block[0][r]
    for local_c in range(4):
        # C[:, 4]: k=4, i in [0,4), index = 4*4+i = 16+i
        global_c = 16 + local_c
        for r in range(rank4):
            W4_full[global_c][r] = W4_block[local_c][r]
    factor_list.append((U4_full, V4_full, W4_full, rank4))
    
    # Combine all blocks
    U_combined, V_combined, W_combined, total_rank = combine_factors(factor_list)
    
    # Verify the full decomposition
    U_arr = np.array(U_combined, dtype=np.float64)
    V_arr = np.array(V_combined, dtype=np.float64)
    W_arr = np.array(W_combined, dtype=np.float64)
    
    T = np.zeros((nm_full, mp_full, pn_full), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    err = np.max(np.abs(T_rec - T))
    
    if err < 1e-6:
        # Convert to list format with integer/half-integer entries
        def to_clean_numbers(matrix):
            result = []
            for row in matrix:
                clean_row = []
                for val in row:
                    # Round to nearest 0.5
                    rounded = round(val * 2) / 2
                    if abs(rounded - int(rounded)) < 1e-10:
                        clean_row.append(int(rounded))
                    else:
                        clean_row.append(rounded)
                result.append(clean_row)
            return result
        
        U_final = to_clean_numbers(U_combined)
        V_final = to_clean_numbers(V_combined)
        W_final = to_clean_numbers(W_combined)
        
        # Final verification
        U_check = np.array(U_final, dtype=np.float64)
        V_check = np.array(V_final, dtype=np.float64)
        W_check = np.array(W_final, dtype=np.float64)
        T_rec_check = np.einsum('ar,br,cr->abc', U_check, V_check, W_check)
        err_check = np.max(np.abs(T_rec_check - T))
        
        if err_check < 1e-6:
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback to schoolbook if Strassen approach failed
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

