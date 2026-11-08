import time
import numpy as np

N, M, P = 4, 4, 9
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 3.0

def _time_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)

def _validate(U, V, W):
    n, m, p = N, M, P
    r = len(U[0])
    # Build tensors
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check that for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if i==j==k (trivially not right)
    # Actually: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k? No.
    # It should be 1 if the (i,j,k) term is part of the matrix multiplication.
    # Matrix multiplication C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # So yes, for ALL i in 0..n-1, j in 0..m-1, k in 0..p-1, the sum should be 1.
    # Wait, is that right?
    # The tensor for matrix multiplication is T[i,j,k] = 1 if the indices are valid.
    # So T is the all-ones tensor of size n x m x p.
    # So yes, the decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # AND the decomposition should be exact, meaning no extra terms.
    # Since the tensor is all ones, there are no "extra" terms to check other than the ones that should be 1.
    # But wait, the standard check is:
    # 1. For all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1
    # 2. For all "invalid" combinations (if any), the sum should be 0.
    # But in matrix multiplication, ALL combinations of i,j,k are "valid" in the sense that the tensor element is 1.
    # So condition 1 is sufficient.
    # However, we must also ensure that the factors are of the correct shape and that the decomposition is of the tensor.
    # Let's just check condition 1.
    
    # Use numpy for speed
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # We need to compute S[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a tensor contraction.
    
    # We can reshape U, V, W to 3D tensors and use einsum.
    # U_tensor[i,j,c] = U[i*m+j, c]
    # V_tensor[j,k,c] = V[j*p+k, c]
    # W_tensor[k,i,c] = W[k*n+i, c]
    # S[i,j,k] = sum_c U_tensor[i,j,c] * V_tensor[j,k,c] * W_tensor[k,i,c]
    
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # S[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
    # This is not a standard einsum because of the i in W_t.
    # Let's use a loop over i to be safe, or reshape cleverly.
    # Actually, for fixed i, S[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
    # Let W_t_i = W_t[:, i, :]  # shape (p, r)
    # Then S[i,j,k] = sum_c U_t[i,j,c] * (V_t * W_t_i)[j,k,c]  -- no, that's not right.
    
    # Let's just use a loop over i, j, k. It's small (4*4*9=144).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_2x2x2_strassen():
    # Strassen 2x2x2
    # n=m=p=2
    # rank = 7
    # U, V, W of size 4x7
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M3
    # C01 = M5 + M6
    # C11 = M1 - M2 + M3 + M4
    
    r = 7
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(4)]
    W = [[0]*r for _ in range(4)]
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01  -- Wait, C01 is k=0,i=1? No, C01 is C[0,1] which is i=0,k=1.
    # W index is k*2+i.
    # C00: i=0,k=0 -> 0
    # C01: i=0,k=1 -> 1
    # C10: i=1,k=0 -> 2
    # C11: i=1,k=1 -> 3
    # So W[1][2] is C01. Correct.
    W[2][2] = 1  # C10
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[3][3] = 1  # C11
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1  # C01 (from C00 = ... - M5, so W[0][4] should be -1? No, C00 = M1 + M4 - M5 + M7. So W[0][4] = -1, W[1][4] = 1? No, M5 is in C01 and C00.)
    # C01 = M5 + M6. So W[1][4] = 1.
    # C00 = M1 + M4 - M5 + M7. So W[0][4] = -1.
    W[0][4] = -1
    W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1  # C01
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    W[3][6] = -1  # C11 (from C11 = M1 - M2 + M3 + M4, M7 is not in C11? Wait, C11 = M1 - M2 + M3 + M4. M7 is not there.)
    # Let's recheck C11.
    # C11 = M1 - M2 + M3 + M4
    # M1: W[3][0] = 1
    # M2: W[3][1] = -1
    # M3: W[3][2] = 1
    # M4: W[3][3] = 1
    # M5: W[3][4] = 0
    # M6: W[3][5] = 0
    # M7: W[3][6] = 0
    # So W[3][6] should be 0.
    W[3][6] = 0
    
    # Let's re-verify all W entries.
    # C00 = M1 + M4 - M5 + M7
    # M1: W[0][0]=1
    # M4: W[0][3]=1
    # M5: W[0][4]=-1
    # M7: W[0][6]=1
    # C00: [1, 0, 0, 1, -1, 0, 1]
    
    # C01 = M5 + M6
    # M5: W[1][4]=1
    # M6: W[1][5]=1
    # C01: [0, 0, 0, 0, 1, 1, 0]
    
    # C10 = M2 + M3
    # M2: W[2][1]=1
    # M3: W[2][2]=1
    # C10: [0, 1, 1, 0, 0, 0, 0]
    
    # C11 = M1 - M2 + M3 + M4
    # M1: W[3][0]=1
    # M2: W[3][1]=-1
    # M3: W[3][2]=1
    # M4: W[3][3]=1
    # C11: [1, -1, 1, 1, 0, 0, 0]
    
    # Let's re-fill W carefully.
    W = [[0]*7 for _ in range(4)]
    # C00 (idx 0)
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    # C01 (idx 1)
    W[1][4] = 1
    W[1][5] = 1
    # C10 (idx 2)
    W[2][1] = 1
    W[2][2] = 1
    # C11 (idx 3)
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][3] = 1
    
    # U and V seem correct from above.
    return U, V, W

def _build_4x4x4_strassen():
    # Compose 7 Strassen 2x2x2 blocks for 4x4x4.
    # n=m=p=4
    # rank = 7 * 7 = 49
    U2, V2, W2 = _build_2x2x2_strassen()
    r2 = len(U2[0])
    r = 7 * r2
    n = 4
    U = [[0]*r for _ in range(n*n)]
    V = [[0]*r for _ in range(n*n)]
    W = [[0]*r for _ in range(n*n)]
    
    # The 7 Strassen terms for 4x4x4 are the same as 2x2x2 but with 2x2 blocks.
    # Let's denote the blocks of A as A00, A01, A10, A11 (each 2x2).
    # The 7 products are M1..M7, each a 2x2x2 multiplication.
    # For each term t in 0..6, we have a 2x2x2 decomposition.
    # We need to map the local indices to global.
    
    # U_global[i*4+j, c]
    # V_global[j*4+k, c]
    # W_global[k*4+i, c]
    
    # For term t, the local U2 is 2x2x2.
    # U2[i_loc*2+j_loc, s]
    # V2[j_loc*2+k_loc, s]
    # W2[k_loc*2+i_loc, s]
    
    # The coefficients for the A blocks in each term:
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    # The coefficients for the B blocks:
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # The coefficients for the C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M3
    # C01 = M5 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # Block positions:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    # Similarly for B and C.
    
    coeff_A = [
        {(0,0):1, (1,1):1},  # M1
        {(1,0):1, (1,1):1},  # M2
        {(0,0):1},           # M3
        {(1,1):1},           # M4
        {(0,0):1, (0,1):1},  # M5
        {(1,0):1, (0,0):-1}, # M6
        {(0,1):1, (1,1):-1}  # M7
    ]
    coeff_B = [
        {(0,0):1, (1,1):1},  # M1
        {(0,0):1},           # M2
        {(0,1):1, (1,1):-1}, # M3
        {(1,0):1, (0,0):-1}, # M4
        {(1,1):1},           # M5
        {(0,0):1, (0,1):1},  # M6
        {(1,0):1, (1,1):1}   # M7
    ]
    coeff_C = [
        # C00: M1 + M4 - M5 + M7
        {0:1, 3:1, 4:-1, 6:1},
        # C10: M2 + M3
        {1:1, 2:1},
        # C01: M5 + M6
        {4:1, 5:1},
        # C11: M1 - M2 + M3 + M4
        {0:1, 1:-1, 2:1, 3:1}
    ]
    
    for t in range(7):
        for s in range(r2):
            c = t * r2 + s
            # Fill U
            for (r_a, c_a), ca in coeff_A[t].items():
                for i_loc in range(2):
                    for j_loc in range(2):
                        i = 2*r_a + i_loc
                        j = 2*c_a + j_loc
                        U[i*4+j][c] += ca * U2[i_loc*2+j_loc][s]
            # Fill V
            for (r_b, c_b), cb in coeff_B[t].items():
                for j_loc in range(2):
                    for k_loc in range(2):
                        j = 2*r_b + j_loc
                        k = 2*c_b + k_loc
                        V[j*4+k][c] += cb * V2[j_loc*2+k_loc][s]
            # Fill W
            for c_block in range(4):
                r_c = c_block // 2
                c_c = c_block % 2
                cc = coeff_C[c_block].get(t, 0)
                if cc == 0: continue
                for k_loc in range(2):
                    for i_loc in range(2):
                        k = 2*r_c + k_loc
                        i = 2*c_c + i_loc
                        W[k*4+i][c] += cc * W2[k_loc*2+i_loc][s]
    return U, V, W, r

def _build_4x4x1_schoolbook():
    n, m, p = 4, 4, 1
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W, r

def construct():
    # 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # Rank = 49 + 49 + 16 = 114
    # Schoolbook rank = 4*4*9 = 144.
    # So 114 is better.
    
    U1, V1, W1, r1 = _build_4x4x4_strassen()
    U2, V2, W2, r2 = _build_4x4x4_strassen()
    U3, V3, W3, r3 = _build_4x4x1_schoolbook()
    
    r = r1 + r2 + r3
    n, m, p = 4, 4, 9
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Copy Part 1 (cols 0..r1-1)
    for i in range(n*m):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for j in range(m):
        for k in range(4):
            idx_v = j*p + k
            for c in range(r1):
                V[idx_v][c] = V1[j*4+k][c]
    for k in range(4):
        for i in range(n):
            idx_w = k*n + i
            for c in range(r1):
                W[idx_w][c] = W1[k*4+i][c]
                
    # Copy Part 2 (cols r1..r1+r2-1)
    off2 = r1
    for i in range(n*m):
        for c in range(r2):
            U[i][off2+c] = U2[i][c]
    for j in range(m):
        for k in range(4):
            idx_v = j*p + (k+4)
            for c in range(r2):
                V[idx_v][off2+c] = V2[j*4+k][c]
    for k in range(4):
        for i in range(n):
            idx_w = (k+4)*n + i
            for c in range(r2):
                W[idx_w][off2+c] = W2[k*4+i][c]
                
    # Copy Part 3 (cols r1+r2..r-1)
    off3 = r1 + r2
    for i in range(n*m):
        for c in range(r3):
            U[i][off3+c] = U3[i][c]
    for j in range(m):
        idx_v = j*p + 8
        for c in range(r3):
            V[idx_v][off3+c] = V3[j*1+0][c]
    for i in range(n):
        idx_w = 8*n + i
        for c in range(r3):
            W[idx_w][off3+c] = W3[0*4+i][c]
            
    # Validate
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
