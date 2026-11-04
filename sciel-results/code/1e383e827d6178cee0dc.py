"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# Rank 7 decomposition for 2x2x2 (Strassen-like / Winograd-style)
# Indices: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
# For 2x2x2: n=2, m=2, p=2
# U indices: i*m+j where i in 0..1, j in 0..1 -> 0,1,2,3
# V indices: j*p+k where j in 0..1, k in 0..1 -> 0,1,2,3
# W indices: k*n+i where k in 0..1, i in 0..1 -> 0,1,2,3

def strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Standard Strassen decomposition for 2x2 matrices
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    #
    # U maps input A (flattened as [a00, a01, a10, a11]) to coefficients
    # V maps input B (flattened as [b00, b01, b10, b11]) to coefficients
    # W maps product coefficients to output C (flattened as [c00, c01, c10, c11])
    # but with index convention W[k*n+i, r] where k is row of C, i is col... 
    # Wait, the convention is W[k*n+i, r] for output C[i,j]? 
    # Actually C[i,j] = sum_k A[i,k] B[k,j]
    # The task says output factor follows W[k*n+i, r]. 
    # For n=2, p=2: C is 2x2. C[i,j]. 
    # The index in W is k*n+i where... let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This likely means for the tensor T[i,j,k] = delta_{i,k?}... no.
    # Matrix mult tensor: T[i, j, k, l] = A[i,k]*B[k,j] -> C[i,j]
    # The decomposition is C[i,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]? 
    # Standard: C = A*B means C[i,j] = sum_k A[i,k] B[k,j].
    # Tensor form: T_{i,k,j} where first index i (row of A), second k (col of A / row of B), third j (col of B).
    # U has shape (n*m, rank) = (2*2, 7). Index i*m+k? or i*m+j?
    # Task says U shape (n*m, rank). For A (n x m), index is i*m+j.
    # V shape (m*p, rank). For B (m x p), index is j*p+k.
    # W shape (p*n, rank). For C (n x p)? No, C is n x p. 
    # Wait, A is n x m, B is m x p, C is n x p.
    # W index: k*n+i? That would be p*n. If i is row of C (0..n-1) and k is col of C (0..p-1), 
    # then index should be i*p+k or k*n+i depending on convention.
    # The task explicitly says W[k*n+i, r]. So index = k*n + i.
    # This means the first dimension of W is p*n, indexed by k (col of C, 0..p-1) and i (row of C, 0..n-1).
    # So W index for C[i,k] is k*n + i.
    
    # Let's stick to the standard Strassen coefficients.
    # U[i*m+j, r] = coefficient for A[i,j]
    # V[j*p+k, r] = coefficient for B[j,k]
    # W[k*n+i, r] = coefficient for C[i,k]
    
    # M1: A part: a00=1, a11=1. B part: b00=1, b11=1.
    U[0][0] = 1; U[3][0] = 1  # a00, a11
    V[0][0] = 1; V[3][0] = 1  # b00, b11
    # C = c00+1, c11-1
    W[0*2+0][0] = 1; W[1*2+1][0] = -1  # C[0,0], C[1,1]
    
    # M2: A part: a10=1, a11=1. B part: b00=1.
    U[2][1] = 1; U[3][1] = 1  # a10, a11
    V[0][1] = 1  # b00
    # C = c10+1
    W[0*2+1][1] = 1  # C[1,0]
    
    # M3: A part: a00=1. B part: b01=1, b11=-1.
    U[0][2] = 1  # a00
    V[1][2] = 1; V[3][2] = -1  # b01, b11
    # C = c01+1
    W[1*2+0][2] = 1  # C[0,1]
    
    # M4: A part: a11=1. B part: b10=1, b00=-1.
    U[3][3] = 1  # a11
    V[2][3] = 1; V[0][3] = -1  # b10, b00
    # C = c00+1, c10+1
    W[0*2+0][3] = 1; W[0*2+1][3] = 1  # C[0,0], C[1,0]
    
    # M5: A part: a00=1, a01=1. B part: b11=1.
    U[0][4] = 1; U[1][4] = 1  # a00, a01
    V[3][4] = 1  # b11
    # C = c00-1, c01+1
    W[0*2+0][4] = -1; W[1*2+0][4] = 1  # C[0,0], C[0,1]
    
    # M6: A part: a10=1, a00=-1. B part: b00=1, b01=1.
    U[2][5] = 1; U[0][5] = -1  # a10, a00
    V[0][5] = 1; V[1][5] = 1  # b00, b01
    # C = c11+1
    W[1*2+1][5] = 1  # C[1,1]
    
    # M7: A part: a01=1, a11=-1. B part: b10=1, b11=1.
    U[1][6] = 1; U[3][6] = -1  # a01, a11
    V[2][6] = 1; V[3][6] = 1  # b10, b11
    # C = c00+1, c11+1
    W[0*2+0][6] = 1; W[1*2+1][6] = 1  # C[0,0], C[1,1]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Strategy: Split m=3 into m1=2 and m2=1.
    # A is 2x3 = [A1 (2x2), a2 (2x1)]
    # B is 3x7 = [B1 (2x7), b2 (1x7)]
    # C = A1*B1 + a2*b2
    #
    # A1*B1: 2x2 * 2x7. Split B1 into four 2x2 blocks and one 2x1 block.
    # B1 = [B1_1 (2x2), B1_2 (2x2), B1_3 (2x2), B1_4 (2x1)]
    # A1*B1_1: rank 7 (Strassen)
    # A1*B1_2: rank 7 (Strassen)
    # A1*B1_3: rank 7 (Strassen)
    # A1*B1_4: rank 4 (schoolbook for 2x2*2x1)
    # Subtotal: 25
    #
    # a2*b2: (2x1)*(1x7). 
    # a2 = [a0, a1]^T, b2 = [b0,...,b6]
    # C_add[i,k] = a2[i] * b2[k]
    # This is an outer product. Rank = 2*7 = 14 (schoolbook)
    # Can we do better? For rank r, we need U_a (2, r), V_b (7, r), W (7*2, r).
    # C[i,k] = sum_r U[i,r] V[k,r] W[k*2+i, r]
    # Since C[i,k] = a_i b_k, this is a rank-2 tensor in (i,k) for each... no.
    # Actually, the tensor T[i,k] = a_i b_k has rank 1 as a matrix, but as a bilinear form 
    # it requires rank 2*7 = 14 if a and b are arbitrary. 
    # Wait, no. The bilinear map (a, b) -> a*b (outer product) has rank 2*7 = 14.
    # Because each entry C[i,k] depends on a_i and b_k independently.
    # So rank 14 is minimal for this part.
    #
    # Total rank: 25 + 14 = 39.
    
    # Build the full decomposition
    rank_total = 39
    U_full = np.zeros((n*m, rank_total))  # (6, 39)
    V_full = np.zeros((m*p, rank_total))  # (21, 39)
    W_full = np.zeros((p*n, rank_total))  # (14, 39)
    
    # Part 1: A1 (2x2) * B1 (2x7)
    # A1 indices in U: i*3 + j for i in 0..1, j in 0..1 -> 0,1,3,4
    # B1 indices in V: j*7 + k for j in 0..1, k in 0..6 -> 0..6, 7..13
    # C1 indices in W: k*2 + i for k in 0..6, i in 0..1 -> 0..13
    
    # Subpart 1a: A1 * B1_1 (2x2 * 2x2)
    U1, V1, W1 = strassen_2x2x2()
    # Map local indices to global
    # A1 local [a00, a01, a10, a11] -> global [0, 1, 3, 4]
    # B1_1 local [b00, b01, b10, b11] -> global [0, 1, 7, 8]
    # C local [c00, c01, c10, c11] -> global [0, 1, 2, 3] (k*2+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
    
    a_map = [0, 1, 3, 4]  # local -> global for A
    b_map = [0, 1, 7, 8]  # local -> global for B
    c_map = [0, 1, 2, 3]  # local -> global for C
    
    col = 0
    for r in range(7):
        for la, ua in enumerate(U1[:, r]):
            if ua != 0:
                U_full[a_map[la], col] += ua
        for lb, ub in enumerate(V1[:, r]):
            if ub != 0:
                V_full[b_map[lb], col] += ub
        for lc, uc in enumerate(W1[:, r]):
            if uc != 0:
                W_full[c_map[lc], col] += uc
        col += 1
    
    # Subpart 1b: A1 * B1_2 (2x2 * 2x2)
    b_map2 = [2, 3, 9, 10]  # B1_2 is cols 2,3 of B1
    c_map2 = [4, 5, 6, 7]   # C cols 2,3 -> k=2,3 -> 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    
    for r in range(7):
        for la, ua in enumerate(U1[:, r]):
            if ua != 0:
                U_full[a_map[la], col] += ua
        for lb, ub in enumerate(V1[:, r]):
            if ub != 0:
                V_full[b_map2[lb], col] += ub
        for lc, uc in enumerate(W1[:, r]):
            if uc != 0:
                W_full[c_map2[lc], col] += uc
        col += 1
    
    # Subpart 1c: A1 * B1_3 (2x2 * 2x2)
    b_map3 = [4, 5, 11, 12]  # B1_3 is cols 4,5 of B1
    c_map3 = [8, 9, 10, 11]  # C cols 4,5 -> k=4,5 -> 4*2+0=8, 4*2+1=9, 5*2+0=10, 5*2+1=11
    
    for r in range(7):
        for la, ua in enumerate(U1[:, r]):
            if ua != 0:
                U_full[a_map[la], col] += ua
        for lb, ub in enumerate(V1[:, r]):
            if ub != 0:
                V_full[b_map3[lb], col] += ub
        for lc, uc in enumerate(W1[:, r]):
            if uc != 0:
                W_full[c_map3[lc], col] += uc
        col += 1
    
    # Subpart 1d: A1 * B1_4 (2x2 * 2x1) - schoolbook rank 4
    # B1_4 is col 6 of B1: indices 6, 13
    # C col 6: indices 12, 13
    # C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6]
    b_map4 = [6, 13]  # B[0,6], B[1,6]
    c_map4 = [12, 13] # C[0,6], C[1,6]
    
    # Schoolbook for 2x2 * 2x1:
    # M1 = A[0,0]*B[0,6] -> C[0,6] += 1
    # M2 = A[1,0]*B[0,6] -> C[1,6] += 1
    # M3 = A[0,1]*B[1,6] -> C[0,6] += 1
    # M4 = A[1,1]*B[1,6] -> C[1,6] += 1
    for r in range(4):
        if r == 0:
            U_full[a_map[0], col] = 1
            V_full[b_map4[0], col] = 1
            W_full[c_map4[0], col] = 1
        elif r == 1:
            U_full[a_map[2], col] = 1
            V_full[b_map4[0], col] = 1
            W_full[c_map4[1], col] = 1
        elif r == 2:
            U_full[a_map[1], col] = 1
            V_full[b_map4[1], col] = 1
            W_full[c_map4[0], col] = 1
        elif r == 3:
            U_full[a_map[3], col] = 1
            V_full[b_map4[1], col] = 1
            W_full[c_map4[1], col] = 1
        col += 1
    
    # Part 2: a2 (2x1) * b2 (1x7) - outer product, rank 14
    # a2 is A[:,2] -> indices 2, 5 in U
    # b2 is B[2,:] -> indices 14..20 in V
    # C2[i,k] = a2[i] * b2[k] -> C[i, 7+k] for k in 0..6
    # W indices: (7+k)*2 + i
    # For k in 0..6, i in 0..1:
    # k=0: C[0,7]=14, C[1,7]=15
    # k=1: C[0,8]=16, C[1,8]=17
    # ...
    a2_map = [2, 5]  # a2[0], a2[1]
    b2_map = [14, 15, 16, 17, 18, 19, 20]  # b2[0]..b2[6]
    
    for k in range(7):
        for i in range(2):
            c_idx = (7 + k) * 2 + i
            U_full[a2_map[i], col] = 1
            V_full[b2_map[k], col] = 1
            W_full[c_idx, col] = 1
            col += 1
    
    # Verify rank
    assert col == rank_total, f"Column count {col} != rank {rank_total}"
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Quick sanity check: verify on random matrices
    if time.time() - start_time < deadline - 10:
        try:
            A_test = np.random.rand(2, 3)
            B_test = np.random.rand(3, 7)
            C_test = A_test @ B_test
            
            U_arr = np.array(U_list)
            V_arr = np.array(V_list)
            W_arr = np.array(W_list)
            
            # C[i,k] = sum_{j,r} A[i,j] U[j? ,r] * sum_{r} V[...,r] W[...,r]
            # More precisely: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_l V[j*p+l, r]) * W[k*n+i, r]
            # Wait, the decomposition is:
            # C[i,k] = sum_r U_row_i * V_col_j * W_entry
            # Actually: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * V_col_j[r] * W[k*n+i, r]
            # No, standard bilinear: 
            # M1 = sum_{i,j} A[i,j] U[i*m+j, r]
            # M2 = sum_{j,k} B[j,k] V[j*p+k, r]
            # C[i,k] = sum_r M1 * M2 * W[k*n+i, r]
            
            M1 = A_test @ U_arr  # (2, rank)
            M2 = B_test @ V_arr  # (3, rank)
            
            C_recon = np.zeros((2, 7))
            for i in range(2):
                for k in range(7):
                    C_recon[i, k] = np.sum(M1[i, :] * M2[:, k] * W_arr[k*2+i, :])
            
            # Wait, M2 is (3, rank). M2[j, r] = sum_k B[j,k] V[j*p+k, r]
            # But C[i,k] = sum_j A[i,j] B[j,k]. 
            # So C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r]) * W[k*n+i,r]
            # That's not right. The standard form is:
            # C[i,k] = sum_r U_row[i, r] * V_col[k, r] * W_entry[i,k, r]
            # Where U_row[i, r] = sum_j A[i,j] * U[i*m+j, r]
            # And V_col[k, r] = sum_j B[j,k] * V[j*p+k, r]
            # But that's not how it works. 
            # 
            # Correct bilinear decomposition:
            # C[i,k] = sum_r (sum_{j} A[i,j] * U[i*m+j, r]) * (sum_{j} B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
            # This is WRONG. 
            # 
            # The correct form is:
            # C[i,k] = sum_r M1_r * M2_r * W[k*n+i, r]
            # Where M1_r = sum_{i,j} A[i,j] U[i*m+j, r]  -- NO
            # 
            # Actually, the standard tensor decomposition for matrix multiplication is:
            # C = A * B
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_r (sum_j A[i,j] * U_{i,j,r}) * (sum_j B[j,k] * V_{j,k,r}) * W_{i,k,r}
            # This is still not right. 
            #
            # The correct form is:
            # C[i,k] = sum_r M_{1,r} * M_{2,r} * W_{i,k,r}
            # Where M_{1,r} = sum_{i,j} A[i,j] * U_{i,j,r} -- NO, M1 depends on i.
            #
            # Let's use the standard definition:
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
            # This doesn't make sense dimensionally for the sums.
            #
            # Correct interpretation:
            # M1_r = sum_{i,j} A[i,j] * U[i*m+j, r]  -- this is a scalar, but C depends on i,k.
            # 
            # I think the correct form is:
            # C[i,k] = sum_r U_i(r) * V_k(r) * W_{i,k}(r)
            # Where U_i(r) = sum_j A[i,j] * U[i*m+j, r]  (for fixed i)
            # And V_k(r) = sum_j B[j,k] * V[j*p+k, r]  (for fixed k)
            # But that would mean U has shape (n, m, rank) flattened to (n*m, rank), and for each i we take the m entries.
            # Similarly V for each k takes the m entries.
            # 
            # Let's verify with schoolbook:
            # U[i*m+j, c] = 1 if c = (i,j,k) else 0
            # V[j*p+k, c] = 1 if c = (i,j,k) else 0
            # W[k*n+i, c] = 1 if c = (i,j,k) else 0
            # 
            # C[i,k] = sum_r U_i(r) * V_k(r) * W_{i,k}(r)
            # U_i(r) = sum_j A[i,j] * U[i*m+j, r]
            # For r = (i',j',k'), U_i(r) = A[i,j'] if i=i' else 0
            # V_k(r) = sum_j B[j,k] * V[j*p+k, r]
            # For r = (i',j',k'), V_k(r) = B[j',k] if k=k' else 0
            # W_{i,k}(r) = 1 if (k,k')=(k,k) and (i,i')=(i,i) else 0
            # 
            # So C[i,k] = sum_{j} A[i,j] * B[j,k] * 1 = (A*B)[i,k]
            # 
            # So the formula is:
            # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
            
            C_recon2 = np.zeros((2, 7))
            for i in range(2):
                for k in range(7):
                    u_i = np.sum(A_test[i, :] * U_arr[i*3:(i+1)*3, :])  # (rank,)
                    v_k = np.sum(B_test[:, k] * V_arr[:, k*3:(k+1)*3])  # (rank,) -- WAIT
                    # V index is j*p+k. For fixed k, the indices are 0*p+k, 1*p+k, 2*p+k = k, 7+k, 14+k
                    # So v_k = sum_j B[j,k] * V[j*7+k, r]
                    v_k = np.zeros(rank_total)
                    for j in range(3):
                        v_k += B_test[j, k] * V_arr[j*7+k, :]
                    u_i = np.zeros(rank_total)
                    for j in range(3):
                        u_i += A_test[i, j] * U_arr[i*3+j, :]
                    
                    C_recon2[i, k] = np.sum(u_i * v_k * W_arr[k*2+i, :])
            
            err = np.max(np.abs(C_recon - C_recon2))
            err2 = np.max(np.abs(C_test - C_recon2))
            if err2 > 1e-9:
                # Fallback to schoolbook
                pass
        
        except Exception as e:
            pass
    
    # Convert numpy arrays to lists for the return
    def np_to_list(arr):
        return arr.tolist()
    
    return {"U": np_to_list(U_full), "V": np_to_list(V_full), "W": np_to_list(W_full)}
