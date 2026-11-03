"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Block decomposition
    # n=3 -> 2+1
    # m=4 -> 2+2
    # p=5 -> 2+3
    
    n1, n2 = 2, 1
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    # We will build U, V, W matrices
    # U size: n*m x Rank
    # V size: m*p x Rank
    # W size: p*n x Rank
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of length n*m
        v_vec: list of length m*p
        w_vec: list of length p*n
        """
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    def zero_vec(size):
        return [0] * size

    # Helper to create index
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # We need to implement Strassen for 2x2x2 and the specific low-rank decompositions for the other blocks.
    
    # Strassen 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (a-b)(e+g)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def add_strassen_block(i_off, j_off, k_off):
        """
        Adds Strassen terms for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        contributing to C[i_off:i_off+2, k_off:k_off+2]
        """
        # m1 = (a+d)(e+h)
        u1 = zero_vec(n*m); v1 = zero_vec(m*p); w1 = zero_vec(p*n)
        # A part: a+d -> A[i,j] + A[i+1,j+1]
        u1[idx_u(i_off, j_off)] = 1
        u1[idx_u(i_off+1, j_off+1)] = 1
        # B part: e+h -> B[j,k] + B[j+1,k+1]
        v1[idx_v(j_off, k_off)] = 1
        v1[idx_v(j_off+1, k_off+1)] = 1
        # W part: C11 += 1, C22 += 1
        w1[idx_w(k_off, i_off)] = 1
        w1[idx_w(k_off+1, i_off+1)] = 1
        add_term(u1, v1, w1)
        
        # m2 = (c+d)e
        u2 = zero_vec(n*m); v2 = zero_vec(m*p); w2 = zero_vec(p*n)
        u2[idx_u(i_off+1, j_off)] = 1
        u2[idx_u(i_off+1, j_off+1)] = 1
        v2[idx_v(j_off, k_off)] = 1
        w2[idx_w(k_off, i_off+1)] = 1
        add_term(u2, v2, w2)
        
        # m3 = a(f-h)
        u3 = zero_vec(n*m); v3 = zero_vec(m*p); w3 = zero_vec(p*n)
        u3[idx_u(i_off, j_off)] = 1
        v3[idx_v(j_off, k_off+1)] = 1
        v3[idx_v(j_off+1, k_off+1)] = -1
        w3[idx_w(k_off+1, i_off)] = 1
        add_term(u3, v3, w3)
        
        # m4 = d(g-c)
        u4 = zero_vec(n*m); v4 = zero_vec(m*p); w4 = zero_vec(p*n)
        u4[idx_u(i_off+1, j_off+1)] = 1
        v4[idx_v(j_off+1, k_off)] = 1
        v4[idx_v(j_off, k_off)] = -1
        w4[idx_w(k_off, i_off+1)] = 1
        w4[idx_w(k_off+1, i_off+1)] = 1
        add_term(u4, v4, w4)
        
        # m5 = (a+b)h
        u5 = zero_vec(n*m); v5 = zero_vec(m*p); w5 = zero_vec(p*n)
        u5[idx_u(i_off, j_off)] = 1
        u5[idx_u(i_off, j_off+1)] = 1
        v5[idx_v(j_off+1, k_off+1)] = 1
        w5[idx_w(k_off+1, i_off)] = -1
        w5[idx_w(k_off+1, i_off+1)] = 1
        add_term(u5, v5, w5)
        
        # m6 = (c-d)(f+g)
        u6 = zero_vec(n*m); v6 = zero_vec(m*p); w6 = zero_vec(p*n)
        u6[idx_u(i_off+1, j_off)] = 1
        u6[idx_u(i_off+1, j_off+1)] = -1
        v6[idx_v(j_off, k_off+1)] = 1
        v6[idx_v(j_off+1, k_off)] = 1
        w6[idx_w(k_off+1, i_off+1)] = 1
        add_term(u6, v6, w6)
        
        # m7 = (a-b)(e+g)
        u7 = zero_vec(n*m); v7 = zero_vec(m*p); w7 = zero_vec(p*n)
        u7[idx_u(i_off, j_off)] = 1
        u7[idx_u(i_off, j_off+1)] = -1
        v7[idx_v(j_off, k_off)] = 1
        v7[idx_v(j_off+1, k_off)] = 1
        w7[idx_w(k_off, i_off)] = 1
        add_term(u7, v7, w7)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Note: B rows 2:4 correspond to j=2,3.
    # A cols 2:4 correspond to j=2,3.
    add_strassen_block(0, 2, 0)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # B[0:2, 2:5] is 2x3. Split into B[0:2, 2:4] (2x2) and B[0:2, 4:5] (2x1).
    # 3a: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # 3b: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # 2x2x1 Rank 2
    # i_off=0, j_off=0, k_off=4
    # C[0:2, 4:5]
    # Target: C[i, 4] = sum_j A[i,j] B[j,4]
    # Rank 2 solution:
    # r1: U[i=0, j=0]=1, V[j=0, k=4]=1, W[k=4, i=0]=1/2
    # r2: U[i=0, j=1]=1, V[j=1, k=4]=1, W[k=4, i=0]=1/2
    # r3: U[i=1, j=0]=1, V[j=0, k=4]=1, W[k=4, i=1]=1/2
    # r4: U[i=1, j=1]=1, V[j=1, k=4]=1, W[k=4, i=1]=1/2
    # Wait, my previous derivation for 2x2x1 rank 2 was:
    # U^0_1=(1,0), U^0_2=(0,1), V_1=(1,0), V_2=(0,1), W_0_1=1/2, W_0_2=1/2.
    # This covers i=0.
    # For i=1, we need U^1_1=(1,0), U^1_2=(0,1), W_1_1=1/2, W_1_2=1/2.
    # So we need 4 terms?
    # Term 1: i=0, j=0. U[0,0]=1, V[0]=1, W[0]=1/2.
    # Term 2: i=0, j=1. U[0,1]=1, V[1]=1, W[0]=1/2.
    # Term 3: i=1, j=0. U[1,0]=1, V[0]=1, W[1]=1/2.
    # Term 4: i=1, j=1. U[1,1]=1, V[1]=1, W[1]=1/2.
    # Yes, 4 terms.
    # Can we do better?
    # My previous "Rank 2" claim was for the tensor identity sum, but the algorithm requires separate linear forms.
    # Let's re-verify 2x2x1 rank.
    # If rank 2, we have 2 multiplications.
    # M1 = L1(A) L1(B), M2 = L2(A) L2(B).
    # C0 = a M1 + b M2.
    # C1 = c M1 + d M2.
    # C0 = A00 B0 + A01 B1.
    # C1 = A10 B0 + A11 B1.
    # This implies A00 B0 + A01 B1 = a (L1A L1B) + b (L2A L2B).
    # This is generally impossible with rank 2.
    # So 2x2x1 rank is 4.
    
    # Correcting the cost estimates:
    # 2x2x2: 7
    # 2x2x1: 4
    # 1x2x2: 4
    # 1x2x1: 2
    
    # Recalculate Total:
    # C00: 7 + 7 = 14
    # C01: (7+4) + (7+4) = 22
    # C10: 4 + 4 = 8
    # C11: (4+2) + (4+2) = 12
    # Total: 14 + 22 + 8 + 12 = 56.
    
    # So we are back to 56.
    # The "Rank 2" for 2x2x1 was an error in my quick check.
    # I will implement the 56 rank solution.
    
    def add_2x2x1_block(i_off, j_off, k_off):
        # i_off: 0 or 1 (row of A)
        # j_off: 0 or 2 (col of A / row of B)
        # k_off: 4 (col of B)
        # A[i_off, j_off:j_off+2] * B[j_off:j_off+2, k_off] -> C[i_off, k_off]
        # 4 terms
        for j_rel in range(2):
            for i_rel in [0]: # Only 1 row in A block? No, 2x2x1 means A is 2x2?
                # Wait, 2x2x1 means A is 2x2, B is 2x1.
                # In our context A[0:2, 0:2] is 2x2.
                # So i_rel is 0,1.
                pass
        
        # Let's just hardcode the 4 terms for 2x2x1
        # A is 2x2, B is 2x1.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Term 1: A00 B0 -> C0
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[idx_u(i_off, j_off)] = 1
        v[idx_v(j_off, k_off)] = 1
        w[idx_w(k_off, i_off)] = 1
        add_term(u, v, w)
        
        # Term 2: A01 B1 -> C0
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[idx_u(i_off, j_off+1)] = 1
        v[idx_v(j_off+1, k_off)] = 1
        w[idx_w(k_off, i_off)] = 1
        add_term(u, v, w)
        
        # Term 3: A10 B0 -> C1
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[idx_u(i_off+1, j_off)] = 1
        v[idx_v(j_off, k_off)] = 1
        w[idx_w(k_off, i_off+1)] = 1
        add_term(u, v, w)
        
        # Term 4: A11 B1 -> C1
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[idx_u(i_off+1, j_off+1)] = 1
        v[idx_v(j_off+1, k_off)] = 1
        w[idx_w(k_off, i_off+1)] = 1
        add_term(u, v, w)

    def add_1x2x2_block(i_off, j_off, k_off):
        # A[i_off, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2] -> C[i_off, k_off:k_off+2]
        # 4 terms
        for j_rel in range(2):
            for k_rel in range(2):
                u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
                u[idx_u(i_off, j_off+j_rel)] = 1
                v[idx_v(j_off+j_rel, k_off+k_rel)] = 1
                w[idx_w(k_off+k_rel, i_off)] = 1
                add_term(u, v, w)

    def add_1x2x1_block(i_off, j_off, k_off):
        # A[i_off, j_off:j_off+2] * B[j_off:j_off+2, k_off] -> C[i_off, k_off]
        # 2 terms
        for j_rel in range(2):
            u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
            u[idx_u(i_off, j_off+j_rel)] = 1
            v[idx_v(j_off+j_rel, k_off)] = 1
            w[idx_w(k_off, i_off)] = 1
            add_term(u, v, w)

    # Block 3b: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    add_2x2x1_block(0, 0, 4)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    # 4a: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2)
    # 4b: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    add_2x2x1_block(0, 2, 4)
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    add_1x2x2_block(2, 0, 0)
    
    # Block 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    add_1x2x2_block(2, 2, 0)
    
    # Block 7: A[2:3, 0:2] * B[0:2, 2:5] -> C[2:3, 2:5]
    # 7a: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    add_1x2x2_block(2, 0, 2)
    # 7b: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5]
    add_1x2x1_block(2, 0, 4)
    
    # Block 8: A[2:3, 2:4] * B[2:4, 2:5] -> C[2:3, 2:5]
    # 8a: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    add_1x2x2_block(2, 2, 2)
    # 8b: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5]
    add_1x2x1_block(2, 2, 4)
    
    rank = len(U)
    
    # Convert to numpy arrays
    U_arr = np.array(U).T
    V_arr = np.array(V).T
    W_arr = np.array(W).T
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}
# EVOLVE-BLOCK-END
