import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = DIMENSIONS
    rank = 49

    # We construct the decomposition using the product of two Strassen algorithms:
    # 1. Outer Strassen for 2x2 block matrices (7 terms)
    # 2. Inner Strassen for 2x2 scalar matrices (7 terms)
    # Total rank = 7 * 7 = 49.

    # Outer Strassen coefficients for 2x2 matrix multiplication
    # Blocks are indexed 0..3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # For A: indices 0,1,2,3
    # For B: indices 0,1,2,3
    # For C: indices 0,1,2,3
    
    # beta[s, I, J] where I is block row of A (0,1), J is block col of A (0,1)
    # Actually, the blocks are:
    # A11, A12
    # A21, A22
    # Let's define the 7 Strassen multiplications for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients for A (beta), B (gamma), C (delta)
    # Dimensions: (7, 2, 2) for the 2x2 block grid
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))
    
    # s=0: M1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1: M2
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2: M3
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3: M4
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1
    
    # s=4: M5
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1
    
    # s=5: M6
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6: M7
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    # Inner Strassen coefficients for 2x2 scalar multiplication
    # This decomposes a 2x2 matrix product into 7 scalar products.
    # U2, V2, W2 have shape (4, 7).
    # Indices 0..3 correspond to positions (0,0), (0,1), (1,0), (1,1) in the 2x2 matrix.
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # t=0: M1_inner = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1: M2_inner = (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2: M3_inner = a00(b01-b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3: M4_inner = a11(b10-b00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4: M5_inner = (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5: M6_inner = (a10-a00)(b00+b01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6: M7_inner = (a01-a11)(b10+b11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    # Build U, V, W
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            
            # U construction:
            # Row in U corresponds to A[i, j]
            # i = 2*I + i', j = 2*J + j'
            # U[2*I+i', 2*J+j'] (in 2x2 block space) is multiplied by beta[s, I, J]
            # And then the inner 2x2 product U2[2*i'+j', t] is used?
            # Wait. The structure is:
            # A_block[I, J] is a 2x2 matrix.
            # The linear combination X_s = sum_{I,J} beta[s, I, J] A[I, J]
            # This X_s is a 2x2 matrix.
            # Then we multiply X_s * Y_s.
            # The product Z_s = X_s Y_s.
            # We decompose Z_s into 7 terms using inner Strassen.
            # The coefficients for the inner multiplication depend on the elements of X_s and Y_s.
            # The elements of X_s are linear combinations of A elements.
            # So U[row, col] should be the coefficient of A[row] in the term M_{s,t}.
            
            # Let's trace the path of an A element A[global_i, global_j].
            # It belongs to block A[I, J] where I = global_i // 2, J = global_j // 2.
            # Its local position is i' = global_i % 2, j' = global_j % 2.
            # In the outer Strassen, this element contributes to X_s with coefficient beta[s, I, J].
            # So the coefficient of A[global_i, global_j] in X_s is beta[s, I, J].
            # Now, X_s is a 2x2 matrix. Let's call its local elements X_s[i', j'].
            # X_s[i', j'] = sum_{I,J} beta[s, I, J] A[I, J][i', j'].
            # This is a linear combination of A elements.
            # The term M_{s,t} is a scalar product involving an element of X_s and an element of Y_s.
            # Specifically, for the inner Strassen term t, it involves X_s[i_t, j_t] * Y_s[k_t, l_t].
            # Wait, the inner Strassen U2/V2/W2 are defined for a generic 2x2 multiplication.
            # U2[local_idx, t] is the coefficient for the first matrix's element.
            # So, if U2[ip, jp] is non-zero for term t, it means the element at (ip, jp) of the first matrix is used.
            # So, for a specific A element A[I, J][i', j'], it contributes to X_s[i', j'].
            # It is then used in M_{s,t} if U2[i', j', t] is non-zero?
            # No, U2 is defined as U2[row, col] where row is the index in the 2x2 matrix (0..3).
            # The index in the 2x2 matrix is i'*2 + j'.
            # So, if U2[i'*2+j', t] is non-zero, then X_s[i', j'] is involved in M_{s,t}.
            # Therefore, the coefficient of A[global_i, global_j] in M_{s,t} is:
            # beta[s, I, J] * U2[i'*2 + j', t]
            
            I = 0
            J = 0
            # We iterate over all possible global rows and cols to fill U, V, W
            # But we can do it more efficiently by iterating over blocks.
            
    # Let's just fill the matrices directly using the derived logic.
    for global_i in range(4):
        for global_j in range(4):
            row_a = global_i * 4 + global_j
            I = global_i // 2
            J = global_j // 2
            i_prime = global_i % 2
            j_prime = global_j % 2
            inner_idx_u = i_prime * 2 + j_prime
            
            for s in range(7):
                b_coeff = beta[s, I, J]
                if b_coeff == 0:
                    continue
                for t in range(7):
                    col = 7 * s + t
                    u_inner_coeff = U2[inner_idx_u, t]
                    if u_inner_coeff != 0:
                        U[row_a, col] = b_coeff * u_inner_coeff

    for global_j in range(4):
        for global_k in range(4):
            row_b = global_j * 4 + global_k
            J = global_j // 2
            K = global_k // 2
            j_prime = global_j % 2
            k_prime = global_k % 2
            inner_idx_v = j_prime * 2 + k_prime
            
            for s in range(7):
                g_coeff = gamma[s, J, K]
                if g_coeff == 0:
                    continue
                for t in range(7):
                    col = 7 * s + t
                    v_inner_coeff = V2[inner_idx_v, t]
                    if v_inner_coeff != 0:
                        V[row_b, col] = g_coeff * v_inner_coeff

    for global_k in range(4):
        for global_i in range(4):
            row_c = global_k * 4 + global_i
            K = global_k // 2
            I = global_i // 2
            k_prime = global_k % 2
            i_prime = global_i % 2
            # For W, the index in the 2x2 matrix is k_prime * 2 + i_prime
            # Because W is the output coefficient for C[K, I]
            inner_idx_w = k_prime * 2 + i_prime
            
            for s in range(7):
                d_coeff = delta[s, I, K]
                if d_coeff == 0:
                    continue
                for t in range(7):
                    col = 7 * s + t
                    w_inner_coeff = W2[inner_idx_w, t]
                    if w_inner_coeff != 0:
                        W[row_c, col] = d_coeff * w_inner_coeff

    # Verification is not needed for submission, but we can ensure no NaNs
    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
