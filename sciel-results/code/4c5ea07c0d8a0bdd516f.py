"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U, V, W for M_2
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Rank 7
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    # a=0,0 b=0,1
    # c=1,0 d=1,1
    # e=0,0 f=0,1
    # g=1,0 h=1,1
    
    # U2 (A part)
    # U2[0,0]=1 (a), U2[3,0]=1 (d)
    # U2[1,1]=1 (c), U2[3,1]=1 (d) -> Wait, M2 is (c+d)e. U2[1,1]=1, U2[3,1]=1.
    # U2[0,2]=1 (a)
    # U2[3,3]=1 (d)
    # U2[0,4]=1 (a), U2[1,4]=1 (b)
    # U2[1,5]=1 (c), U2[0,5]=-1 (a)
    # U2[1,6]=1 (b), U2[3,6]=-1 (d)
    
    # Let's define Strassen terms explicitly.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = AB
    
    # Term 1: M1 = (a+d)(e+h)
    # U2[0, 0] = 1; U2[3, 0] = 1  # a, d
    # V2[0, 0] = 1; V2[3, 0] = 1  # e, h
    # W2[0, 0] = 1; W2[1, 0] = -1; W2[2, 0] = -1; W2[3, 0] = 1 # C11, C12, C21, C22?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's assign columns 0..6 to M1..M7.
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C11 gets M1, C22 gets M1
    
    # M2: (c+d)e
    U2[1, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C21 gets M2, C22 gets -M2
    
    # M3: a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C12 gets M3, C22 gets M3
    
    # M4: d(g-f)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C11 gets M4, C21 gets M4
    
    # M5: (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C11 gets -M5, C12 gets M5
    
    # M6: (c-a)(e+f)
    U2[1, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C22 gets M6
    
    # M7: (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11 gets M7
    
    # Verify Strassen:
    # C11 = M1 - M5 + M4 + M7. W2[0, :] is [1, 0, 0, 1, -1, 0, 1]. Correct.
    # C12 = M3 + M5. W2[1, :] is [0, 0, 1, 0, 1, 0, 0]. Correct.
    # C21 = M2 + M4. W2[2, :] is [0, 1, 0, 1, 0, 0, 0]. Correct.
    # C22 = M1 - M2 + M3 + M6. W2[3, :] is [1, -1, 1, 0, 0, 1, 0]. Correct.

    # Schoolbook decomposition for 3x3 matrices
    # U3, V3, W3 for M_3
    # Rank 27
    U3 = np.zeros((9, 27), dtype=float)
    V3 = np.zeros((9, 27), dtype=float)
    W3 = np.zeros((9, 27), dtype=float)
    
    col3 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col3] = 1
                V3[j * 3 + k, col3] = 1
                W3[k * 3 + i, col3] = 1
                col3 += 1

    # Combine M_3 and M_2
    # Total rank = 27 * 7 = 189
    R = 27 * 7
    U6 = np.zeros((36, R), dtype=float)
    V6 = np.zeros((36, R), dtype=float)
    W6 = np.zeros((36, R), dtype=float)
    
    # Map indices
    # Global I = 2*i + alpha
    # Global J = 2*j + beta
    # Global K = 2*k + gamma
    
    # For each block in M3 (i, j, k) and each term in Strassen (l, r)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                l3 = k * 3 + i # Column in U3/V3/W3
                # Actually, U3 index is i*3+j. V3 index is j*3+k. W3 index is k*3+i.
                # Let's use the loop variable col3 which increments.
                # We need to map col3 to the specific i, j, k.
                # It's easier to iterate l3 from 0 to 26 and derive i, j, k.
                
    # Let's redo the loop structure for clarity.
    for l3 in range(27):
        # Derive i, j, k from l3
        # l3 = (i*3 + j)*3 + k? No.
        # In the construction above:
        # i in 0..2, j in 0..2, k in 0..2.
        # col3 = i*9 + j*3 + k.
        i = l3 // 9
        rem = l3 % 9
        j = rem // 3
        k = rem % 3
        
        u3_val = 1 # U3[i*3+j, l3] is 1
        v3_val = 1 # V3[j*3+k, l3] is 1
        w3_val = 1 # W3[k*3+i, l3] is 1
        
        # Actually, we can just check if U3[i*3+j, l3] is non-zero.
        # Since it's schoolbook, it's 1 only if the indices match the block.
        # But we are iterating over all l3.
        # For a given l3, it corresponds to a unique (i,j,k).
        # U3[i*3+j, l3] is 1.
        # V3[j*3+k, l3] is 1.
        # W3[k*3+i, l3] is 1.
        
        for r2 in range(7):
            col6 = l3 * 7 + r2
            
            # U6
            for alpha in range(2):
                u2_val = U2[alpha * 2 + 0, r2] # Wait, U2 index is i*2+j.
                # For U2, the "i" index is the row of A.
                # In U2, row is 0..3.
                # We need to map the global I = 2*i + alpha to the U2 row.
                # But U2 is for a specific 2x2 block.
                # The block is determined by i, j, k.
                # The Strassen decomposition is applied to the multiplication of the block A_ij and B_jk.
                # So U2's "i" corresponds to the row of the 2x2 block A_ij.
                # Let's call the local row alpha.
                # U2[alpha * 2 + beta_local, r2]? No.
                # U2 is 4x7. Row index is 0..3.
                # Row 0: A[0,0] (a)
                # Row 1: A[0,1] (b)
                # Row 2: A[1,0] (c)
                # Row 3: A[1,1] (d)
                
                # The global row of U6 is I = 2*i + alpha.
                # But U6 also depends on j?
                # No, U6[I, col] sums over j?
                # In the tensor decomposition, U is a matrix.
                # The element U6[I, col] is the coefficient of the variable A[I, J] in the term?
                # No, the tensor decomposition is:
                # C_{K, I} = \sum_{J} \sum_{r} U_{I, J, r} V_{J, K, r} W_{K, I, r}?
                # The interface is:
                # U: (n*m, rank) -> U[I, J, r]? No, U is (n*m, rank).
                # The multiplication is:
                # C = \sum_r U_r \otimes V_r \otimes W_r?
                # Specifically:
                # C_{K, I} = \sum_{J} \sum_{r} U[I, J, r] * V[J, K, r] * W[K, I, r]?
                # The standard format is:
                # T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}.
                # Here, the tensor is M_{n, m, p}.
                # Indices:
                # U is (n*m, rank). Index (i*m + j, r).
                # V is (m*p, rank). Index (j*p + k, r).
                # W is (p*n, rank). Index (k*n + i, r).
                
                # So for our block decomposition:
                # Global I = 2*i + alpha.
                # Global J = 2*j + beta.
                # Global K = 2*k + gamma.
                
                # The term corresponds to l3 (block indices i, j, k) and r2 (Strassen term).
                # The contribution to U6[I, J, col6] comes from U3 and U2.
                # But U3 is (9, 27). U3[i*3+j, l3].
                # U2 is (4, 7). U2[alpha*2+beta, r2]?
                # No, U2's rows are indexed by (local_i, local_j).
                # local_i is the row of the 2x2 block A.
                # local_j is the column of the 2x2 block A.
                # So U2 row is local_i * 2 + local_j.
                # In our case, local_i = alpha, local_j = beta.
                # So U2 row is alpha * 2 + beta.
                
                # So U6[I, J, col6] += U3[i*3+j, l3] * U2[alpha*2+beta, r2].
                # But we are iterating over r2. We need to fill U6[I, J, col6].
                # I = 2*i + alpha.
                # J = 2*j + beta.
                
                # We can iterate alpha, beta.
                for alpha in range(2):
                    for beta in range(2):
                        I = 2 * i + alpha
                        J = 2 * j + beta
                        u3 = U3[i * 3 + j, l3]
                        u2 = U2[alpha * 2 + beta, r2]
                        if u3 != 0 and u2 != 0:
                            U6[I * 6 + J, col6] += u3 * u2
                        
                        V3_val = V3[j * 3 + k, l3]
                        v2 = V2[beta * 2 + gamma_placeholder, r2] # Need gamma
                        # V2 row is local_j * 2 + local_k.
                        # local_j = beta. local_k = gamma.
                        # We need to iterate gamma for V and W.
                        
    # This nested loop is getting messy. Let's vectorize or simplify.
    # Since U3, V3, W3 are sparse (one 1 per column), and U2, V2, W2 are sparse.
    # We can just iterate over the non-zero elements.
    
    # Let's rewrite the construction more cleanly.
    
    U6 = np.zeros((36, 189), dtype=float)
    V6 = np.zeros((36, 189), dtype=float)
    W6 = np.zeros((36, 189), dtype=float)
    
    # Get non-zero indices for U3, V3, W3
    # For schoolbook M3, U3[i*3+j, l3] is 1 if l3 = i*9+j*3+k.
    # So for a given l3, i = l3//9, j = (l3%9)//3, k = l3%3.
    # U3[i*3+j, l3] = 1.
    # V3[j*3+k, l3] = 1.
    # W3[k*3+i, l3] = 1.
    
    # Get non-zero indices for U2, V2, W2
    # U2 is 4x7.
    # We can just iterate over all 4*7 elements of U2.
    
    for l3 in range(27):
        i = l3 // 9
        rem = l3 % 9
        j = rem // 3
        k = rem % 3
        
        # U3 part
        u3_row = i * 3 + j
        v3_row = j * 3 + k
        w3_row = k * 3 + i
        
        # Iterate over Strassen terms
        for r2 in range(7):
            col6 = l3 * 7 + r2
            
            # U6
            for alpha in range(2):
                for beta in range(2):
                    I = 2 * i + alpha
                    J = 2 * j + beta
                    u2_val = U2[alpha * 2 + beta, r2]
                    if u2_val != 0:
                        U6[I * 6 + J, col6] = u2_val # u3 is 1
                        
            # V6
            for beta in range(2):
                for gamma in range(2):
                    J = 2 * j + beta
                    K = 2 * k + gamma
                    v2_val = V2[beta * 2 + gamma, r2]
                    if v2_val != 0:
                        V6[J * 6 + K, col6] = v2_val
                        
            # W6
            for gamma in range(2):
                for alpha in range(2):
                    K = 2 * k + gamma
                    I = 2 * i + alpha
                    w2_val = W2[gamma * 2 + alpha, r2]
                    if w2_val != 0:
                        W6[K * 6 + I, col6] = w2_val

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
