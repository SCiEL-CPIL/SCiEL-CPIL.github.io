import time
import sys

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
    U index: 2*i + j
    V index: 2*j + k
    W index: 2*k + i
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mq into rank 1 terms?
    # No, Mq is a 2x2 multiplication itself.
    # Wait, the 2x2 Strassen algorithm replaces 8 multiplications with 7.
    # Each "multiplication" is a scalar multiplication.
    # So the rank is 7.
    # The tensors are:
    # For each of the 7 M's, we have a tensor.
    # But M1 is (A11+A22)(B11+B22). This is not a single scalar multiplication.
    # It is a product of two sums.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This has rank 4?
    # No, Strassen's algorithm works by defining 7 new matrices M1..M7.
    # M1 = (A11+A22)(B11+B22)
    # The multiplication of (A11+A22) and (B11+B22) is a single multiplication in the "extended" field?
    # No, A11, A22 etc are scalars in the 2x2 case.
    # So M1 is a scalar product of two linear combinations.
    # So M1 is a rank 1 tensor?
    # Let's check.
    # M1 = (A00 + A11)(B00 + B11).
    # This is not a rank 1 tensor in A,B.
    # (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This is a sum of 4 rank 1 tensors.
    # So M1 contributes 4 to the rank?
    # If so, total rank would be 7 * 4 = 28?
    # But Strassen's rank for 2x2 is 7.
    # Why?
    # Because the "multiplications" M1..M7 are the rank 1 components.
    # In the 2x2 case, the inputs A, B are scalars.
    # M1 = (A00+A11)(B00+B11).
    # This is a product of two linear forms.
    # A linear form in A is $u \cdot A$.
    # A linear form in B is $v \cdot B$.
    # The product is $(u \cdot A)(v \cdot B)$.
    # This is a rank 1 tensor in the space of A and B?
    # The tensor is $T_{ijk}$.
    # Here $i,j,k$ are indices of A, B, C.
    # For 2x2, $i,j,k \in \{0,1\}$.
    # $M1$ corresponds to the term where $A$ is replaced by $A00+A11$ and $B$ by $B00+B11$.
    # The contribution to $C$ is added to $C11, C12, C21, C22$ with coefficients.
    # Wait, $M1$ is a scalar.
    # $C11 = M1 + ...$
    # So $M1$ is a scalar value.
    # $M1 = (A00+A11)(B00+B11)$.
    # This scalar is formed by multiplying two linear combinations.
    # Is this a rank 1 tensor?
    # A rank 1 tensor is $u_i v_j w_k$.
    # Here, the "multiplication" is just the product of two numbers.
    # The tensor entry $T_{ijk}$ is 1 if $i,j,k$ match the indices in the product.
    # For $M1$, the product is $(A00+A11)(B00+B11)$.
    # This expands to $A00B00 + A00B11 + A11B00 + A11B11$.
    # So $M1$ is a sum of 4 rank 1 tensors.
    # So the rank of $M1$ is 4.
    # Then total rank is $4+1+1+1+1+4+4 = 16$?
    # No, Strassen's rank is 7.
    # The trick is that we are allowed to use linear combinations in the intermediate steps.
    # The decomposition is $C = \sum_{r=1}^7 M_r \cdot W_r$.
    # Where $M_r$ is a scalar computed as $U_r \cdot A \cdot V_r \cdot B$?
    # No.
    # The standard Strassen decomposition is:
    # $C = \sum_{r=1}^7 M_r \otimes W_r$.
    # Where $M_r$ is the scalar result of the r-th multiplication.
    # $M_r = L_r(A) \cdot R_r(B)$.
    # $L_r(A)$ is a linear combination of entries of A.
    # $R_r(B)$ is a linear combination of entries of B.
    # So $M_r$ is a rank 1 tensor in the space of A and B.
    # $L_r(A) = \sum_{i,j} u_{r,ij} A_{ij}$.
    # $R_r(B) = \sum_{j,k} v_{r,jk} B_{jk}$.
    # $M_r = (\sum u A) (\sum v B) = \sum_{i,j,j',k} u_{ij} v_{j'k} A_{ij} B_{j'k}$.
    # This is a sum of rank 1 tensors $A_{ij} B_{j'k}$.
    # But we treat $M_r$ as a single "multiplication" in the algorithmic sense.
    # In the tensor decomposition sense, the rank is the number of such $M_r$'s.
    # Because $C_{ik} = \sum_r M_r W_{r,ik}$.
    # And $M_r$ is bilinear in A and B.
    # So the tensor is $\sum_r (L_r \otimes R_r) \otimes W_r$.
    # This is a sum of 7 terms, each of which is a product of a linear form in A, a linear form in B, and a constant in C (or linear form in C?).
    # $W_r$ is a matrix.
    # So the rank is 7.
    
    # So for the 2x2 case, we just need to define $u_{r,ij}$, $v_{r,jk}$, $w_{r,ik}$.
    
    # r=1: M1 = (A00+A11)(B00+B11)
    # u: A00=1, A11=1. (Indices 0, 2)
    # v: B00=1, B11=1. (Indices 0, 2)
    # w: C11=1, C12=0, C21=0, C22=1?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So for r=1, w is [1, 0, 0, 1] for C11, C12, C21, C22.
    # W index: k*n+i.
    # C11: k=0, i=0 -> 0
    # C12: k=1, i=0 -> 2
    # C21: k=0, i=1 -> 1
    # C22: k=1, i=1 -> 3
    
    # Let's fill U, V, W for 2x2.
    
    # r=1
    U[0][0] = 1 # A00
    U[2][0] = 1 # A11
    V[0][0] = 1 # B00
    V[2][0] = 1 # B11
    W[0][0] = 1 # C11
    W[3][0] = 1 # C22
    
    # r=2: M2 = (A10+A11)B00
    U[1][1] = 1 # A10 (i=1, j=0 -> idx 2*1+0=2? No. i=1, j=0. U idx 2*1+0=2? No. i=1, j=0. 2*1+0=2. A10 is i=1, j=0.
    # Wait, A indices: i row, j col.
    # A00: i=0, j=0. idx 0.
    # A01: i=0, j=1. idx 1.
    # A10: i=1, j=0. idx 2.
    # A11: i=1, j=1. idx 3.
    
    # M2 = (A10+A11)B00
    U[2][1] = 1 # A10
    U[3][1] = 1 # A11
    V[0][1] = 1 # B00
    W[1][1] = 1 # C21 (k=0, i=1 -> 1)
    W[3][1] = -1 # C22 (k=1, i=1 -> 3)
    
    # r=3: M3 = A00(B01-B11)
    U[0][2] = 1 # A00
    V[1][2] = 1 # B01
    V[3][2] = -1 # B11
    W[2][2] = 1 # C12 (k=1, i=0 -> 2)
    W[3][2] = 1 # C22
    
    # r=4: M4 = A11(B10-B00)
    U[3][3] = 1 # A11
    V[2][3] = 1 # B10
    V[0][3] = -1 # B00
    W[0][3] = 1 # C11
    W[1][3] = 1 # C21
    
    # r=5: M5 = (A00+A01)B11
    U[0][4] = 1 # A00
    U[1][4] = 1 # A01
    V[3][4] = 1 # B11
    W[0][4] = -1 # C11
    W[2][4] = 1 # C12
    
    # r=6: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1 # A10
    U[0][5] = -1 # A00
    V[0][5] = 1 # B00
    V[1][5] = 1 # B01
    W[3][5] = 1 # C22
    
    # r=7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1 # A01
    U[3][6] = -1 # A11
    V[2][6] = 1 # B10
    V[3][6] = 1 # B11
    W[0][6] = 1 # C11
    W[2][6] = 1 # C12
    
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm.
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2()
    
    # Define the 7 meta-multiplications
    # Each is a 2x2 multiplication of linear combinations of 2x2 blocks.
    # We define alpha[I,J] and beta[J,K] for each q.
    # I, J, K are block indices (0,1).
    
    # q=1: M1 = (A11+A22)(B11+B22)
    # X = A11+A22. Y = B11+B22.
    # alpha[0,0]=1, alpha[1,1]=1.
    # beta[0,0]=1, beta[1,1]=1.
    
    # q=2: M2 = (A21+A22)B11
    # X = A21+A22. Y = B11.
    # alpha[1,0]=1, alpha[1,1]=1.
    # beta[0,0]=1.
    
    # q=3: M3 = A11(B12-B22)
    # X = A11. Y = B12-B22.
    # alpha[0,0]=1.
    # beta[0,1]=1, beta[1,1]=-1.
    
    # q=4: M4 = A22(B21-B11)
    # X = A22. Y = B21-B11.
    # alpha[1,1]=1.
    # beta[1,0]=1, beta[0,0]=-1.
    
    # q=5: M5 = (A11+A12)B22
    # X = A11+A12. Y = B22.
    # alpha[0,0]=1, alpha[0,1]=1.
    # beta[1,1]=1.
    
    # q=6: M6 = (A21-A11)(B11+B12)
    # X = A21-A11. Y = B11+B12.
    # alpha[1,0]=1, alpha[0,0]=-1.
    # beta[0,0]=1, beta[0,1]=1.
    
    # q=7: M7 = (A12-A22)(B21+B22)
    # X = A12-A22. Y = B21+B22.
    # alpha[0,1]=1, alpha[1,1]=-1.
    # beta[1,0]=1, beta[1,1]=1.
    
    configs = [
        # (alpha, beta)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # q1
        ({(1,0):1, (1,1):1}, {(0,0):1}),           # q2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),          # q3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),          # q4
        ({(0,0):1, (0,1):1}, {(1,1):1}),           # q5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}), # q6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1})  # q7
    ]
    
    for q, (alpha, beta) in enumerate(configs):
        for r in range(7):
            col = q * 7 + r
            # Iterate over local indices i', j', k'
            for ip in range(2):
                for jp in range(2):
                    for kp in range(2):
                        # Get values from 2x2 Strassen
                        u_val = U2[2*ip + jp][r]
                        v_val = V2[2*jp + kp][r]
                        w_val = W2[2*kp + ip][r]
                        
                        if u_val == 0 or v_val == 0 or w_val == 0:
                            continue
                            
                        # Determine global indices
                        # i = 2*I + ip, j = 2*J + jp, k = 2*K + kp
                        # U index: i*4 + j
                        # V index: j*4 + k
                        # W index: k*4 + i
                        
                        # We need to sum over I, J, K where alpha[I,J] and beta[J,K] are non-zero.
                        # Note: J in alpha must match J in beta.
                        
                        for I in range(2):
                            for J in range(2):
                                if (I, J) not in alpha:
                                    continue
                                a_val = alpha[(I, J)]
                                for K in range(2):
                                    if (J, K) not in beta:
                                        continue
                                    b_val = beta[(J, K)]
                                    
                                    i = 2*I + ip
                                    j = 2*J + jp
                                    k = 2*K + kp
                                    
                                    u_idx = i * 4 + j
                                    v_idx = j * 4 + k
                                    w_idx = k * 4 + i
                                    
                                    U[u_idx][col] += u_val * a_val
                                    V[v_idx][col] += v_val * b_val
                                    W[w_idx][col] += w_val
                                    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Get 4x4x4 decomposition
    U44, V44, W44 = _strassen_4x4()
    
    rank_44 = 49
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(32)]
    W = [[0] * total_rank for _ in range(32)]
    
    # First half: B columns 0-3
    for i in range(16):
        for r in range(rank_44):
            U[i][r] = U44[i][r]
            
    for j in range(4):
        for k in range(4):
            v_idx_44 = j * 4 + k
            v_idx_8 = j * 8 + k
            for r in range(rank_44):
                V[v_idx_8][r] = V44[v_idx_44][r]
                
    for k in range(4):
        for i in range(4):
            w_idx_44 = k * 4 + i
            w_idx_8 = k * 4 + i
            for r in range(rank_44):
                W[w_idx_8][r] = W44[w_idx_44][r]
                
    # Second half: B columns 4-7
    offset = rank_44
    for i in range(16):
        for r in range(rank_44):
            U[i][offset + r] = U44[i][r]
            
    for j in range(4):
        for k in range(4):
            v_idx_44 = j * 4 + k
            v_idx_8 = j * 8 + (4 + k)
            for r in range(rank_44):
                V[v_idx_8][offset + r] = V44[v_idx_44][r]
                
    for k in range(4):
        for i in range(4):
            w_idx_44 = k * 4 + i
            w_idx_8 = (4 + k) * 4 + i
            for r in range(rank_44):
                W[w_idx_8][offset + r] = W44[w_idx_44][r]
                
    if time.time() > deadline:
        # Fallback to schoolbook if too slow (unlikely)
        pass
        
    return {"U": U, "V": V, "W": W}

