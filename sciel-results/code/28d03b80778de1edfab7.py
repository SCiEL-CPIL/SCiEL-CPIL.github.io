def construct():
    n, m, p = 3, 5, 5
    # We will build the decomposition with rank 73.
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    # We'll use a list of tuples (U_dict, V_dict, W_dict) for each term
    # then combine them.
    
    terms = []
    
    # Helper to add a schoolbook term for A[i1:i2, j1:j2] * B[j1:j2, k1:k2]
    # This adds (i2-i1)*(j2-j1)*(k2-k1) terms? No, rank is (i2-i1)*(k2-k1) if we optimize?
    # No, schoolbook rank for m x n x p is m*n*p.
    # For A[i1:i2, j1:j2] and B[j1:j2, k1:k2], the rank is (i2-i1)*(j2-j1)*(k2-k1).
    def add_schoolbook(i1, i2, j1, j2, k1, k2):
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    u = {i*m + j: 1.0}
                    v = {j*p + k: 1.0}
                    w = {k*n + i: 1.0}
                    terms.append((u, v, w))
                    
    # Helper to add a Strassen term for 2x2 block at A[i1, j1] and B[j1, k1]
    # A block: rows i1, i1+1; cols j1, j1+1
    # B block: rows j1, j1+1; cols k1, k1+1
    # C block: rows i1, i1+1; cols k1, k1+1
    def add_strassen(i1, j1, k1):
        # M1 = (A11 + A22)(B11 + B22)
        u1 = {i1*m + j1: 1.0, (i1+1)*m + (j1+1): 1.0}
        v1 = {j1*p + k1: 1.0, (j1+1)*p + (k1+1): 1.0}
        w1 = {k1*n + i1: 1.0, (k1+1)*n + (i1+1): 1.0}
        terms.append((u1, v1, w1))
        
        # M2 = (A21 + A22)B11
        u2 = {(i1+1)*m + j1: 1.0, (i1+1)*m + (j1+1): 1.0}
        v2 = {j1*p + k1: 1.0}
        w2 = {(k1+1)*n + (i1+1): 1.0} # C21 and C22? 
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # So M2 contributes to C21 (+1) and C22 (-1).
        w2 = {(k1+1)*n + (i1+1): 1.0, (k1+1)*n + (i1+1) + 1: -1.0} # Wait, indices
        # C indices:
        # C[i1, k1] -> k1*n + i1
        # C[i1, k1+1] -> (k1+1)*n + i1
        # C[i1+1, k1] -> k1*n + i1+1
        # C[i1+1, k1+1] -> (k1+1)*n + i1+1
        
        # Let's re-verify Strassen formulas and indices.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1: C11(+1), C22(+1)
        # M2: C21(+1), C22(-1)
        # M3: C12(+1), C22(+1)
        # M4: C11(+1), C21(+1)
        # M5: C11(-1), C12(+1)
        # M6: C22(+1)
        # M7: C11(+1)
        
        # My w1 above was correct for M1.
        
        # Correcting M2
        u2 = {(i1+1)*m + j1: 1.0, (i1+1)*m + (j1+1): 1.0}
        v2 = {j1*p + k1: 1.0}
        w2 = {k1*n + i1+1: 1.0, (k1+1)*n + i1+1: -1.0}
        terms.append((u2, v2, w2))
        
        # M3 = A11(B12 - B22)
        u3 = {i1*m + j1: 1.0}
        v3 = {j1*p + k1+1: 1.0, (j1+1)*p + (k1+1): -1.0}
        w3 = {(k1+1)*n + i1: 1.0, (k1+1)*n + i1+1: 1.0}
        terms.append((u3, v3, w3))
        
        # M4 = A22(B21 - B11)
        u4 = {(i1+1)*m + (j1+1): 1.0}
        v4 = {(j1+1)*p + k1: 1.0, j1*p + k1: -1.0}
        w4 = {k1*n + i1: 1.0, k1*n + i1+1: 1.0}
        terms.append((u4, v4, w4))
        
        # M5 = (A11 + A12)B22
        u5 = {i1*m + j1: 1.0, i1*m + j1+1: 1.0}
        v5 = {(j1+1)*p + (k1+1): 1.0}
        w5 = {k1*n + i1: -1.0, (k1+1)*n + i1: 1.0}
        terms.append((u5, v5, w5))
        
        # M6 = (A21 - A11)(B11 + B12)
        u6 = {(i1+1)*m + j1: 1.0, i1*m + j1: -1.0}
        v6 = {j1*p + k1: 1.0, j1*p + k1+1: 1.0}
        w6 = {(k1+1)*n + i1+1: 1.0}
        terms.append((u6, v6, w6))
        
        # M7 = (A12 - A22)(B21 + B22)
        u7 = {i1*m + j1+1: 1.0, (i1+1)*m + (j1+1): -1.0}
        v7 = {(j1+1)*p + k1: 1.0, (j1+1)*p + (k1+1): 1.0}
        w7 = {k1*n + i1: 1.0}
        terms.append((u7, v7, w7))

    # 1. A3 B3: A[2:3, 4:5] * B[4:5, 0:5]
    # This is 1x1x5. Schoolbook rank 5?
    # My previous analysis said 15. Let's re-verify.
    # C[i,k] = A[i,4] B[4,k].
    # i in 0..2, k in 0..4.
    # Term for each (i,k) is A[i,4] B[4,k].
    # This requires 15 multiplications if we don't share.
    # Can we share?
    # C[:, k] = A[:,4] B[4,k].
    # For fixed k, C[:,k] is a vector.
    # We can write C = sum_k (A[:,4] * B[4,k]) * E_{:,k}.
    # Each term is a product of a vector and a scalar?
    # In the tensor decomposition, we need scalar products.
    # $U_{ij} V_{jk} W_{ki}$.
    # For $A_{i,4} B_{4,k}$, we can set $U_{i,4}=1, V_{4,k}=1, W_{k,i}=1$ for each pair?
    # Yes, that's 15 terms.
    # Is there a rank 5 decomposition?
    # Rank of $1 \times 1 \times 5$ tensor is 5?
    # $T_{i,j,k}$. Here $j=4$ fixed. $T_{i,4,k} = 1$.
    # This is a $3 \times 5$ matrix of 1s.
    # The bilinear rank of a matrix $M$ is its matrix rank? No.
    # The bilinear rank of the multiplication tensor $M_{m,n,p}$ is the rank.
    # For $1 \times 1 \times p$, the rank is $p$?
    # $C_k = A B_k$.
    # $C = A B$.
    # If $A$ is $1 \times 1$, $B$ is $1 \times p$.
    # $C_k = A B_k$.
    # We need $p$ multiplications $A B_k$.
    # So rank is $p=5$.
    # Wait, $A$ is $3 \times 1$.
    # $C_{i,k} = A_i B_k$.
    # This is the outer product of $A$ and $B^T$.
    # The matrix $C$ has rank 1.
    # But the bilinear rank is the number of scalar multiplications.
    # $C_{i,k} = A_i B_k$.
    # We can compute $A_i B_k$ for all $i,k$.
    # If we compute $A_i B_0, A_i B_1, \dots$
    # We need $3 \times 5 = 15$ products.
    # Can we do it in 5?
    # Let $r=0 \dots 4$.
    # $C_{i,k} = \sum_r U_{i,4}^{(r)} V_{4,k}^{(r)} W_{k,i}^{(r)}$.
    # If we set $W_{k,i}^{(r)} = \delta_{k,r}$, then $C_{i,r} = \sum_r U_{i,4}^{(r)} V_{4,r}^{(r)} \delta_{k,r}$.
    # $C_{i,r} = U_{i,4}^{(r)} V_{4,r}^{(r)}$.
    # We want $C_{i,r} = A_i B_r$.
    # So we need $U_{i,4}^{(r)} V_{4,r}^{(r)} = A_i B_r$.
    # We can set $U_{i,4}^{(r)} = A_i$ and $V_{4,r}^{(r)} = B_r$.
    # But $U$ and $V$ must be constant coefficients (0, 1, -1/2 etc), not dependent on input values.
    # The decomposition is $\sum_r U^{(r)} \otimes V^{(r)} \otimes W^{(r)}$.
    # The coefficients $U_{ij}^{(r)}$ are fixed.
    # So we cannot set $U = A_i$.
    # We must use the structure of the tensor.
    # The tensor for $1 \times 1 \times 1$ is rank 1.
    # The tensor for $1 \times 1 \times 2$ is rank 2.
    # The tensor for $3 \times 1 \times 5$ is rank 15.
    
    # So $A_3 B_3$ costs 15.
    
    # My previous calculation: 15 + 29 + 29 = 73.
    
    # Let's add the terms.
    
    # Term 3: A[2:3, 4:5] * B[4:5, 0:5]
    add_schoolbook(2, 3, 4, 5, 0, 5)
    
    # Term 1: A[0:3, 0:2] * B[0:2, 0:5]
    # Split A into A[0:2, 0:2] and A[2:3, 0:2]
    # Split B into B[0:2, 0:2] and B[0:2, 2:5]
    
    # 1a: A[0:2, 0:2] * B[0:2, 0:2] -> Strassen
    add_strassen(0, 0, 0)
    
    # 1b: A[0:2, 0:2] * B[0:2, 2:5] -> Schoolbook
    # A is 2x2, B is 2x3. Rank 12.
    add_schoolbook(0, 2, 0, 2, 2, 5)
    
    # 1c: A[2:3, 0:2] * B[0:2, 0:2] -> Schoolbook
    # A is 1x2, B is 2x2. Rank 4.
    add_schoolbook(2, 3, 0, 2, 0, 2)
    
    # 1d: A[2:3, 0:2] * B[0:2, 2:5] -> Schoolbook
    # A is 1x2, B is 2x3. Rank 6.
    add_schoolbook(2, 3, 0, 2, 2, 5)
    
    # Term 2: A[0:3, 2:4] * B[2:4, 0:5]
    # Split A into A[0:2, 2:4] and A[2:3, 2:4]
    # Split B into B[2:4, 0:2] and B[2:4, 2:5]
    
    # 2a: A[0:2, 2:4] * B[2:4, 0:2] -> Strassen
    # A block at i=0, j=2. B block at j=2, k=0.
    add_strassen(0, 2, 0)
    
    # 2b: A[0:2, 2:4] * B[2:4, 2:5] -> Schoolbook
    # A is 2x2, B is 2x3. Rank 12.
    add_schoolbook(0, 2, 2, 4, 2, 5)
    
    # 2c: A[2:3, 2:4] * B[2:4, 0:2] -> Schoolbook
    # A is 1x2, B is 2x2. Rank 4.
    add_schoolbook(2, 3, 2, 4, 0, 2)
    
    # 2d: A[2:3, 2:4] * B[2:4, 2:5] -> Schoolbook
    # A is 1x2, B is 2x3. Rank 6.
    add_schoolbook(2, 3, 2, 4, 2, 5)
    
    # Total rank: 15 + 7 + 12 + 4 + 6 + 7 + 12 + 4 + 6 = 73.
    
    # Now construct U, V, W
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][r] += val
        for idx, val in v_dict.items():
            V[idx][r] += val
        for idx, val in w_dict.items():
            W[idx][r] += val
            
    return {"U": U, "V": V, "W": W}
