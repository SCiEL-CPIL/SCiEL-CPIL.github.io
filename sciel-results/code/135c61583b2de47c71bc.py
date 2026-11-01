import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    
    # Target Rank: 42
    # Construction: Kronecker Product of T_{2,2,2} (Strassen) and T_{1,2,3} (Standard)
    
    # 1. Define Strassen's decomposition for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Rank 7.
    # The tensors U, V, W are 4x7 matrices.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen's 7 terms (M1..M7)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We need to map these to the tensor coefficients T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: The problem statement uses W[k*n+i, r]. For n=2, index is k*2+i.
    # The tensor identity is: T[i,j,k] = delta_{i, i} delta_{j, j} delta_{k, k} ? 
    # Actually, the matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In the tensor decomposition context provided:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct the "multiplication map".
    # For the standard basis, the tensor T_{n,m,p} has T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Since every (i,j,k) contributes to exactly one C[i,k] with coeff 1, the target tensor is:
    # T[i,j,k] = 1 for all i,j,k? 
    # Wait, the "rank" of the all-ones tensor is 1. But matrix multiplication is not the all-ones tensor in the scalar field sense if we view it as a bilinear map to a vector space.
    # Let's stick to the Kronecker product property which is robust.
    # If T1 has rank r1 and T2 has rank r2, then T1 \otimes T2 has rank r1*r2.
    # The tensor for 2x2x2 matrix multiplication is M_2.
    # The tensor for 1x2x3 matrix multiplication is M_{1,2,3}.
    # The tensor for 2x4x6 is M_2 \otimes M_{1,2,3}.
    
    # Let's explicitly build M_{1,2,3} with rank 6 (trivial).
    # n1=1, m1=2, p1=3.
    # U1 shape (1*2, 6) = (2, 6)
    # V1 shape (2*3, 6) = (6, 6)
    # W1 shape (3*1, 6) = (3, 6)
    
    U1 = np.zeros((2, 6))
    V1 = np.zeros((6, 6))
    W1 = np.zeros((3, 6))
    
    col = 0
    for i1 in range(1):
        for j1 in range(2):
            for k1 in range(3):
                U1[i1*2+j1, col] = 1
                V1[j1*3+k1, col] = 1
                W1[k1*1+i1, col] = 1
                col += 1
                
    # Now we need the Strassen decomposition for M_2 (2x2x2).
    # We need U_s, V_s, W_s such that sum_r U_s[i,j,r] V_s[j,k,r] W_s[k,i,r] = 1 for all i,j,k?
    # No, that's not right. The tensor M_2 is defined by:
    # M_2 = sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{i,k}
    # This tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Wait, if T[i,j,k]=1 for all i,j,k, the rank is 1.
    # Why is Strassen rank 7?
    
    # The confusion lies in the definition of the tensor.
    # The tensor for matrix multiplication $M_{n,m,p}$ is an element of $\mathbb{R}^{nm} \otimes \mathbb{R}^{mp} \otimes \mathbb{R}^{pn}$.
    # It is defined as:
    # $M = \sum_{i=0}^{n-1} \sum_{j=0}^{m-1} \sum_{k=0}^{p-1} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$
    # where $e_{i,j}$ is the basis vector for the (i,j) entry of A, etc.
    # The rank of this tensor is the minimum number of rank-1 tensors needed to sum to it.
    # A rank-1 tensor is $u \otimes v \otimes w$.
    # The condition for a decomposition $\sum_{r=1}^R u_r \otimes v_r \otimes w_r = M$ is:
    # $\sum_{r=1}^R u_r[i,j] v_r[j,k] w_r[k,i] = 1$ for all valid i,j,k?
    # No, that would imply the tensor is the all-ones tensor, which has rank 1.
    
    # Let's re-read the standard definition of the matrix multiplication tensor.
    # $T_{i,j,k} = 1$ if the product of the i-th row of A and j-th column of B contributes to the k-th entry?
    # No.
    # The bilinear map is $\mu(A, B) = AB$.
    # $\mu(A, B) = \sum_{i,k} (\sum_j A_{i,j} B_{j,k}) E_{i,k}$.
    # This can be written as $\sum_{i,j,k} A_{i,j} B_{j,k} E_{i,k}$.
    # The tensor $T$ is defined by $T(e_{i,j}, e_{j',k}) = \delta_{j,j'} e_{i,k}$.
    # So $T = \sum_{i,j,k} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}$.
    # Here the indices are coupled.
    # For a fixed rank-1 term $u \otimes v \otimes w$:
    # Contribution to $T[i,j,k]$ is $u[i,j] v[j,k] w[k,i]$?
    # No, the tensor lives in $V^* \otimes W^* \otimes Z^*$.
    # The components are $T_{i,j,k}$.
    # The decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # This means $T_{i,j,k} = \sum_r u_r[i,j] v_r[j,k] w_r[k,i]$.
    # For the matrix multiplication tensor, $T_{i,j,k} = 1$ if the indices match the "flow" of multiplication?
    # Actually, the tensor $M_{n,m,p}$ has $T_{i,j,k} = 1$ for ALL $i,j,k$?
    # If $T_{i,j,k}=1$ for all $i,j,k$, then $u_r[i,j]=1, v_r[j,k]=1, w_r[k,i]=1$ works for R=1.
    # But we know rank is > 1.
    
    # The error is in the index mapping of the third factor.
    # The output is $C$. The basis is $E_{i,k}$.
    # The tensor is $T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$.
    # This is a sum of $nmp$ terms.
    # Is this sum rank 1?
    # Let's check $n=1, m=1, p=1$. $T = E_{0,0} \otimes E_{0,0} \otimes E_{0,0}$. Rank 1.
    # Let's check $n=2, m=1, p=1$.
    # $T = E_{0,0} \otimes E_{0,0} \otimes E_{0,0} + E_{1,0} \otimes E_{0,0} \otimes E_{1,0}$.
    # Can this be rank 1?
    # $u \otimes v \otimes w$.
    # $u[0,0]=1, v[0,0]=1, w[0,0]=1 \implies T[0,0,0]=1$.
    # $u[1,0]=1, v[0,0]=1, w[1,0]=1 \implies T[1,0,0]=1$.
    # But $v[0,0]$ is shared.
    # If $u[0,0]=1, u[1,0]=1$, then $v[0,0]=1$.
    # Then $w[0,0]$ must be 1 and $w[1,0]$ must be 1.
    # Then $T[0,0,0] = u[0,0]v[0,0]w[0,0] = 1$.
    # $T[1,0,0] = u[1,0]v[0,0]w[1,0] = 1$.
    # This works?
    # Wait, the tensor is $T = \sum_{i} e_{i,0} \otimes e_{0,0} \otimes e_{i,0}$.
    # This is NOT $u \otimes v \otimes w$ because $u$ and $w$ are coupled on index $i$, but $v$ is constant.
    # A rank-1 tensor $u \otimes v \otimes w$ has $T[i,j,k] = u[i,j] v[j,k] w[k,i]$.
    # In the case $n=2, m=1, p=1$:
    # $T[i,0,0] = u[i,0] v[0,0] w[0,i]$.
    # We want $T[i,0,0] = 1$ for $i=0,1$.
    # $u[0,0] v[0,0] w[0,0] = 1$
    # $u[1,0] v[0,0] w[0,1] = 1$
    # This is solvable with rank 1?
    # If $v[0,0]=1, u[0,0]=1, w[0,0]=1, u[1,0]=1, w[0,1]=1$.
    # Then $T[0,0,0]=1, T[1,0,0]=1$.
    # So rank is 1 for $2 \times 1 \times 1$?
    # Yes, because it's just scaling.
    
    # So when does rank increase?
    # It increases when the "pattern" of 1s cannot be factored.
    # For $2 \times 2 \times 2$:
    # $T[i,j,k] = 1$ for all $i,j,k$?
    # If $T[i,j,k]=1$ for all $i,j,k$, then $u[i,j]=1, v[j,k]=1, w[k,i]=1$ works. Rank 1.
    # BUT, the tensor is NOT all-ones.
    # The tensor is $T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}$.
    # The indices of the tensor components are $(I, J, K)$ where $I \in \{0..nm-1\}$, etc.
    # $T[I, J, K] = 1$ if $I=(i,j), J=(j,k), K=(k,i)$ for some $i,j,k$.
    # Otherwise 0.
    # So $T$ is SPARSE.
    # For $2 \times 2 \times 2$, there are 8 non-zero entries.
    # $T[0,0,0]=1$ (i=0,j=0,k=0)
    # $T[0,1,1]=1$ (i=0,j=0,k=1) -> I=0, J=1, K=1? 
    # Let's map indices:
    # $I = i \cdot m + j$.
    # $J = j \cdot p + k$.
    # $K = k \cdot n + i$.
    
    # For $i=0, j=0, k=0$: $I=0, J=0, K=0$. $T[0,0,0]=1$.
    # For $i=0, j=0, k=1$: $I=0, J=1, K=2$ (since $k=1, n=2 \implies 1\cdot2+0=2$). $T[0,1,2]=1$.
    # For $i=0, j=1, k=0$: $I=1, J=0, K=0$. $T[1,0,0]=1$.
    # For $i=0, j=1, k=1$: $I=1, J=1, K=2$. $T[1,1,2]=1$.
    # For $i=1, j=0, k=0$: $I=2, J=0, K=1$. $T[2,0,1]=1$.
    # For $i=1, j=0, k=1$: $I=2, J=1, K=3$. $T[2,1,3]=1$.
    # For $i=1, j=1, k=0$: $I=3, J=0, K=1$. $T[3,0,1]=1$.
    # For $i=1, j=1, k=1$: $I=3, J=1, K=3$. $T[3,1,3]=1$.
    
    # This sparse structure is what makes the rank > 1.
    
    # Now, we construct the Kronecker product.
    # $T_{2,4,6} = T_{2,2,2} \otimes T_{1,2,3}$.
    # If we have decompositions:
    # $T_{2,2,2} = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$
    # $T_{1,2,3} = \sum_{s=1}^6 u'_s \otimes v'_s \otimes w'_s$
    # Then $T_{2,4,6} = \sum_{r=1}^7 \sum_{s=1}^6 (u_r \otimes u'_s) \otimes (v_r \otimes v'_s) \otimes (w_r \otimes w'_s)$.
    # This gives rank $7 \times 6 = 42$.
    
    # We need the explicit Strassen decomposition for $T_{2,2,2}$.
    # Strassen's algorithm computes $C = AB$ using 7 multiplications.
    # The tensor decomposition corresponds to the 7 "multiplications" $M_1 \dots M_7$.
    # We need to extract $u_r, v_r, w_r$ for each $M_r$.
    
    # Let's define the Strassen terms explicitly.
    # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
    # $M_2 = (A_{10}+A_{11})B_{00}$
    # $M_3 = A_{00}(B_{10}-B_{11})$
    # $M_4 = A_{11}(B_{01}-B_{00})$
    # $M_5 = (A_{00}+A_{01})B_{11}$
    # $M_6 = (A_{10}-A_{00})(B_{00}+B_{10})$
    # $M_7 = (A_{01}-A_{11})(B_{01}+B_{11})$
    
    # The output $C$ is:
    # $C_{00} = M_1 + M_4 - M_5 + M_7$
    # $C_{01} = M_3 + M_5$
    # $C_{10} = M_2 + M_4$
    # $C_{11} = M_1 - M_2 + M_3 + M_6$
    
    # The tensor decomposition is:
    # $T = \sum_r u_r \otimes v_r \otimes w_r$
    # where $u_r$ is the vector of coefficients for $A$ in $M_r$,
    # $v_r$ is the vector of coefficients for $B$ in $M_r$,
    # $w_r$ is the vector of coefficients for $C$ in the combination of $M_r$'s.
    
    # Let's construct $U_s, V_s, W_s$ for $2 \times 2 \times 2$.
    # Dimensions: $U_s (4,7), V_s (4,7), W_s (4,7)$.
    # Indices for U: $i \cdot 2 + j$.
    # Indices for V: $j \cdot 2 + k$.
    # Indices for W: $k \cdot 2 + i$.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # r=0: M1
    # A: A00(1), A11(1) -> U[0,0]=1, U[3,0]=1
    # B: B00(1), B11(1) -> V[0,0]=1, V[3,0]=1
    # C: C00(1), C11(1) -> W[0,0]=1, W[3,0]=1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # r=1: M2
    # A: A10(1), A11(1) -> U[2,1]=1, U[3,1]=1
    # B: B00(1) -> V[0,1]=1
    # C: C10(1), C11(-1) -> W[1,1]=1, W[3,1]=-1
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1
    
    # r=2: M3
    # A: A00(1) -> U[0,2]=1
    # B: B10(1), B11(-1) -> V[2,2]=1, V[3,2]=-1
    # C: C01(1), C11(1) -> W[2,2]=1, W[3,2]=1
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1
    
    # r=3: M4
    # A: A11(1) -> U[3,3]=1
    # B: B01(1), B00(-1) -> V[1,3]=1, V[0,3]=-1
    # C: C00(1), C10(1) -> W[0,3]=1, W[1,3]=1
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    # r=4: M5
    # A: A00(1), A01(1) -> U[0,4]=1, U[1,4]=1
    # B: B11(1) -> V[3,4]=1
    # C: C00(-1), C01(1) -> W[0,4]=-1, W[2,4]=1
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    # r=5: M6
    # A: A10(1), A00(-1) -> U[2,5]=1, U[0,5]=-1
    # B: B00(1), B10(1) -> V[0,5]=1, V[2,5]=1
    # C: C11(1) -> W[3,5]=1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    W_s[3, 5] = 1
    
    # r=6: M7
    # A: A01(1), A11(-1) -> U[1,6]=1, U[3,6]=-1
    # B: B01(1), B11(1) -> V[1,6]=1, V[3,6]=1
    # C: C00(1) -> W[0,6]=1
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # Now construct the final matrices for 2x4x6.
    # Rank R = 7 * 6 = 42.
    # U shape (2*4, 42) = (8, 42)
    # V shape (4*6, 42) = (24, 42)
    # W shape (6*2, 42) = (12, 42)
    
    U_final = np.zeros((8, 42))
    V_final = np.zeros((24, 42))
    W_final = np.zeros((12, 42))
    
    # Kronecker product index mapping:
    # For $T_{n1,m1,p1} \otimes T_{n2,m2,p2}$:
    # $n = n1 n2, m = m1 m2, p = p1 p2$.
    # Here $n1=2, m1=2, p1=2$ and $n2=1, m2=2, p2=3$.
    # $n=2, m=4, p=6$.
    # 
    # Index in U: $I = (i1 \cdot m1 + j1) \cdot m2 + j2 = (i1 \cdot 2 + j1) \cdot 2 + j2$.
    # Wait, the Kronecker product of matrices $A \otimes B$ has index $(i_A, i_B)$.
    # The tensor $T = T1 \otimes T2$ has components $T[I, J, K] = T1[I1, J1, K1] T2[I2, J2, K2]$.
    # The decomposition is $\sum_{r1, r2} (u_{r1} \otimes u_{r2}) \otimes (v_{r1} \otimes v_{r2}) \otimes (w_{r1} \otimes w_{r2})$.
    # The column index in the final matrix is $r = r1 \cdot R2 + r2$.
    
    # Indices for U1 (2x2x2 part): $i1 \in \{0,1\}, j1 \in \{0,1\}$. Index $u1\_idx = i1 \cdot 2 + j1$.
    # Indices for U2 (1x2x3 part): $i2 \in \{0\}, j2 \in \{0,1\}$. Index $u2\_idx = i2 \cdot 2 + j2$.
    # Combined Index $U\_idx = u1\_idx \cdot 2 + u2\_idx$?
    # Let's check dimensions.
    # $U1$ is $4 \times 7$. $U2$ is $2 \times 6$.
    # $U\_final$ should be $8 \times 42$.
    # The Kronecker product $U1 \otimes U2$ has shape $(4 \cdot 2, 7 \cdot 6) = (8, 42)$.
    # The entry $(i1 \cdot 2 + i2, r1 \cdot 6 + r2)$ is $U1[i1, r1] U2[i2, r2]$.
    # This matches the structure.
    
    # We can just use np.kron.
    U_final = np.kron(U_s, U1)
    V_final = np.kron(V_s, V1)
    W_final = np.kron(W_s, W1)
    
    # Verify shapes
    # U_s (4,7), U1 (2,6) -> (8, 42). Correct.
    # V_s (4,7), V1 (6,6) -> (24, 42). Correct.
    # W_s (4,7), W1 (3,6) -> (12, 42). Correct.
    
    # Convert to list of lists for return
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
