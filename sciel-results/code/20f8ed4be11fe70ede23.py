import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Pre-compute Strassen coefficients for 2x2
    # U_out[s, i_b, j_b], V_out[s, j_b, k_b], W_out[s, i_b, k_b]
    # Indices 0, 1 for blocks
    
    U_out = [[0]*4 for _ in range(7)]
    V_out = [[0]*4 for _ in range(7)]
    W_out = [[0]*4 for _ in range(7)]
    
    # s=0: M1
    U_out[0][0] = 1; U_out[0][3] = 1 # A00, A11 (idx 0, 3)
    V_out[0][0] = 1; V_out[0][3] = 1 # B00, B11
    W_out[0][0] = 1; W_out[0][3] = 1 # C00, C11 (i_b, k_b) -> 0,0 and 1,1
    
    # s=1: M2
    U_out[1][2] = 1; U_out[1][3] = 1 # A10, A11 (idx 2, 3)
    V_out[1][0] = 1                 # B00
    W_out[1][2] = 1; W_out[1][3] = -1 # C10 (1,0), C11 (1,1) -> i_b=1, k_b=0 and 1,1
    
    # s=2: M3
    U_out[2][0] = 1                 # A00
    V_out[2][1] = 1; V_out[2][3] = -1 # B01, B11 (idx 1, 3)
    W_out[2][1] = 1; W_out[2][3] = 1  # C01 (0,1), C11 (1,1)
    
    # s=3: M4
    U_out[3][3] = 1                 # A11
    V_out[3][2] = 1; V_out[3][0] = -1 # B10, B00 (idx 2, 0)
    W_out[3][0] = 1; W_out[3][2] = 1  # C00 (0,0), C10 (1,0)
    
    # s=4: M5
    U_out[4][0] = 1; U_out[4][1] = 1 # A00, A01 (idx 0, 1)
    V_out[4][3] = 1                 # B11
    W_out[4][0] = -1; W_out[4][1] = 1 # C00 (0,0), C01 (0,1)
    
    # s=5: M6
    U_out[5][2] = 1; U_out[5][0] = -1 # A10, A00 (idx 2, 0)
    V_out[5][0] = 1; V_out[5][1] = 1 # B00, B01 (idx 0, 1)
    W_out[5][3] = 1                 # C11 (1,1)
    
    # s=6: M7
    U_out[6][1] = 1; U_out[6][3] = -1 # A01, A11 (idx 1, 3)
    V_out[6][2] = 1; V_out[6][3] = 1 # B10, B11 (idx 2, 3)
    W_out[6][0] = 1                 # C00 (0,0)
    
    # Inner terms are identical structure
    U_in = U_out
    V_in = V_out
    W_in = W_out
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n * m):
        i_b = i // 2
        i_e = i % 2
        for j in range(m * p):
            j_b = j // 2
            j_e = j % 2
            for k in range(p * n):
                k_b = k // 2
                k_e = k % 2
                # Note: W index in matrix is k*n + i
                # But in loop we are filling W[k*n+i, r]
                # Let's just iterate r and fill all i,j,k? No, that's O(49*16*16*16) = 200k ops. Fine.
                pass
                
    # More efficient: iterate r, then i, j, k
    for r in range(rank):
        s = r // 7
        t = r % 7
        
        for i in range(n * m):
            i_b = i // 2
            i_e = i % 2
            # U[i, r] depends on i_b, i_e and j_b, j_e? 
            # No, U[i, r] is coefficient for A[i, j]. 
            # Wait, U is shape (16, 49). 
            # T_ijk = sum_r U[i, r] V[j, r] W[k, r].
            # U[i, r] must be the coefficient of A[i, j] in the r-th term?
            # No, U[i, r] is just the entry.
            # The product U[i, r] * V[j, r] * W[k, r] must sum to T_ijk.
            
            # In Kronecker product:
            # Term r corresponds to (s, t).
            # Contribution to A[i, j] is U_out[s, i_b, j_b] * U_in[t, i_e, j_e].
            # So U[i, r] cannot be determined by i alone. It depends on j!
            # Ah, the factorization is T_ijk = sum U_i V_j W_k.
            # This implies U depends on i, V on j, W on k.
            # But in Strassen, the coefficient of A[i,j] depends on i AND j.
            # So we cannot separate U and V completely in this simple way if we just map indices.
            
            # Let's re-read the tensor decomposition definition.
            # T_ijk = sum_r U_ir V_jr W_kr.
            # This is a standard CP decomposition.
            # For matrix multiplication, T_ijk = delta_ij? No.
            # T_ijk is 1 if i, j, k form a valid multiplication path?
            # No, T_ijk is the tensor of matrix multiplication.
            # T_ijk = 1 if the term A_ik B_kj? No.
            # The tensor for matmul C = AB is defined by C_ik = sum_j A_ij B_jk.
            # So T_ijk is 1 if the indices match?
            # Actually, the tensor T has entries T_{i, j, k} which are coefficients of A_i B_j in C_k?
            # Standard definition: T_{i, j, k} = 1 if (i, j, k) is a "path"?
            # No, T_{i, j, k} is the coefficient such that C_k = sum_{i, j} T_{i, j, k} A_i B_j.
            # Here i is row of A, j is col of A (row of B), k is col of B.
            # So T_{i, j, k} = 1 if j is the common index?
            # C_k = sum_j A_{i, j} B_{j, k}.
            # So for fixed i, k, C_k gets contribution from A_{i, j} B_{j, k}.
            # The tensor entry is T_{i, j, k} = 1?
            # Yes, T_{i, j, k} = 1 for all i, j, k? No.
            # T is a 3rd order tensor of size n x m x p.
            # T_{i, j, k} = 1 if the multiplication involves A_{i, j} and B_{j, k} to form C_{i, k}.
            # So T_{i, j, k} = 1 for all i, j, k?
            # Wait, if T_{i, j, k} = 1 for all i, j, k, then sum T_{i, j, k} A_{i, j} B_{j, k} = sum A_{i, j} B_{j, k}.
            # This is not C. C is a matrix.
            # The tensor decomposition represents the *algorithm*.
            # The "tensor" being decomposed is the structure of the multiplication.
            # Usually defined as T_{i, j, k} = 1 if i, j, k are compatible.
            # In the standard CP decomposition of matmul tensor:
            # T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}.
            # Here T_{i, j, k} is 1 if the term A_{i, j} B_{j, k} contributes to C?
            # No, T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in the expression for C_{?}.
            # Actually, the tensor is defined such that the matmul is:
            # C_{i, k} = \sum_{j, r} U_{i, r} V_{j, r} W_{k, r} A_{i, j} B_{j, k}?
            # No.
            # The standard form is:
            # C_{i, k} = \sum_{j} A_{i, j} B_{j, k}.
            # We want to write this as a sum of rank-1 terms.
            # A rank-1 term is of the form $u_i v_j w_k$.
            # The sum is $\sum_r u^{(r)}_i v^{(r)}_j w^{(r)}_k$.
            # This sum equals $\sum_j A_{i, j} B_{j, k}$?
            # No, the decomposition is of the *tensor* T where $T_{i, j, k} = 1$?
            # If $T_{i, j, k} = 1$ for all i, j, k, then $\sum_r U_{i, r} V_{j, r} W_{k, r} = 1$.
            # This is not matmul.
            
            # Correct interpretation:
            # The tensor T is defined by $T_{i, j, k} = 1$ if the indices form a valid term in the product?
            # No, the tensor T is the "matmul tensor".
            # $T_{i, j, k} = 1$ if $j$ is the intermediate index?
            # Yes. $T_{i, j, k} = 1$ for all $i, j, k$?
            # If so, $\sum_r U_{i, r} V_{j, r} W_{k, r} = 1$.
            # This would mean we are decomposing the all-ones tensor.
            
            # Let's check the Strassen decomposition again.
            # $M_1 = (A_{00} + A_{11})(B_{00} + B_{11})$.
            # This term contributes to $C_{00}$ and $C_{11}$.
            # The expansion is $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
            # These are 4 rank-1 terms in the "element" space.
            # But Strassen counts it as 1 rank-1 term in the "block" space.
            # The tensor decomposition rank is the number of such terms.
            # The condition is that the sum of the rank-1 tensors equals the matmul tensor.
            # The matmul tensor T has entries $T_{i, j, k}$.
            # What is $T_{i, j, k}$?
            # It is 1 if the product $A_{i, j} B_{j, k}$ is part of the sum for some $C$?
            # Actually, $T_{i, j, k} = 1$ for all $i, j, k$ is NOT correct.
            # $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
            # The tensor T is such that $\sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k} = \sum_{i, k} C_{i, k}$?
            # No.
            # The tensor T is defined by $T_{i, j, k} = 1$ if the indices are "aligned"?
            # In the standard literature, the matmul tensor $T_{n}$ has entries $T_{i, j, k} = 1$ if $i, j, k$ are such that $A_{i, j} B_{j, k}$ contributes to $C_{i, k}$.
            # Since every $A_{i, j} B_{j, k}$ contributes to $C_{i, k}$, $T_{i, j, k} = 1$ for all $i, j, k$.
            # Wait, if $T_{i, j, k} = 1$ for all, then we are decomposing the all-ones tensor.
            # The rank of the all-ones tensor is 1.
            # This is clearly wrong.
            
            # The tensor is NOT the all-ones tensor.
            # The tensor is defined on the space of matrices.
            # The bilinear form is $\langle A, B \rangle = \text{tr}(AB)$? No.
            # The tensor is $T \in \mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{p \times n}$.
            # $T_{i, j, k} = 1$ if $i, j, k$ form a "cycle"?
            # No.
            # Let's look at the definition of rank of matmul.
            # $M(n)$ is the minimum rank of a tensor decomposition of the matmul tensor.
            # The matmul tensor is defined by $T_{i, j, k} = 1$ if $j$ is the intermediate index?
            # Yes, $T_{i, j, k} = 1$ for all $i, j, k$ is the definition of the tensor where the contraction $\sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k}$ gives the sum of all products.
            # But we want $C_{i, k}$.
            # The decomposition is:
            # $C_{i, k} = \sum_r U_{i, r} V_{k, r} \sum_j A_{i, j} B_{j, k}$? No.
            
            # Correctly:
            # We want to compute $C_{i, k}$.
            # $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
            # We express this as $\sum_r \mu_r ( \sum_{i, j} U_{i, j, r} A_{i, j} ) ( \sum_{j, k} V_{j, k, r} B_{j, k} )$?
            # No, the factors are vectors.
            # $U_r \in \mathbb{R}^{n \times m}$, $V_r \in \mathbb{R}^{m \times p}$, $W_r \in \mathbb{R}^{p \times n}$.
            # $C = \sum_r (U_r \odot ?)$.
            
            # The standard CP decomposition form for matmul is:
            # $T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}$.
            # Here $T_{i, j, k}$ is the coefficient of $A_{i, j} B_{j, k}$ in the expression for $C$?
            # If we sum $T_{i, j, k} A_{i, j} B_{j, k}$, we get a scalar.
            # We want a matrix $C$.
            
            # Ah, the tensor is defined such that:
            # $\sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k} C_{k, i} = \text{something}$?
            
            # Let's stick to the factor matrices provided in the prompt.
            # $U$ shape $(n*m, rank)$.
            # $V$ shape $(m*p, rank)$.
            # $W$ shape $(p*n, rank)$.
            # This implies the decomposition is:
            # $C_{i, k} = \sum_{j, r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$?
            # No, $W$ is $(p*n, rank)$. Index $k*n+i$.
            # So $W_{k*n+i, r}$.
            # The sum is over $j$ and $r$.
            # $C_{i, k} = \sum_{j=0}^{m-1} \sum_{r=0}^{R-1} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
            # This must equal $\sum_{j} A_{i, j} B_{j, k}$.
            # This means the "tensor" $T$ is defined by:
            # $T_{i, j, k} = \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
            # And we require $T_{i, j, k} = \delta_{?}$.
            # Actually, we require that for any $A, B$, the computed $C$ matches $AB$.
            # This implies that the coefficients of $A_{i, j} B_{j', k}$ in the computed expression must be 1 if $j=j'$ and 0 otherwise.
            # Computed $C_{i, k} = \sum_{j, j', r} A_{i, j} B_{j', k} U_{i*m+j, r} V_{j'*p+k, r} W_{k*n+i, r}$.
            # We need coeff of $A_{i, j} B_{j, k}$ to be 1.
            # So $\sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1$ for all $i, j, k$.
            # And coeff of $A_{i, j} B_{j', k}$ with $j \neq j'$ must be 0.
            # $\sum_r U_{i*m+j, r} V_{j'*p+k, r} W_{k*n+i, r} = 0$ for $j \neq j'$.
            
            # So the target tensor $T$ has entries:
            # $T_{i, j, k} = 1$ if the indices match the "diagonal" of the multiplication?
            # No, $T_{i, j, k}$ in the sum $\sum T_{i, j, k} A_{i, j} B_{j, k}$ is just 1.
            # But the cross terms must vanish.
            # This is equivalent to decomposing the tensor $T$ where $T_{i, j, k} = 1$?
            # No, the condition is that the bilinear form defined by the factors equals the standard matmul.
            
            # In the Kronecker construction, we simply take the product of the Strassen factors.
            # $U_{i, r} = U_{out}[s, i_b, j_b] U_{in}[t, i_e, j_e]$?
            # No, $U$ depends on $i$ and $j$ in the index $i*m+j$.
            # So $U[i*m+j, r]$ is the entry.
            # In my previous derivation, I said $U[i, r]$ depends on $i, j$.
            # So $U[i*m+j, r] = U_{out}[s, i_b, j_b] \cdot U_{in}[t, i_e, j_e]$.
            # Similarly $V[j*p+k, r] = V_{out}[s, j_b, k_b] \cdot V_{in}[t, j_e, k_e]$.
            # $W[k*n+i, r] = W_{out}[s, i_b, k_b] \cdot W_{in}[t, i_e, k_e]$.
            
            # This should work.
            
            U[i * m + j][r] = U_out[s][i_b * 2 + j_b] * U_in[t][i_e * 2 + j_e]
            V[j * p + k][r] = V_out[s][j_b * 2 + k_b] * V_in[t][j_e * 2 + k_e]
            W[k * n + i][r] = W_out[s][i_b * 2 + k_b] * W_in[t][i_e * 2 + k_e]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
