"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    This implements Strassen's algorithm, which is the rank-minimal solution
    for 2x2 matrix multiplication over the reals (rank 7).
    
    Indexing conventions:
      U: shape (n*m, rank) = (4, 7). Row index i*m + j corresponds to A[i, j].
      V: shape (m*p, rank) = (4, 7). Row index j*p + k corresponds to B[j, k].
      W: shape (p*n, rank) = (4, 7). Row index k*n + i corresponds to C[i, k]
         following the trace convention W[k*n+i, r].
    
    The tensor identity is:
      T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    where T[i, j, k] is 1 if the multiplication tensor component is active.
    For matrix multiplication, the tensor is defined such that 
    C[i, k] = sum_j A[i, j] B[j, k].
    The decomposition must satisfy:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is a valid 
    multiplication path? No, the tensor T_{ijk} for matrix mult is 1 if 
    the indices match the multiplication structure. Specifically, the 
    multiplication tensor M_{ijk} is 1 if the indices correspond to 
    A_{ij} B_{jk} contributing to C_{ik}. 
    Actually, the standard definition is that the tensor is the sum of 
    rank-1 tensors. The condition is that the reconstructed product 
    C = A B is correct.
    
    Verification:
    For all i, j, k, the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if the term A[i,j]B[j,k] contributes to C[i,k]?
    Wait, the tensor T for matrix multiplication is defined by T_{ijk} = 1 
    for the specific indices involved in the sum? No.
    
    Let's look at the definition in the prompt: 
    "exactly decompose the matrix multiplication tensor".
    The matrix multiplication tensor is the 3-tensor T such that 
    T[i, j, k] = 1 if the indices are part of the multiplication? 
    No, the tensor is usually defined as the linear map.
    
    Standard check:
    Let U, V, W be the factors.
    For any matrices A and B, C = A B.
    C[i, k] = sum_j A[i, j] B[j, k].
    Using the decomposition:
    C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i, j]) * (sum_k' V[j'*p+k', r] B[j', k'])?
    No, the rank-1 term is u_r v_r w_r.
    The product computed is sum_r (sum_j U[i*m+j, r] A[i, j]) * (sum_k V[j*p+k, r] B[j, k]) * W[k*n+i, r]?
    This is not quite right.
    
    The correct identity for tensor decomposition of matrix multiplication:
    The tensor T has entries T_{ijk}. For matrix multiplication, T_{ijk} = 1 
    is not the right way to think about it.
    
    The tensor is defined such that 
    <T, A \otimes B \otimes C> = Tr(C^T A B).
    Or more simply, the decomposition must satisfy:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 
    is NOT the condition.
    
    The condition is that the reconstructed multiplication is correct.
    Specifically, for the basis elements of A, B, C:
    Let e_{ij} be the matrix with 1 at (i,j) and 0 elsewhere.
    The tensor T is defined by T(e_{ij}, e_{jk}, e_{ki}) = 1?
    
    Actually, the standard check for the factors U, V, W of the multiplication tensor is:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 
    IF the indices (i, j, k) form a valid "cycle" in the multiplication?
    
    Let's re-read the prompt's interface:
    "U', 'V', 'W': factor matrices ... that exactly decompose the matrix multiplication tensor"
    
    The matrix multiplication tensor $T$ is a 3-tensor.
    $T_{ijk} = 1$ if the indices correspond to the multiplication operation?
    No, $T$ is the tensor such that $T(A, B, C) = \text{Tr}(C^T A B)$.
    In basis coordinates, $T_{ijk}$ is 1 if $A=e_{ij}, B=e_{jk}, C=e_{ki}$?
    Wait, if $A=e_{ij}$, $B=e_{jk}$, then $AB = e_{ik}$.
    So $C$ must be $e_{ik}$ for the trace to be 1.
    The index for $C$ in W is $k*n+i$.
    So, for a specific triple of indices $(i, j, k)$ corresponding to $A_{ij}$, $B_{jk}$, $C_{ik}$:
    The term in the sum for the product $C_{ik}$ involves $A_{ij}$ and $B_{jk}$.
    
    The decomposition is valid if:
    For all $i, j, k$:
    $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$
    is WRONG.
    
    The correct condition is that the tensor reconstructed from U, V, W matches the multiplication tensor.
    The multiplication tensor $M$ has $M_{ijk} = 1$ if the indices are such that $A_{ij} B_{jk}$ contributes to $C_{ik}$?
    Actually, the tensor $M$ is defined by $M_{ijk} = 1$ for the specific indices?
    
    Let's look at the "seed" or standard implementations.
    Usually, the check is:
    For all $i, j, k$:
    $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = \delta_{\text{valid}}$?
    
    Let's test with rank 8 schoolbook.
    $U[i*m+j, r] = 1$ if $r$ corresponds to $(i,j)$.
    $V[j*p+k, r] = 1$ if $r$ corresponds to $(j,k)$.
    $W[k*n+i, r] = 1$ if $r$ corresponds to $(k,i)$.
    
    If we sum over $r$:
    The term is non-zero only if $r$ corresponds to a tuple $(i', j', k')$.
    $U$ is non-zero for $r=(i', j')$ only if $i'=i, j'=j$.
    $V$ is non-zero for $r=(i', j')$ only if $j'=j, k'=k$.
    $W$ is non-zero for $r=(i', j')$ only if $k'=k, i'=i$.
    
    So the sum is 1 if there exists an $r$ such that $r$ corresponds to $(i,j)$ for U, $(j,k)$ for V, and $(k,i)$ for W.
    This implies $i'=i, j'=j$ and $j'=j, k'=k$ and $k'=k, i'=i$.
    This is always consistent for any $i,j,k$.
    So for schoolbook, the sum is 1 for ALL $i,j,k$.
    
    Is the multiplication tensor 1 for all $i,j,k$?
    No. The multiplication tensor is sparse?
    $T_{ijk}$ is 1 if $A_{ij} B_{jk}$ is part of the product.
    Actually, the tensor $T$ for matrix multiplication is defined such that 
    $T_{ijk} = 1$ is not correct.
    
    The tensor is the map $(A, B) \to AB$.
    In basis, $T(e_{ij}, e_{jk}) = e_{ik}$.
    So the component $T_{ijk}$ (where $k$ is the output index) is 1 if $i,j,k$ are the indices of the input A, input B, and output C?
    Yes. $T_{ijk} = 1$ if the multiplication of basis element $e_{ij}$ and $e_{jk}$ yields $e_{ik}$.
    This is true for ALL $i,j,k$ in the range $0..n-1$.
    Wait, $e_{ij} e_{jk} = e_{ik}$. This is always true.
    So $T_{ijk} = 1$ for all $i,j,k$.
    
    Therefore, the condition is:
    $\sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1$ for all $i,j,k$.
    
    Let's verify this for Strassen.
    If this condition holds, the decomposition is valid.
    
    I will implement this check.
    """
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 intermediate products
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices (W[k*n+i]): 
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    
    # Product 0: M0 = (A00 + A11)(B00 + B11)
    # C00 += M0, C11 += M0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 1: M1 = (A10 + A11)B00
    # C10 += M1, C11 -= M1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Product 2: M2 = A00(B01 - B11)
    # C01 += M2, C11 += M2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Product 3: M3 = A11(B10 - B00)
    # C00 += M3, C10 += M3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Product 4: M4 = (A00 + A01)B11
    # C00 -= M4, C01 += M4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Product 5: M5 = (A10 - A00)(B00 + B01)
    # C11 += M5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Product 6: M6 = (A01 - A11)(B10 + B11)
    # C00 += M6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    # Verification
    # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to Rank 8 (Schoolbook)
        rank = 8
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
