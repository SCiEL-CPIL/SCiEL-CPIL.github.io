import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    rank = 15
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add Strassen decomposition for 2x2x2 block
    # A block: rows i0..i1, cols j0..j1
    # B block: rows j0..j1, cols k0..k1
    # C block: rows i0..i1, cols k0..k1
    def add_strassen(i0, j0, k0, base_col):
        # Indices in A: (i, j) -> i*m + j
        # Indices in B: (j, k) -> j*p + k
        # Indices in W: (k, i) -> k*n + i
        
        # A elements
        # A11: (i0, j0), A12: (i0, j1)
        # A21: (i1, j0), A22: (i1, j1)
        
        # B elements
        # B11: (j0, k0), B12: (j0, k1)
        # B21: (j1, k0), B22: (j1, k1)
        
        # Strassen terms
        # M1 = (A11 + A22)(B11 + B22)
        U[i0*m + j0, base_col] += 1
        U[i1*m + j1, base_col] += 1
        V[j0*p + k0, base_col] += 1
        V[j1*p + k1, base_col] += 1
        # W contributions: C11 += M1, C22 += M1
        W[k0*n + i0, base_col] += 1
        W[k1*n + i1, base_col] += 1
        
        # M2 = (A21 + A22)B11
        U[i1*m + j0, base_col+1] += 1
        U[i1*m + j1, base_col+1] += 1
        V[j0*p + k0, base_col+1] += 1
        # W contributions: C21 += M2, C22 -= M2
        W[k0*n + i1, base_col+1] += 1
        W[k1*n + i1, base_col+1] -= 1
        
        # M3 = A11(B12 - B22)
        U[i0*m + j0, base_col+2] += 1
        V[j0*p + k1, base_col+2] += 1
        V[j1*p + k1, base_col+2] -= 1
        # W contributions: C12 += M3, C22 += M3
        W[k1*n + i0, base_col+2] += 1
        W[k1*n + i1, base_col+2] += 1
        
        # M4 = A22(B21 - B11)
        U[i1*m + j1, base_col+3] += 1
        V[j1*p + k0, base_col+3] += 1
        V[j0*p + k0, base_col+3] -= 1
        # W contributions: C11 += M4, C21 += M4
        W[k0*n + i0, base_col+3] += 1
        W[k0*n + i1, base_col+3] += 1
        
        # M5 = (A11 + A12)B22
        U[i0*m + j0, base_col+4] += 1
        U[i0*m + j1, base_col+4] += 1
        V[j1*p + k1, base_col+4] += 1
        # W contributions: C11 -= M5, C12 += M5
        W[k0*n + i0, base_col+4] -= 1
        W[k1*n + i0, base_col+4] += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[i1*m + j0, base_col+5] += 1
        U[i0*m + j0, base_col+5] -= 1
        V[j0*p + k0, base_col+5] += 1
        V[j0*p + k1, base_col+5] += 1
        # W contributions: C22 += M6
        W[k1*n + i1, base_col+5] += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[i0*m + j1, base_col+6] += 1
        U[i1*m + j1, base_col+6] -= 1
        V[j1*p + k0, base_col+6] += 1
        V[j1*p + k1, base_col+6] += 1
        # W contributions: C11 += M7
        W[k0*n + i0, base_col+6] += 1
        
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2, 7)
    
    # Part 3: i in {0,1}, j=2, k in {0,1,2,3}
    # T_{i,2,k} = 1
    # U_{i,14} = 1
    # V_{2,14} = 1
    # W_{k,14} = 1
    for i in range(n):
        U[i*m + 2, 14] = 1
    V[2*p + 0, 14] = 1 # V index is j*p + k. Here j=2, but k varies?
    # Wait. V index is (j, k).
    # The term is A_{i,2} B_{2,k}.
    # So we need V_{2,k} to be 1 for all k?
    # No, the tensor term is $U_{i,r} V_{j,r} W_{k,r}$.
    # We want $C_{i,k} += A_{i,2} B_{2,k}$.
    # So we need $V_{2,r} = 1$ and $W_{k,r} = 1$?
    # If $V_{2,r}=1$ and $W_{k,r}=1$, then sum is $A_{i,2} \cdot 1 \cdot 1 = A_{i,2}$.
    # This adds $A_{i,2}$ to all $C_{i,k}$.
    # But we want $A_{i,2} B_{2,k}$.
    # The decomposition must be linear in B.
    # $V_{j,r}$ must extract $B_{2,k}$?
    # No, $V$ is a matrix of size $m \times p \times rank$.
    # $V_{j*p+k, r}$.
    # We need $\sum_r U_{i,r} V_{2*p+k, r} W_{k'*n+i', r}$?
    # No, the output index is $k$ (col of C) and $i$ (row of C).
    # $C_{i,k} = \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$?
    # The summation is over $j$ and $r$.
    # $C_{i,k} = \sum_{j=0}^{m-1} \sum_{r=0}^{rank-1} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
    
    # For the term $A_{i,2} B_{2,k}$:
    # We need to isolate $j=2$.
    # So $U_{i*m+2, r} = 1$.
    # $V_{2*p+k, r} = 1$?
    # If $V_{2*p+k, r} = 1$ for all $k$, then the term is $A_{i,2} \cdot 1 \cdot W_{k*n+i, r}$.
    # We want this to be $A_{i,2} B_{2,k}$.
    # This implies $W_{k*n+i, r}$ must depend on $B_{2,k}$?
    # No, $W$ is constant.
    # So this simple rank-1 assumption for $M_{2,1,4}$ was wrong in the context of the full tensor?
    # Let's re-evaluate $M_{2,1,4}$.
    # $C_{i,k} = A_{i,1} B_{1,k}$.
    # Tensor $T_{i,1,k} = 1$.
    # Decomposition: $T_{i,1,k} = \sum_r U_{i,r} V_{1,r} W_{k,r}$.
    # Here indices are $i, j=1, k$.
    # $U_{i,r}$ depends on $i$.
    # $V_{1,r}$ depends on $j=1$.
    # $W_{k,r}$ depends on $k$.
    # We need $\sum_r U_{i,r} V_{1,r} W_{k,r} = 1$ for all $i,k$.
    # Let $r=1$. $U_{i,1}=1, V_{1,1}=1, W_{k,1}=1$.
    # Then $\sum A_{i,1} V_{1,1} W_{k,1} = A_{i,1} \cdot 1 \cdot 1 = A_{i,1}$.
    # This gives $C_{i,k} = A_{i,1}$.
    # But we want $C_{i,k} = A_{i,1} B_{1,k}$.
    # The $B_{1,k}$ is missing!
    # Ah, $V$ multiplies $B$.
    # In the formula $C = U B V^T W$? No.
    # The tensor contraction is:
    # $C_{i,k} = \sum_{j,r} A_{i,j} U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # No, standard tensor decomposition for matrix multiplication:
    # $C_{i,k} = \sum_{j,r} A_{i,j} V_{j,r} W_{k,r} U_{i,r}$?
    # Let's stick to the interface:
    # $U$ shape $(n*m, rank)$. Index $i*m+j$.
    # $V$ shape $(m*p, rank)$. Index $j*p+k$.
    # $W$ shape $(p*n, rank)$. Index $k*n+i$.
    # $C_{i,k} = \sum_{j,r} A_{i,j} V_{j*p+k, r} W_{k*n+i, r}$?
    # No, $U$ must be involved.
    # Usually: $C_{i,k} = \sum_r (\sum_j A_{i,j} U_{i*m+j, r}) (\sum_k B_{j,k} V_{j*p+k, r})$?
    # No, the sum over $r$ is outside.
    # $C_{i,k} = \sum_r U_{i, r} V_{k, r} W_{?}$?
    
    # Let's check the schoolbook implementation in the prompt.
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # This corresponds to $A_{i,j} B_{j,k}$.
    # So $C_{i,k} = \sum_{j,r} A_{i,j} V_{j*p+k, r} W_{k*n+i, r}$?
    # Where is U?
    # If $U_{i*m+j, r} = 1$, then it acts as a selector.
    # Actually, the bilinear form is $\langle A, B \rangle = \sum_{i,j,k} C_{i,k} T_{i,j,k}$.
    # $T_{i,j,k} = 1$.
    # $C_{i,k} = \sum_{j,r} A_{i,j} B_{j,k} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$?
    # No, that's not right.
    
    # Correct interpretation:
    # $C_{i,k} = \sum_r (\sum_j A_{i,j} U_{i*m+j, r}) (\sum_k B_{j,k} V_{j*p+k, r})$?
    # No, the indices must match.
    
    # Let's assume the standard:
    # $C_{i,k} = \sum_r U_{i,r} V_{k,r} \dots$?
    
    # Let's look at the schoolbook:
    # For each (i,j,k), there is a column c.
    # U[i*m+j, c] = 1.
    # V[j*p+k, c] = 1.
    # W[k*n+i, c] = 1.
    # This implies that the contribution to $C_{i',k'}$ is:
    # $\sum_{i,j,k} A_{i,j} B_{j,k} \delta_{i,i'} \delta_{k,k'}$?
    
    # It is safer to just verify the Strassen part works with the same indexing.
    # And for the last part, we need to represent $A_{i,2} B_{2,k}$.
    # We need $C_{i,k} += A_{i,2} B_{2,k}$.
    # Using 1 column:
    # $U_{i*m+2, r} = 1$.
    # $V_{2*p+k, r} = 1$ for all $k$?
    # If $V_{2*p+k, r} = 1$, then $\sum_k B_{2,k} V_{2*p+k, r} = \sum_k B_{2,k}$.
    # This sums B, which is wrong.
    # We need $V_{2*p+k, r}$ to pick out $B_{2,k}$.
    # So $V_{2*p+k, r} = \delta_{k, k_{target}}$?
    # But $k$ is the output index.
    # We need to map $B_{2,k}$ to $C_{i,k}$.
    # This requires 4 terms?
    # Term for $k=0$: $U_{i*m+2, r_0}=1, V_{2*p+0, r_0}=1, W_{0*n+i, r_0}=1$.
    # Term for $k=1$: $U_{i*m+2, r_1}=1, V_{2*p+1, r_1}=1, W_{1*n+i, r_1}=1$.
    # ...
    # So $M_{2,1,4}$ has rank 4?
    # $C_{i,k} = A_{i,2} B_{2,k}$.
    # This is 4 independent scalar products $A_{i,2} B_{2,k}$ for $k=0..3$.
    # Each requires 1 rank?
    # For fixed $k$, $C_{i,k} = A_{i,2} B_{2,k}$.
    # This is rank 1 in $i$?
    # $U_{i*m+2, r_k} = 1$.
    # $V_{2*p+k, r_k} = 1$.
    # $W_{k*n+i, r_k} = 1$.
    # Yes, 4 ranks.
    
    # So total rank = 7 + 7 + 4 = 18.
    
    # 18 is still better than 24.
    # Score 24/18 = 1.33.
    
    # Can we do better on the last part?
    # No, it's a diagonal matrix in k. Rank 4.
    
    # So I will implement Rank 18.

    # Update add_strassen to be correct.
    # The previous add_strassen logic for W might be flawed if I didn't check indices carefully.
    # Let's re-verify Strassen W indices.
    # $C_{11} = M_1 + M_4 - M_5 + M_7$.
    # $W_{k0*n + i0, r}$.
    # $k0$ corresponds to $B$ col 0 (local).
    # $i0$ corresponds to $A$ row 0 (local).
    # This maps to $C_{i0, k0}$.
    # Correct.

    # So I will use Rank 18.

    # Re-initialize
    rank = 18
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 2, 7)
    
    # Part 3: j=2
    # k=0
    for i in range(n):
        U[i*m + 2, 14] = 1
    V[2*p + 0, 14] = 1
    for i in range(n):
        W[0*n + i, 14] = 1
        
    # k=1
    for i in range(n):
        U[i*m + 2, 15] = 1
    V[2*p + 1, 15] = 1
    for i in range(n):
        W[1*n + i, 15] = 1
        
    # k=2
    for i in range(n):
        U[i*m + 2, 16] = 1
    V[2*p + 2, 16] = 1
    for i in range(n):
        W[2*n + i, 16] = 1
        
    # k=3
    for i in range(n):
        U[i*m + 2, 17] = 1
    V[2*p + 3, 17] = 1
    for i in range(n):
        W[3*n + i, 17] = 1
        
    return {"U": U, "V": V, "W": W}
