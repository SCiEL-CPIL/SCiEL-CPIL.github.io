import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_part, V_part, W_part for each of the 7 products
    # Indices for 2x2 matrix: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    # Define the 7 rank-1 tensors for 2x2x2
    # U is 4x1, V is 4x1, W is 4x1
    # U[i,j], V[j,k], W[k,i]
    
    strassen = []
    
    # M1 = (A11+A22)(B11+B22)
    # A: 11, 22 -> 1, 1 at indices 0, 3
    # B: 11, 22 -> 1, 1 at indices 0, 3 (B is 2x2, indices 0:11, 1:12, 2:21, 3:22)
    # W: C11, C22 -> 1, 1 at indices 0, 3
    strassen.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A21+A22)B11
    # A: 21, 22 -> 1, 1 at indices 2, 3
    # B: 11 -> 1 at index 0
    # W: C21 -> 1 at index 2
    strassen.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3 = A11(B12-B22)
    # A: 11 -> 1 at index 0
    # B: 12, 22 -> 1, -1 at indices 1, 3
    # W: C12 -> 1 at index 1
    strassen.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4 = A22(B21-B11)
    # A: 22 -> 1 at index 3
    # B: 21, 11 -> 1, -1 at indices 2, 0
    # W: C11, C21 -> 1, 1 at indices 0, 2
    strassen.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # M5 = (A11+A12)B22
    # A: 11, 12 -> 1, 1 at indices 0, 1
    # B: 22 -> 1 at index 3
    # W: C11, C12 -> -1, 1 at indices 0, 1
    strassen.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A21-A11)(B11+B12)
    # A: 21, 11 -> 1, -1 at indices 2, 0
    # B: 11, 12 -> 1, 1 at indices 0, 1
    # W: C22 -> 1 at index 3
    strassen.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A12-A22)(B21+B22)
    # A: 12, 22 -> 1, -1 at indices 1, 3
    # B: 21, 22 -> 1, 1 at indices 2, 3
    # W: C11, C22 -> 1, -1 at indices 0, 3
    strassen.append(([0,1,0,-1], [0,0,1,1], [1,0,0,-1]))
    
    # Verify Strassen
    # C = sum M * U * V * W? No, C = sum U_c * V_c * W_c
    # Actually, the tensor decomposition is C = sum_r U_r V_r W_r
    # Where U_r is a vector in A-space, V_r in B-space, W_r in C-space.
    # My definition above: U is coeffs for A, V for B, W for C.
    # Product is sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]?
    # No, the tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # Wait, the standard form is C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 if indices match? No.
    # The tensor T is such that <T, A, B> = C.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 if the j in A and j in B are the same?
    # Yes, T[i,j,k] is 1 if the "inner" index matches.
    # In the decomposition, we have rank-1 tensors.
    # A rank-1 tensor is defined by vectors u, v, w.
    # T[i,j,k] = u[i,j] v[j,k] w[k,i]? No.
    # The term is u[i,j] * v[j,k] * w[k,i] summed?
    # No, the rank-1 tensor is $u \otimes v \otimes w$ where $u$ is on A indices, $v$ on B indices, $w$ on C indices?
    # No, C is the output.
    # The multiplication is a map $A \times B \to C$.
    # The tensor $T$ lives in $A^* \otimes B^* \otimes C$.
    # $T_{i,j,k}$ corresponds to $A_{i,j} B_{j,k} \to C_{i,k}$.
    # So the indices are $i$ (A row), $j$ (A col / B row), $k$ (B col / C col).
    # Wait, C index is $(i,k)$.
    # So the tensor has indices $i, j, k$.
    # $T_{i,j,k} = 1$ if... wait.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This is $\sum_j T_{i,j,k} A_{i,j} B_{j,k}$?
    # No, $T_{i,j,k}$ must be such that when contracted with A and B, it gives C.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # This implies the tensor is $\sum_{j} e_i \otimes e_j \otimes e_k$? No.
    # The tensor is $\sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # If we contract with $A = \sum A_{i,j} e_i \otimes e_j$ and $B = \sum B_{j,k} e_j \otimes e_k$.
    # The result is a vector in C-space?
    # The task says: "decompose the matrix multiplication tensor".
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]".
    # U has size $n \times m \times \text{rank}$.
    # V has size $m \times p \times \text{rank}$.
    # W has size $p \times n \times \text{rank}$.
    # Note W index is $k*n+i$. So W is associated with $(k,i)$ which is $(C_{col}, C_{row})$.
    # So $W_{k,i,r}$.
    # The reconstruction is $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # Let's check the sum.
    # $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # This sums over $r$. But $j$ is free?
    # No, the tensor $T$ is $\sum_r U_r \otimes V_r \otimes W_r$.
    # $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # If we contract $T$ with $A$ and $B$:
    # $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = \sum_r (\sum_{i,j} U_{i,j,r} A_{i,j}) (\sum_{j,k} V_{j,k,r} B_{j,k}) (\sum_{k,i} W_{k,i,r} C_{i,k})$?
    # No, that's not right.
    # The standard definition:
    # $C = \sum_r (U_r^T A) (V_r^T B) W_r^T$?
    # Let's look at the indices in the prompt:
    # $U[i*m+j, c]$, $V[j*p+k, c]$, $W[k*n+i, c]$.
    # This implies $U$ is $nm \times R$, $V$ is $mp \times R$, $W$ is $pn \times R$.
    # The multiplication is $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # The decomposition is $C = \sum_r (U_r \cdot A) (V_r \cdot B) W_r$?
    # Let $u_r$ be the vector of $U[:,r]$. $u_r$ has indices $i,j$.
    # Let $v_r$ be the vector of $V[:,r]$. $v_r$ has indices $j,k$.
    # Let $w_r$ be the vector of $W[:,r]$. $w_r$ has indices $k,i$.
    # Then $C_{i,k} = \sum_r (\sum_{i,j} u_r[i,j] A[i,j]) (\sum_{j,k} v_r[j,k] B[j,k]) w_r[k,i]$?
    # This doesn't look like it produces a matrix $C$ directly.
    # Usually, the decomposition is:
    # $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$ is not a valid contraction if $j$ is not summed.
    # Ah, the tensor $T$ has indices $i,j,k$.
    # $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # If we sum over $r$, we get a tensor $T_{i,j,k}$.
    # Then $C_{i,k} = \sum_j T_{i,j,k} A_{i,j} B_{j,k}$?
    # No, $T$ should be such that $C = \text{contract}(T, A, B)$.
    # If $T_{i,j,k} = \delta_{j,j'}$, then $\sum_j \delta_{j,j'} A_{i,j} B_{j,k} = A_{i,j'} B_{j',k}$.
    # This is not $C_{i,k}$.
    # The correct tensor for matrix multiplication is $T_{i,j,k} = 1$ if $j$ is the inner index?
    # No, the tensor is $\sum_{j} e_i \otimes e_j \otimes e_k$?
    # Let's check the rank-1 structure.
    # A rank-1 tensor is $u \otimes v \otimes w$.
    # $T^{(r)}_{i,j,k} = u_r[i,j] v_r[j,k] w_r[k,i]$.
    # Then $T = \sum_r T^{(r)}$.
    # Then $C_{i,k} = \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$?
    # $\sum_{i,j,k} (\sum_r u_r[i,j] v_r[j,k] w_r[k,i]) A_{i,j} B_{j,k}$
    # $= \sum_r (\sum_{i,j} u_r[i,j] A_{i,j}) (\sum_{j,k} v_r[j,k] B_{j,k}) (\sum_{k,i} w_r[k,i] C_{i,k})$?
    # This results in a scalar. But we want a matrix $C$.
    # The prompt says "decompose the matrix multiplication tensor".
    # And "U, V, W ... exactly decompose".
    # And "output factor follows ... W[k*n+i,r]".
    # This implies the reconstruction is:
    # $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # If we sum over $r$, do we sum over $j$?
    # No, $j$ is an index of $U$ and $V$.
    # If the formula is $C_{i,k} = \sum_{j} \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$, then $W$ depends on $j$? No.
    # This formula doesn't make sense because $W$ doesn't have $j$.
    # Unless $W$ is also a function of $j$? No, $W$ is $p \times n$.
    # Let's look at the seed code.
    # `U[i * m + j][column] = 1`
    # `V[j * p + k][column] = 1`
    # `W[k * n + i][column] = 1`
    # In the seed, for a fixed column (rank-1 term), $i,j,k$ are fixed.
    # So $U$ has 1 at $(i,j)$, $V$ has 1 at $(j,k)$, $W$ has 1 at $(k,i)$.
    # The sum is over columns $c$.
    # $C_{i,k} = \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$?
    # If we fix $i,k$, and sum over $c$:
    # $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
    # But $j$ is not summed in the expression?
    # Wait, in the seed, for each $c$, $i,j,k$ are specific values.
    # So $U_{i',j',c} = \delta_{i',i} \delta_{j',j}$.
    # $V_{j',k',c} = \delta_{j',j} \delta_{k',k}$.
    # $W_{k',i',c} = \delta_{k',k} \delta_{i',i}$.
    # If we compute $\sum_{j} \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$:
    # For a fixed $c$, the term is non-zero only if $j$ matches the $j$ in $U$ and $V$.
    # So $\sum_j U_{i,j,c} V_{j,k,c}$ is 1 if the $j$ in $U$ equals the $j$ in $V$ (which it does, by construction) AND $i,k$ match?
    # Actually, $U_{i,j,c}$ is non-zero only for specific $i,j$.
    # So $\sum_j U_{i,j,c} V_{j,k,c}$ is 1 if $i,j$ matches $U$'s index and $j,k$ matches $V$'s index.
    # This requires $j$ to be the same.
    # Then we multiply by $W_{k,i,c}$.
    # So the term contributes to $C_{i,k}$ only if $W$'s index $(k,i)$ matches.
    # So $C_{i,k} = \sum_c \delta_{i, i_c} \delta_{k, k_c}$.
    # In the seed, $i_c, j_c, k_c$ are all distinct for each $c$?
    # No, $c$ iterates over all $i,j,k$.
    # So for a given $i,k$, the sum is over all $j$ and all $c$ such that $i_c=i, k_c=k$.
    # $c$ is determined by $i_c, j_c, k_c$.
    # So $C_{i,k} = \sum_j 1 = m = 4$.
    # But we want $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The seed code returns U,V,W. The evaluator likely does:
    # `C = np.einsum('ijc,jkc,kic->ik', U, V, W)`?
    # Or `C = np.einsum('ijc,jkc,kic->ik', U, V, W)`
    # Let's check the indices in einsum.
    # U: i,j,c
    # V: j,k,c
    # W: k,i,c
    # Result: i,k
    # Sum over j and c.
    # $\sum_{j,c} U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
    # This matches my derivation.
    # So the algorithm is:
    # For each rank-1 component $r$, define $U_r, V_r, W_r$.
    # $U_r$ is a vector of length $nm$.
    # $V_r$ is a vector of length $mp$.
    # $W_r$ is a vector of length $pn$.
    # The contribution to $C_{i,k}$ is $\sum_j U_{r}[i*m+j] V_{r}[j*p+k] W_{r}[k*n+i]$.
    
    # So for Strassen, we need to map the 2x2x2 decomposition to the 2x4x4 indices.
    # For $A_L B_L$:
    # $A_L$ corresponds to $A_{i, j}$ with $j \in \{1,2\}$.
    # $B_L$ corresponds to $B_{j, k}$ with $j \in \{1,2\}$.
    # $C_L$ corresponds to $C_{i, k}$ with $k \in \{1,2\}$.
    # In the 2x2x2 Strassen, indices are 0,1,2,3.
    # Map 0->1, 1->2 for j?
    # Let's use 0-based indexing in code.
    # $i \in \{0,1\}$.
    # $j \in \{0,1,2,3\}$.
    # $k \in \{0,1,2,3\}$.
    
    # For $A_L B_L$:
    # $j \in \{0,1\}$.
    # $k \in \{0,1\}$.
    # $i \in \{0,1\}$.
    # The 2x2 matrix $A_L$ has indices $(i, j')$ where $j' \in \{0,1\}$.
    # The 2x2 matrix $B_L$ has indices $(j', k')$ where $j' \in \{0,1\}, k' \in \{0,1\}$.
    # The 2x2 matrix $C_L$ has indices $(i, k')$.
    
    # In the global indices:
    # $U$ index: $i * 4 + j$.
    # $V$ index: $j * 4 + k$.
    # $W$ index: $k * 2 + i$.
    
    # For the Strassen term $r$ in the first block:
    # $U_r[i, j']$ is given by Strassen U.
    # So $U_{global}[i * 4 + j'] = U_{strassen}[i, j']$.
    # $V_r[j', k']$ is given by Strassen V.
    # So $V_{global}[j' * 4 + k'] = V_{strassen}[j', k']$.
    # $W_r[k', i]$ is given by Strassen W.
    # So $W_{global}[k' * 2 + i] = W_{strassen}[k', i]$.
    
    # For $A_R B_R$:
    # $j \in \{2,3\}$.
    # $k \in \{2,3\}$.
    # $i \in \{0,1\}$.
    # $j' = j - 2$.
    # $k' = k - 2$.
    # $U_{global}[i * 4 + (2 + j')] = U_{strassen}[i, j']$.
    # $V_{global}[(2 + j') * 4 + (2 + k')] = V_{strassen}[j', k']$.
    # $W_{global}[(2 + k') * 2 + i] = W_{strassen}[k', i]$.
    
    # We sum these two sets of 14 terms.
    # Total rank 28.

    for r in range(14):
        # First half: A_L B_L
        u_s, v_s, w_s = strassen[r]
        # u_s, v_s, w_s are lists of 4 elements (flattened 2x2)
        # Map to global
        for i in range(2):
            for j in range(2):
                U[i * 4 + j][r] = u_s[i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[j * 4 + k][r] = v_s[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][r] = w_s[k * 2 + i]
        
        # Second half: A_R B_R (offset by 14)
        r2 = r + 14
        for i in range(2):
            for j in range(2):
                U[i * 4 + (2 + j)][r2] = u_s[i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[(2 + j) * 4 + (2 + k)][r2] = v_s[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * 2 + i][r2] = w_s[k * 2 + i]

    return {"U": U, "V": V, "W": W}

