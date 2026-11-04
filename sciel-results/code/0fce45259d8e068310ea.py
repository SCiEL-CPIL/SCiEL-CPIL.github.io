import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition rank is n * m * p
    # For 5x5x5, rank = 125
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The matrix multiplication tensor is T[i, j, k] = A[i, j] * B[j, k]
    # We decompose this into sum of rank-1 tensors.
    # Each term corresponds to a specific index j in the summation.
    # Actually, each term in the bilinear expansion is for a fixed (i, j, k) ?
    # No, the bilinear form is C[i, k] = sum_j A[i, j] B[j, k].
    # The rank-1 decomposition of the multiplication tensor M555 is:
    # For each j (0 to m-1):
    #   For each i (0 to n-1):
    #     For each k (0 to p-1):
    #       Term r:
    #         U[i*m + j, r] = 1
    #         V[j*p + k, r] = 1
    #         W[k*n + i, r] = 1
    #       
    # Let's verify:
    # Sum_r U[i*m+j, r] V[j'*p+k', r] W[k''*n+i'', r]
    # If we fix i, j, k for the output C[i, k], and look at the coefficient of A[i', j'] B[j'', k'']:
    # The term contributes 1 only if (i'=i, j'=j) AND (j''=j, k''=k) AND (k'''=k, i''''=i).
    # Wait, the indices in U, V, W correspond to the input coordinates.
    # U index is (i_A, j_A). V index is (j_B, k_B). W index is (k_C, i_C).
    # The product is U[u, r] V[v, r] W[w, r].
    # We want sum_r U[i_A, j_A, r] V[j_B, k_B, r] W[k_C, i_C, r] to be 1 if (i_A, j_B, k_C) matches the structure?
    # No, the tensor T is defined such that T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k] to produce C[i,k]?
    # The standard tensor for matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k} = 1$ if $j=j$ (the inner index matches).
    # Actually, $T_{i,j,k}$ is the coefficient of $A_{i,j} B_{j,k}$ in $C_{i,k}$.
    # So $T_{i,j,k} = 1$ if the indices are consistent, i.e., the "j" in A is the same "j" in B.
    # The rank-1 decomposition is:
    # For each j in range(m):
    #   For each i in range(n):
    #     For each k in range(p):
    #       r = j * (n * p) + i * p + k
    #       U[i * m + j, r] = 1
    #       V[j * p + k, r] = 1
    #       W[k * n + i, r] = 1
       
    # Let's check if this sums correctly.
    # We compute $\hat{C}_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # Using the decomposition:
    # $\sum_r (\sum_{i', j'} U_{i', j', r} A_{i', j'}) (\sum_{j'', k''} V_{j'', k'', r} B_{j'', k''}) W_{k, i, r}$
    # Wait, the output is linear combination of $W$ terms?
    # The decomposition is $\sum_r U_r \otimes V_r \otimes W_r$.
    # The value at output index $(k, i)$ (since W is $p \times n$) is $\sum_r (\sum_{u} U_{u,r} A_u) (\sum_{v} V_{v,r} B_v) W_{w,r}$?
    # No, the tensor contraction is:
    # $C_{i,k} = \sum_{r} (\sum_{j} U_{i,j,r} A_{i,j}) (\sum_{j} V_{j,k,r} B_{j,k}) W_{k,i,r}$?
    # Let's stick to the indices provided in the prompt:
    # U: (n*m, rank). Index $u = i \cdot m + j$.
    # V: (m*p, rank). Index $v = j \cdot p + k$.
    # W: (p*n, rank). Index $w = k \cdot n + i$.
    
    # The bilinear form for term $r$ is $u_r \otimes v_r \otimes w_r$.
    # The contribution to $C_{i,k}$ is $\sum_r W_{k \cdot n + i, r} \cdot (\sum_{j} U_{i \cdot m + j, r} A_{i,j}) \cdot (\sum_{j} V_{j \cdot p + k, r} B_{j,k})$?
    # No, the standard tensor decomposition for matmul is:
    # $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}$? No, the indices must match.
    
    # Let's look at the definition of the tensor rank.
    # The matrix multiplication tensor $M_{n,m,p}$ is a tensor in $A^* \otimes B^* \otimes C^*$.
    # $M = \sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # Usually, $M = \sum_{j=0}^{m-1} \sum_{i=0}^{n-1} \sum_{k=0}^{p-1} e_i \otimes e_j \otimes e_k$ is not right.
    # The tensor is defined by $T(A, B) = A B$.
    # $T(A, B) = \sum_{i,k} ( \sum_j A_{ij} B_{jk} ) C_{ik}^*$.
    # In tensor notation: $T = \sum_{i,j,k} A_{ij} B_{jk} C_{ik}^*$.
    # So the coefficient of $A_{i,j} B_{j,k} C_{k,i}^*$ (using W index $k \cdot n + i$) is 1.
    # So for each triplet $(i, j, k)$, we have a rank-1 term:
    # $U_{i \cdot m + j, r} = 1$
    # $V_{j \cdot p + k, r} = 1$
    # $W_{k \cdot n + i, r} = 1$
    
    # This creates a rank of $n \cdot m \cdot p$.
    # Let's implement this.
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
