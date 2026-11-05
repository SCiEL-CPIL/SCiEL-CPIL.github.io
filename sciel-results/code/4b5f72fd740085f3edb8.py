import numpy as np

def construct():
    n, m, p = 3, 5, 7
    # Schoolbook decomposition
    # U has shape (n*m, rank) -> (15, 105)
    # V has shape (m*p, rank) -> (35, 105)
    # W has shape (p*n, rank) -> (21, 105)
    # 
    # The tensor is T[i,j,k] = sum_l A[i,l] B[l,j]
    # Factorization: A B = sum_r U[:,r] * V[:,r] * W[:,r]^T (conceptually)
    # Actually, the standard factorization for matrix multiplication tensor T_{ijk} is:
    # T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r} ? No.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the tensor $T$ is decomposed as $T[i,j,k] = \sum_r U[i,r] V[j,r] W[k,r]$.
    # Wait, the shapes are:
    # U: (n*m, rank). This is unusual. Usually U is (n, rank), V is (m, rank), W is (p, rank).
    # Here U is (n*m, rank). This suggests the inputs are flattened?
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # This looks like a decomposition of the *linear map* or a specific tensor format.
    # Let's re-read carefully: "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor $T$ has indices $i \in [0,n), j \in [0,m), k \in [0,p)$.
    # $T[i,j,k] = \delta_{ik}$? No, $C = AB \implies C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So the tensor is $T[i,j,k] = \delta_{i,j?}$ No.
    # The tensor for matrix multiplication $A \times B$ is usually defined on indices $i,j,k$ such that $C_{ik} = \sum_j T_{ijk} A_{ij} B_{jk}$?
    # No, standard definition: $T_{ijk}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
    # So $T_{ijk} = 1$ if the indices match, 0 otherwise? No, it's a 3-tensor.
    # $T[i,j,k] = 1$ if $i,j,k$ form a valid path?
    # Actually, $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    # So the tensor $T$ has entries $T[i,j,k] = 1$ for all $i,j,k$? No.
    # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # The tensor is $T_{ijk} = \delta_{i, ?} \delta_{j, ?}$.
    # Let's look at the shapes again.
    # U: (n*m, rank). V: (m*p, rank). W: (p*n, rank).
    # This looks like the decomposition of the *matrix multiplication map* viewed as a trilinear form?
    # Or perhaps the tensor is flattened?
    # "W[k*n+i,r]" suggests W is indexed by output index (k*n+i) and rank r.
    # Output C is p x n? No, C is n x p.
    # If C is n x p, index is i (row), k (col).
    # "W[k*n+i,r]" implies index $k*n + i$. This flattens C into a vector of size $p*n$.
    # So W maps rank to the flattened output.
    # U maps input A (size n*m) to rank.
    # V maps input B (size m*p) to rank.
    #
    # So we are decomposing the linear map $L: \mathbb{R}^{nm} \times \mathbb{R}^{mp} \to \mathbb{R}^{pn}$.
    # $L(A, B) = AB$ (flattened).
    # This is a bilinear map.
    # The standard schoolbook decomposition has rank $n*m*p$.
    # For each term in the sum $\sum_j A_{ij} B_{jk}$, we have a rank-1 term.
    # Term $j$: $A_{ij}$ is the $(i*m + j)$-th element of flattened A?
    # Flattened A (row-major): index $i*m + j$.
    # Flattened B (row-major): index $j*p + k$.
    # Flattened C (row-major? or column?): index $k*n + i$ as per prompt.
    #
    # So for each $i, j, k$, we have a contribution to $C_{ik}$ from $A_{ij} B_{jk}$.
    # This corresponds to a rank-1 tensor in the bilinear map sense.
    # We can define:
    # U[:, r] has 1 at position $i*m + j$.
    # V[:, r] has 1 at position $j*p + k$.
    # W[:, r] has 1 at position $k*n + i$.
    #
    # Total rank = $n*m*p = 3*5*7 = 105$.
    #
    # This matches the shapes and the convention.
    #
    # I will implement this schoolbook decomposition.

    n, m, p = 3, 5, 7
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, r] = 1.0
                V[idx_v, r] = 1.0
                W[idx_w, r] = 1.0
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
