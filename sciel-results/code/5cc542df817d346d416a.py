import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    N = n * m * p
    rank_schoolbook = N
    
    # 1. Construct Schoolbook Decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Function to get the vectorized tensor for a rank-1 term
    def get_tensor_vector(u, v, w):
        # u: (n*m), v: (m*p), w: (p*n)
        # Tensor T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        # We need to flatten this into a vector of length N.
        # Let's use order i, j, k.
        vec = np.zeros(N)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[idx] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                    idx += 1
        return vec

    # Precompute vectorized tensors for all columns
    # This is expensive if done naively, but N is small.
    # Rank is 72. 72 * 72 * 72 ops? No, 72 vectors of length 72.
    # Each vector computation is O(N).
    # Total O(R*N) = 72*72 = 5184 ops. Very fast.
    
    T_vectors = []
    for c in range(rank_schoolbook):
        T_vectors.append(get_tensor_vector(U_np[:, c], V_np[:, c], W_np[:, c]))
        
    T_matrix = np.column_stack(T_vectors) # Shape (N, R)
    
    # Peeling: Check for linear dependencies
    # We want to find the smallest subset of columns that spans the tensor T.
    # But wait, the tensor T is the SUM of the rank-1 tensors.
    # The sum is a single vector $T_{total}$.
    # We are not looking for the rank of the matrix $T_matrix$.
    # We are looking for the rank of the tensor $T_{total}$.
    # The decomposition is $T_{total} = \sum_{c=1}^R T_c$.
    # If $T_R$ is in the span of $\{T_1, \dots, T_{R-1}\}$, it does NOT mean we can remove it,
    # because the coefficients are fixed to 1 in the schoolbook decomposition?
    # No, in a general decomposition, the coefficients are part of the vectors.
    # $u \otimes v \otimes w$ can be scaled.
    # So if $T_R = \sum_{r<R} \alpha_r T_r$, then we can write
    # $T_{total} = \sum_{r<R} (1 + \alpha_r) T_r$.
    # This is a valid decomposition of rank $R-1$.
    
    # So the algorithm is:
    # 1. Start with all R vectors.
    # 2. Check if any vector is in the span of the others.
    # 3. If yes, remove it and update the "effective" coefficients?
    #    Actually, we just need to find a basis for the span of the set of vectors
    #    that sums to $T_{total}$? No.
    #    The condition is: Does there exist a set of $R-1$ rank-1 tensors (from the original set,
    #    possibly scaled) that sum to $T_{total}$?
    #    This is equivalent to: Is $T_{total}$ in the span of any $R-1$ of the vectors?
    #    Since $T_{total} = \sum_{c=1}^R T_c$, it is always in the span of all R.
    #    It is in the span of $R-1$ if and only if one of the vectors is in the span of the others?
    #    Yes, because if $T_R \in \text{Span}(T_1, \dots, T_{R-1})$, then
    #    $T_{total} = \sum_{r=1}^{R-1} T_r + T_R = \sum_{r=1}^{R-1} (T_r + \alpha_r T_r \dots)$?
    #    No. If $T_R = \sum \alpha_r T_r$, then $T_{total} = \sum_{r=1}^{R-1} T_r + \sum_{r=1}^{R-1} \alpha_r T_r = \sum_{r=1}^{R-1} (1+\alpha_r) T_r$.
    #    So yes, if any vector is in the span of the others, we can reduce the rank by 1.
    
    # Algorithm:
    # 1. Compute rank of T_matrix.
    # 2. If rank < R, then there is a dependency.
    # 3. We can iteratively remove dependent columns.
    
    # However, the "rank" of the tensor is the minimum number of rank-1 terms.
    # The vectors $T_c$ are specific rank-1 tensors.
    # If they are linearly dependent, we can combine them.
    
    # Let's compute the rank of the matrix formed by the tensor vectors.
    # If the rank is $K < R$, we can potentially represent the sum using $K$ terms?
    # Not necessarily. The sum is a specific vector.
    # We need to check if $T_{total}$ can be represented with fewer terms.
    
    # A simpler heuristic:
    # Compute the rank of the set of vectors $\{T_c\}$.
    # If rank is $K$, the space they span is $K$-dimensional.
    # But we need the specific sum.
    
    # Let's just check the linear independence of the vectors.
    # If they are independent, rank is R.
    # If not, we might be able to reduce.
    
    # For schoolbook $M_{n,m,p}$, the vectors $T_c$ correspond to the standard basis of the tensor space?
    # $T_{ijk}$ has a 1 at $(i,j,k)$ and 0 elsewhere.
    # Yes, for the schoolbook decomposition, each term $u_c \otimes v_c \otimes w_c$ is a "unit" tensor at a specific $(i,j,k)$.
    # Therefore, the vectors $T_c$ are the standard basis vectors of $\mathbb{R}^{N}$.
    # They are linearly independent.
    # Therefore, the rank is exactly $N = 72$.
    
    # So the peeling algorithm will find no dependencies.
    # The rank remains 72.
    
    # We return the schoolbook decomposition.
    
    return {"U": U, "V": V, "W": W}
