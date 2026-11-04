import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n = m = p = 5

    def build_rank_r(r):
        """Build U,V,W of shape (n*m, r), (m*p, r), (p*n, r) with rank-r matmul decomposition.

        Structure: pick r vectors alpha in R^m (indexed j) and r vectors beta in R^n (indexed i),
        plus r coefficients gamma in R^n (indexed k). Then
          U[i*m+j, t] = alpha[j,t]          (A-factor depends on j only)
          V[j*p+k, t] = delta(j,k)          (B-factor is identity in j,k)
          W[k*n+i, t] = beta[i,t]*gamma[k,t]  (C-factor)
        Then (U,V,W)[i,j,k] = sum_t alpha[j,t] * delta(j,k) * beta[i,t]*gamma[k,t]
                            = sum_t alpha[k,t] * beta[i,t] * gamma[k,t]
        We need this to equal delta(i,k).
        So for each i,k: sum_t alpha[k,t]*beta[i,t]*gamma[k,t] = delta(i,k).
        """
        # We want: for all i, sum_t beta[i,t]*g_t = e_i (where g_t depends on k)
        # This is hard with the factorized structure. Let me reconsider.
        #
        # Better: U[i*m+j, t] = A[i,t]  (depends on i only, not j)
        #          V[j*p+k, t] = B[j,t]  (depends on j only)
        #          W[k*n+i, t] = C[k,t]  (depends on k only)
        # Then product = A[i,t] * B[j,t] * C[k,t], summed over t.
        # We need sum_t A[i,t]*B[j,t]*C[k,t] = delta(i,k) for all i,j,k.
        #
        # For this to hold for all j, we need sum_t B[j,t]*A[i,t]*C[k,t] = delta(i,k) for each j.
        # This means the matrix (A*C) (n x n via t-contraction) weighted by B[j,t] gives delta(i,k).
        #
        # Simplest: let B[j,t] = delta(j, t mod m) ... this is getting complex.
        #
        # Use the known rank-n construction:
        # M[i,j,k] = delta(i,k). This is rank n because:
        #   M = sum_{i=0}^{n-1} u_i ⊗ v ⊗ u_i  where u_i = e_i, v = sum_j e_j
        # Check: (u_i ⊗ v ⊗ u_i)[a,b,c] = u_i[a]*v[b]*u_i[c] = delta(a,i)*1*delta(c,i) = delta(a,c)*delta(a,i)*1
        # sum_i delta(a,i)*delta(c,i) = delta(a,c) * (1 if a=c else 0) ... 
        # Actually sum_i delta(a,i)delta(c,i) = delta(a,c) since only i=a=c contributes.
        # So rank = n works with U[i*m+j, i] = 1 (for the i-th column), V[j*p+k, i] = 1, W[k*n+i, i] = 1.
        # Wait, let me verify: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
        # = sum_t delta(i,t)*1*delta(k,t) = delta(i,k) (when i=k=t, else 0). Correct!
        
        r_actual = n  # rank = n = 5
        U = np.zeros((n*m, r_actual))
        V = np.zeros((m*p, r_actual))
        W = np.zeros((p*n, r_actual))
        
        for i in range(n):
            # Column i: U has 1s where row i*m+j for all j
            for j in range(m):
                U[i*m + j, i] = 1
            V[:, i] = 1  # all entries in column i are 1
            for k in range(p):
                W[k*n + i, i] = 1
        
        return U, V, W, r_actual

    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - (1.0 if i == k else 0.0)) > 1e-9:
                        return False
        return True

    # Primary: rank-n construction
    U, V, W, r = build_rank_r(n)
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    r = n*m*p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
