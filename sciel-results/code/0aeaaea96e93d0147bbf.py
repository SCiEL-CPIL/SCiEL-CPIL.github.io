"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time
import itertools
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit

    # Schoolbook rank
    rank_schoolbook = n * m * p  # 112

    # Try to find a lower rank decomposition using optimization
    # The matrix multiplication tensor T has indices (i, j, k) -> output index (k*n + i)
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-way tensor factorization (CP decomposition) of the matmul tensor.

    # For small matrices, we can try various ranks and use optimization
    # The matmul tensor is sparse: T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Actually no - the matmul tensor for C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k), and the output is C[i,k]
    # The decomposition: C[i,k] = sum_r U[i, r] * V[r, k] is standard matrix factorization
    # But here we need: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # Wait, let me re-read the problem.

    # The problem says: "Matrix multiplication 4,4,7"
    # A is 4x4, B is 4x7, C is 4x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this sums over r, and the indices of U, V, W depend on i,j,k
    # Actually, the standard formulation is:
    # C[i,k] = sum_r U[i, r] * V[r, k]  (rank-r matrix factorization of the multiplication)
    # But that's not what's being asked. The problem asks for a tensor decomposition
    # where the matmul tensor T[i,j,k] is decomposed as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

    # Let me reconsider. The matmul tensor for multiplying an (n x m) matrix by an (m x p) matrix
    # has entries T[i,j,k] = 1 (it's the identity in the sense that C[i,k] = sum_j A[i,j]B[j,k])
    # The tensor rank is the minimum r such that T[i,j,k] = sum_{r=1}^R U[i,j,r] V[j,k,r] W[k,i,r]
    # Wait, that doesn't look right either. Let me look at the schoolbook construction.

    # In the schoolbook: for each (i,j,k), we have one term with U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So the product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if there's a term for (i,j,k)
    # But we need: C[i,k] = sum_j A[i,j] B[j,k]
    # The way this works: C[i,k] = sum_j (A[i,j] B[j,k]) = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] * A[i,j] * B[j,k]
    # Hmm, that's not quite right. Let me think about this differently.

    # The standard formulation: We want to compute C = AB where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition means: C[i,k] = sum_{r=1}^R U[i, r] V[r, k]
    # But that's just matrix rank, which is min(n,p) for the output. That's not tensor rank.

    # The tensor rank formulation: We view the multiplication as a bilinear map.
    # The multiplication tensor T has T[i,j,k] = delta_{i,k}... no.
    # Actually, the multiplication tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, that's not standard. Let me look at the schoolbook again.

    # Schoolbook: rank = n*m*p, and for each (i,j,k):
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 when r corresponds to (i,j,k)
    # The sum over r gives: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For a fixed (i,j,k), this equals 1 if there's a term for (i,j,k), else 0.
    # But we need the sum over j to give C[i,k] = sum_j A[i,j]B[j,k].

    # I think the correct interpretation is:
    # We want to express the bilinear form: sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_r (sum_i U[i*m+j, r] A[i,j]) (sum_k V[j*p+k, r] B[j,k]) (sum_{i,k} W[k*n+i, r] C[i,k])
    # No, that's not right either.

    # Let me just try to find a lower rank by optimization. The key insight is that
    # for small matrix sizes, the tensor rank might be less than n*m*p.
    # For 4x4 times 4x7, the schoolbook rank is 112.
    # Known results: rank of 4x4x4 matrix multiplication is 49 (Strassen-like).
    # For 4x4x7, it might be lower than 112.

    # Let's try a simple approach: use random initialization and gradient descent
    # to minimize the error of the tensor decomposition.

    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        # Verify: for all (i,j,k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should equal 1
        # Actually, the condition is that the bilinear map is preserved.
        # The correct check: the tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # should equal 1 for all valid (i,j,k).
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    error += abs(val - 1.0)
        return error

    # Try to find a decomposition with rank < 112
    best_rank = rank_schoolbook
    best_U = [[0]*rank_schoolbook for _ in range(n*m)]
    best_V = [[0]*rank_schoolbook for _ in range(m*p)]
    best_W = [[0]*rank_schoolbook for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1

    # Try ranks from 100 down to 50
    for target_rank in range(100, 49, -1):
        if time.time() - start_time > DEADLINE - 20:
            break

        # Random initialization
        np.random.seed(42)
        U_init = np.random.randn(n*m, target_rank) * 0.1
        V_init = np.random.randn(m*p, target_rank) * 0.1
        W_init = np.random.randn(p*n, target_rank) * 0.1

        def objective(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        error += (val - 1.0)**2
            return error

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})
            if result.fun < 1e-8:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                # Round to nearest half-integer if close
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                # Check if rounded version is still valid
                err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                            err_rounded += (val - 1.0)**2
                if err_rounded < 1e-6:
                    best_rank = target_rank
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    break
        except Exception:
            pass

    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
