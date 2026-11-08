"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Start with the schoolbook decomposition as a baseline
    rank_schoolbook = n * m * p  # 128
    
    # The known lower bound for matrix multiplication A(4x4) * B(4x8)
    # is at least max(4*4, 4*8, 4*8) = 32, but the exact minimum is unknown.
    # However, for 4x4 * 4x8, we can potentially do better than 128.
    
    # Let's try to find a better decomposition using randomized methods.
    # We'll use alternating least squares to minimize the rank.
    
    # First, let's try a few different ranks and see if we can find an exact decomposition.
    # Start with a rank that's lower than schoolbook.
    
    # Known results: For 4x4 * 4x4, the best known is 49 (Larsen et al.) or lower.
    # For 4x4 * 4x8, we can potentially use similar techniques.
    
    # Let's try a randomized approach:
    # 1. Initialize U, V, W with random values at a given rank r
    # 2. Alternate between solving for U, V, W given the others
    # 3. Check if the decomposition is exact
    
    # The tensor T[i,j,k] = delta for the multiplication tensor.
    # For A(4x4) * B(4x8), the output C(4x8) has entries:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has indices (i, j, k) where i in [0,4), j in [0,4), k in [0,8)
    # T[i,j,k] = 1 if this term contributes to C[i,k], else 0.
    # Actually, T[i,j,k] = 1 for all valid (i,j,k) since each product A[i,j]*B[j,k]
    # contributes to C[i,k].
    
    # So the tensor is just the outer product structure.
    # The decomposition is: C[i,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
    # Wait, that's not quite right. Let me re-read the convention.
    
    # U has shape (n*m, rank) = (16, rank), indexed by (i, j) -> i*m + j
    # V has shape (m*p, rank) = (32, rank), indexed by (j, k) -> j*p + k
    # W has shape (p*n, rank) = (32, rank), indexed by (k, i) -> k*n + i
    
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k]
    # where C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j] * B[j,k]
    
    # This is equivalent to: the tensor T[i,j,k] with T[i,j,k] = 1 (for all i,j,k)
    # being decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, that's not right either. The LHS should equal A[i,j]*B[j,k], which is
    # the product of the (i,j) entry of A and the (j,k) entry of B.
    
    # So the tensor we're decomposing is the "multiplication tensor" which has
    # T[i,j,k] = 1 for all valid (i,j,k) indices, because each pair (A[i,j], B[j,k])
    # contributes to the output.
    
    # Actually, I think I'm overcomplicating this. Let me think about it differently.
    
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A has 16 entries (4x4), B has 32 entries (4x8), C has 32 entries (4x8).
    
    # The tensor T has components T[i,j,k] where i indexes A rows, j indexes A cols/B rows,
    # k indexes B cols/C cols. T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,8).
    
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need to find rank-r matrices U (16 x r), V (32 x r), W (32 x r) such that
    # for all i in [0,4), j in [0,4), k in [0,8):
    #   sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] = 1
    
    # This is a system of 128 equations (4*4*8 = 128) with 16*r + 32*r + 32*r = 80*r unknowns.
    
    # For r = 128 (schoolbook), we have 10240 unknowns and 128 equations, so it's
    # very underdetermined. The schoolbook solution sets U[i*4+j, i*4+j*8+k] = 1, etc.
    
    # For smaller r, we have fewer unknowns. We need 80*r >= 128, so r >= 2 (since 80*1 = 80 < 128).
    # But the actual minimum rank is likely much higher.
    
    # Let's try a randomized approach:
    # 1. For a given rank r, initialize U, V, W with random values.
    # 2. Use alternating least squares to minimize the error.
    # 3. Check if the error is close to zero (exact decomposition).
    
    # However, this is a non-convex problem, so we might get stuck in local minima.
    
    # Let's try a few ranks starting from a reasonable lower bound.
    
    # The trivial lower bound is max(16, 32, 32) = 32, but this is not tight.
    # For 4x4 * 4x4, the best known is around 49-54. For 4x4 * 4x8, it might be similar.
    
    # Let's try ranks from 32 to 128 and see if we can find an exact decomposition.
    
    # Actually, let's be more systematic. We'll use a randomized tensor rank minimization
    # algorithm.
    
    # Strategy:
    # 1. Start with a rank r (e.g., r = 64 or 80).
    # 2. Initialize U, V, W with random Gaussian values.
    # 3. Alternate between:
    #    - Fix V, W, solve for U (linear least squares)
    #    - Fix U, W, solve for V (linear least squares)
    #    - Fix U, V, solve for W (linear least squares)
    # 4. After convergence, check if the decomposition is exact (error < tolerance).
    # 5. If exact, try to reduce the rank.
    
    # Let's implement this.
    
    def try_rank(r, num_restarts=10, max_iters=100, tol=1e-10):
        """Try to find an exact decomposition of rank r."""
        np.random.seed(42)  # For reproducibility
        
        for restart in range(num_restarts):
            # Initialize U, V, W with random values
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_iters):
                # Solve for U: for each (i,j), U[i*m+j, :] = argmin sum_k (1 - sum_r V[j*p+k, r] * W[k*n+i, r] * U[i*m+j, r])^2
                # This is a linear system in U[i*m+j, :].
                # Let's vectorize: for fixed (i,j), the equation is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p:, :] * W[:, i])^T = ones(p)
                # Wait, that's not quite right. Let me re-derive.
                
                # For fixed (i,j), we have p equations (one for each k):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let u = U[i*m+j, :] (a vector of length r)
                # Let M be a p x r matrix where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then the equation is: u @ M^T = ones(p), i.e., M @ u = ones(p)
                # So we solve M @ u = ones(p) for u.
                
                # Similarly for V and W.
                
                # Let's implement this.
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r] for k in [0, p)
                        M = V[j*p:(j+1)*p, :] * W[::n, i]  # Wait, W[k*n+i, r] for k in [0, p)
                        # W has shape (p*n, r), so W[k*n+i, r] for k in [0, p) is W[0*n+i, :], W[1*n+i, :], ..., W[(p-1)*n+i, :]
                        # This is W[i::n, :] (every n-th row starting from i)
                        M = V[j*p:(j+1)*p, :] * W[i::n, :]  # Shape: (p, r)
                        # Solve M @ u = ones(p)
                        # This is an underdetermined system (r >= p typically), so we use least squares.
                        u, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        # M[i, r] = U[i*m+j, r] * W[k*n+i, r] for i in [0, n)
                        M = U[j::m, :] * W[k*n:(k+1)*n, :]  # Wait, U[i*m+j, r] for i in [0, n)
                        # U has shape (n*m, r), so U[i*m+j, r] for i in [0, n) is U[j, :], U[m+j, :], ..., U[(n-1)*m+j, :]
                        # This is U[j::m, :] (every m-th row starting from j)
                        M = U[j::m, :] * W[k*n:(k+1)*n, :]  # Shape: (n, r)
                        # Solve M @ v = ones(n)
                        v, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        # M[j, r] = U[i*m+j, r] * V[j*p+k, r] for j in [0, m)
                        M = U[i*m:(i+1)*m, :] * V[::p, k]  # Wait, V[j*p+k, r] for j in [0, m)
                        # V has shape (m*p, r), so V[j*p+k, r] for j in [0, m) is V[k, :], V[p+k, :], ..., V[(m-1)*p+k, :]
                        # This is V[k::p, :] (every p-th row starting from k)
                        M = U[i*m:(i+1)*m, :] * V[k::p, :]  # Shape: (m, r)
                        # Solve M @ w = ones(m)
                        w, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                
                # Check convergence: compute the error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1) ** 2
                error = np.sqrt(error / (n * m * p))
                
                if error < tol:
                    # Found an exact decomposition
                    return U, V, W, error
        
        return None, None, None, None
    
    # Try different ranks
    best_rank = rank_schoolbook
    best_U, best_V, best_W = None, None, None
    
    # Start with a rank that's likely to work, e.g., 80 or 100
    for r in [80, 90, 100, 110, 120, 128]:
        U, V, W, error = try_rank(r, num_restarts=5, max_iters=50, tol=1e-10)
        if U is not None and error < 1e-10:
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # If we didn't find a better rank, fall back to schoolbook
    if best_U is None:
        # Use schoolbook decomposition
        rank = rank_schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
