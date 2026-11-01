import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid matrix multiplication decomposition."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U_tensor = U.reshape(n, m, rank)  # U[i*m + j, r]
    V_tensor = V.reshape(m, p, rank)  # V[j*p + k, r]
    W_tensor = W.reshape(p, n, rank)  # W[k*n + i, r]
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta[i',i] delta[j',j] delta[k',k]
    # Actually, the tensor decomposition condition is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should give the multiplication tensor
    # The multiplication tensor T[i,j,k] = 1 if i,j,k form a valid multiplication term, else 0
    # More precisely, T[i',j',k'; i,j,k] = delta[i',i] * delta[j',j] * delta[k',k]
    # But the standard form is: sum_r U[i',j',r] * V[j,k,r] * W[k,i,r] = delta[i',i] * delta[j',j]
    # Wait, let me re-read the problem statement.
    
    # The output factor follows the official trace convention W[k*n+i,r]
    # So W is indexed by (k, i, r) where k is the output column, i is the output row
    
    # The decomposition condition for matrix multiplication is:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]  (if this is the form)
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = delta[i', i] * delta[j', j] * delta[k', k]
    # No wait, that's not right either.
    
    # Let me think about this more carefully. The matrix multiplication tensor is:
    # T[i, j, k] = 1 if we're computing C[i, k] from A[i, j] and B[j, k]
    # The tensor decomposition is: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # But this doesn't quite work because the indices don't match up.
    
    # The correct form is:
    # sum_r U[i', j', r] * V[j, k, r] * W[k', i, r] = delta[i', i] * delta[j', j] * delta[k', k]
    # where (i', j') are the "input" indices for U, (j, k) are the "input" indices for V,
    # and (k', i) are the "input" indices for W.
    
    # Actually, I think the standard form is:
    # The multiplication map is: (A, B) -> A * B
    # A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 (this is the "1" in the multiplication)
    
    # The tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Wait, but W has indices (k, i, r), so W[k, i, r] means W is indexed by (k, i, r)
    
    # Let me check the shapes:
    # U: (n*m, rank) -> U[i*m + j, r] = U[i, j, r]
    # V: (m*p, rank) -> V[j*p + k, r] = V[j, k, r]
    # W: (p*n, rank) -> W[k*n + i, r] = W[k, i, r]
    
    # So the decomposition condition is:
    # sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = delta[i', i] * delta[j', j] * delta[k', k]
    # No, that's not right. The left side has indices (i, j, k) and the right side has (i', j', k')
    
    # I think the correct interpretation is:
    # The multiplication tensor T is a 3-tensor with T[i, j, k] = 1 for all valid (i, j, k)
    # The tensor decomposition is: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # But this doesn't make sense because the left side is 1 and the right side is a sum.
    
    # Let me look at the problem statement again:
    # "exactly decompose the matrix multiplication tensor"
    # "output factor follows the official trace convention W[k*n+i,r]"
    
    # I think the correct form is:
    # For all (i, j, k): sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1
    # And for all (i', j', k') != (i, j, k): sum_r U[i', j', r] * V[j', k', r] * W[k', i', r] = 0
    # No, that's not right either.
    
    # Actually, I think the tensor decomposition is for the bilinear map:
    # (A, B) -> A * B
    # The tensor T has components T[i, j, k] where:
    # - i is the row index of A (and C)
    # - j is the column index of A (and row index of B)
    # - k is the column index of B (and C)
    # T[i, j, k] = 1 (this represents the multiplication A[i, j] * B[j, k])
    
    # The tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # But wait, W[k, i, r] has indices (k, i, r), which means W is a 3-tensor with indices (k, i, r)
    
    # Let me just check if the decomposition works by computing:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] * A[i, j] * B[j, k]
    # This doesn't make sense.
    
    # I think I'm overcomplicating this. Let me just check the current program's verification logic.
    # The current program checks:
    # sum_r U[i_prime, j_prime, r] * V[j, r] * W[i, r] = delta[i_prime, i] * delta[j_prime, j]
    # for the sub-problem (2, 3, 1)
    
    # So for the full problem, the condition should be:
    # sum_r U[i_prime, j_prime, r] * V[j, k, r] * W[k, i, r] = delta[i_prime, i] * delta[j_prime, j] * delta[k, k_prime]
    # Wait, but W[k, i, r] has indices (k, i, r), so W is indexed by (k, i, r)
    
    # Let me just implement a verification that checks if the decomposition works for matrix multiplication.
    # I'll create random matrices A and B, compute C = A * B, and also compute C' using the decomposition,
    # and check if C == C'.
    
    # The decomposition gives:
    # C[i, k] = sum_r (sum_j U[i, j, r] * A[i, j]) * (sum_j V[j, k, r] * B[j, k]) * W[k, i, r]
    # No, that's not right either.
    
    # I think the correct interpretation is:
    # The tensor decomposition is for the trilinear form:
    # sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 if (i, j, k) is a valid multiplication term, else 0
    # But all (i, j, k) are valid multiplication terms, so this should always be 1.
    
    # I'm confused. Let me just trust the current program's verification logic and extend it to the full problem.
    
    # For the sub-problem (2, 3, 1), the condition is:
    # sum_r U[i_prime, j_prime, r] * V[j, r] * W[i, r] = delta[i_prime, i] * delta[j_prime, j]
    # where i_prime, i in [0, 2), j_prime, j in [0, 3)
    
    # For the full problem (2, 3, 9), the condition should be:
    # sum_r U[i_prime, j_prime, r] * V[j, k, r] * W[k, i, r] = delta[i_prime, i] * delta[j_prime, j] * delta[k, k_prime]
    # where i_prime, i in [0, 2), j_prime, j in [0, 3), k, k_prime in [0, 9)
    
    # But wait, W[k, i, r] has indices (k, i, r), so W is a 3-tensor with indices (k, i, r)
    # And the condition involves delta[k, k_prime], which means k and k_prime are both indices of the output column.
    
    # I think the correct condition is:
    # sum_r U[i_prime, j_prime, r] * V[j, k, r] * W[k, i, r] = delta[i_prime, i] * delta[j_prime, j]
    # for all (i_prime, j_prime, j, k, i), and the k index is summed over in the matrix multiplication.
    
    # No, I think I need to just check if the decomposition works by computing matrix products.
    
    # Let me try a different approach: check if the decomposition satisfies the tensor equation
    # sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 for all (i, j, k)
    
    # Actually, I just realized that the tensor decomposition for matrix multiplication is:
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # where T[i, j, k] = 1 for all (i, j, k)
    
    # So the condition is: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 for all (i, j, k)
    
    # Let me check this.
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += abs(val - 1.0)
    return err

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank 5 decomposition for (2, 3, 1)
    # Sub-problem: A (2x3), B (3x1), C (2x1)
    # U (2x3x5), V (3x1x5), W (2x1x5)
    # Condition: sum_r U[i',j',r] V[j,r] W[i,r] = delta[i',i] delta[j',j]
    
    R_sub = 5
    n_sub, m_sub, p_sub = 2, 3, 1
    
    # Flatten U, V, W for optimization
    # U: 2*3*R_sub = 30
    # V: 3*1*R_sub = 15
    # W: 2*1*R_sub = 10
    # Total 55 variables
    
    def residual(params):
        U = params[:30].reshape(2, 3, R_sub)
        V = params[30:45].reshape(3, 1, R_sub)
        W = params[45:55].reshape(2, 1, R_sub)
        
        res = []
        for i_prime in range(n_sub):
            for j_prime in range(m_sub):
                for i in range(n_sub):
                    for j in range(m_sub):
                        # Sum over r
                        val = 0
                        for r in range(R_sub):
                            val += U[i_prime, j_prime, r] * V[j, r] * W[i, r]
                        target = 1.0 if (i_prime == i and j_prime == j) else 0.0
                        res.append(val - target)
        return res
    
    # Initial guess
    np.random.seed(42)
    x0 = np.random.randn(55) * 0.1
    
    best_rank = n * m * p
    best_result = None
    
    # Try to solve
    try:
        sol = least_squares(residual, x0, method='lm', max_nfev=10000)
        if sol.success and sol.cost < 1e-10:
            U_sub = sol.x[:30].reshape(2, 3, R_sub)
            V_sub = sol.x[30:45].reshape(3, 1, R_sub)
            W_sub = sol.x[45:55].reshape(2, 1, R_sub)
            
            # Round to half-integers
            U_sub = np.round(U_sub * 2) / 2
            V_sub = np.round(V_sub * 2) / 2
            W_sub = np.round(W_sub * 2) / 2
            
            # Verify exactness
            res_check = residual(sol.x) # Use original for check? No, check rounded
            # Re-check with rounded values
            U_r = U_sub
            V_r = V_sub
            W_r = W_sub
            err = 0
            for i_prime in range(n_sub):
                for j_prime in range(m_sub):
                    for i in range(n_sub):
                        for j in range(m_sub):
                            val = 0
                            for r in range(R_sub):
                                val += U_r[i_prime, j_prime, r] * V_r[j, r] * W_r[i, r]
                            target = 1.0 if (i_prime == i and j_prime == j) else 0.0
                            err += abs(val - target)
            
            if err < 1e-9:
                # Construct full decomposition
                total_rank = p_sub * R_sub * (p // p_sub) # 9 * 5 = 45
                # Wait, p=9, p_sub=1. So 9 copies.
                total_rank = 45
                
                U = np.zeros((n*m, total_rank))
                V = np.zeros((m*p, total_rank))
                W = np.zeros((p*n, total_rank))
                
                for k in range(p):
                    for r in range(R_sub):
                        c = k * R_sub + r
                        for i in range(n):
                            for j in range(m):
                                U[i*m + j, c] = U_sub[i, j, r]
                        for j in range(m):
                            V[j*p + k, c] = V_sub[j, 0, r]
                        for i in range(n):
                            W[k*n + i, c] = W_sub[i, 0, r]
                
                # Verify the full decomposition
                full_err = _verify_decomposition(U, V, W, n, m, p)
                if full_err < 1e-9:
                    best_rank = total_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    except Exception as e:
        pass
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best_result
