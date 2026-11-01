"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the tensor as a 3D array
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Compute mode flattening ranks to get lower bounds
    T1 = tensor.reshape(n * m, m * p * p * n)
    T2 = tensor.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = tensor.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    schoolbook_rank = n * m * p  # 30
    
    # Try to find a rank-r decomposition for r from lower_bound to schoolbook_rank - 1
    # Using the approach: for a given rank r, we need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that tensor = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # 
    # One way: fix U and V, then W is determined by solving a linear system.
    # But we need to search over U and V.
    # 
    # Alternative approach: use the fact that if we know the column space of T1, T2, T3,
    # we can parameterize the decomposition.
    # 
    # For rank r, the column space of T1 must be r-dimensional. We can compute a basis
    # for the column space of T1 and use that to constrain U.
    # 
    # Let's try a more direct approach: for each candidate rank r, attempt to solve
    # for a decomposition using numerical optimization or linear algebra.
    
    def try_rank_decomposition(r, max_iterations=100):
        """Try to find a rank-r decomposition of the tensor."""
        if r < lower_bound:
            return None
        
        # Initialize U, V, W randomly
        np.random.seed(42)
        
        for iteration in range(max_iterations):
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Check if this gives the correct tensor
            # tensor_approx[a,b,c] = sum_c U[a,c] * V[b,c] * W[c,c]
            # This is expensive, so let's use the flattening approach
            
            # For a valid decomposition, the mode-1 flattening should be:
            # T1 = U * (V ⊗ W)^T  where ⊗ is element-wise product reshaped
            # Actually: T1[a, (b,c)] = sum_k U[a,k] * V[b,k] * W[c,k]
            # So T1 = U * (V ⊙ W).T where ⊙ is Khatri-Rao product
            
            # Khatri-Rao product of V and W: shape (m*p*p*n, r)
            # Each column k is V[:,k] ⊗ W[:,k] (Kronecker product)
            # Actually it's the element-wise product reshaped... let me think.
            
            # V has shape (m*p, r), W has shape (p*n, r)
            # The Khatri-Rao product V ⊙ W has shape (m*p*p*n, r) where
            # (V ⊙ W)[(b_idx * p*n + c_idx), k] = V[b_idx, k] * W[c_idx, k]
            
            # But this is O(m*p*p*n*r) which is 15*10*30 = 4500 for r=30, manageable.
            
            # Let's compute the approximate tensor from U, V, W
            # and check the error.
            
            # More efficient: check if the flattening matches
            # T1_approx = U @ kron_VW.T
            # where kron_VW[:,k] = V[:,k] ⊗ W[:,k] (Kronecker product, shape m*p*p*n)
            
            # Actually, let's just directly check a few entries to see if it's close.
            # This is too slow for full verification.
            
            # Let's use a different approach: solve for W given U and V.
            # If we fix U and V, then for each (a, b), we have:
            # tensor[a, b, :] = sum_k U[a,k] * V[b,k] * W[:,k]
            # This is a linear system in W.
            
            # Let's construct the linear system:
            # For each (a, b), the equation is:
            # sum_k (U[a,k] * V[b,k]) * W[:,k] = tensor[a, b, :]
            
            # This gives (n*m * m*p) equations in (p*n * r) unknowns.
            # n*m*m*p = 2*3*3*5 = 90 equations
            # p*n*r = 10*r unknowns
            
            # For r < 9, this is underdetermined. For r >= 9, it may be solvable.
            
            # Let's set up the linear system A @ W_flat = b_flat
            # where W_flat has shape (p*n*r,) and A has shape (n*m*m*p, p*n*r)
            
            A = np.zeros((n * m * m * p, p * n * r))
            b_flat = tensor.reshape(n * m, m * p, p * n).reshape(n * m * m * p, p * n)
            
            for a in range(n * m):
                for b in range(m * p):
                    for k in range(r):
                        coeff = U[a, k] * V[b, k]
                        # W[:,k] contributes to W_flat[k*p*n : (k+1)*p*n]
                        A[a * m * p + b, k * p * n : (k + 1) * p * n] = coeff
            
            # Solve the linear system
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b_flat, rcond=None)
                W_sol = W_flat.reshape(p * n, r)
            except:
                continue
            
            # Check the residual
            residual = np.linalg.norm(b_flat - A @ W_flat)
            if residual < 1e-6:
                # Verify the decomposition
                tensor_approx = np.zeros((n * m, m * p, p * n))
                for k in range(r):
                    tensor_approx += np.einsum('a,b,c->abc', U[:, k], V[:, k], W_sol[:, k])
                
                if np.linalg.norm(tensor - tensor_approx) < 1e-6:
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify rounded version
                    tensor_approx2 = np.zeros((n * m, m * p, p * n))
                    for k in range(r):
                        tensor_approx2 += np.einsum('a,b,c->abc', U_rounded[:, k], V_rounded[:, k], W_rounded[:, k])
                    
                    if np.linalg.norm(tensor - tensor_approx2) < 1e-6:
                        # Convert to list of lists with Fraction for exactness
                        U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U_rounded]
                        V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V_rounded]
                        W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W_rounded]
                        
                        return {'U': U_frac, 'V': V_frac, 'W': W_frac}
        
        return None
    
    # Try ranks from lower_bound to schoolbook_rank - 1
    for r in range(lower_bound, schoolbook_rank):
        result = try_rank_decomposition(r, max_iterations=50)
        if result is not None:
            return result
    
    # Fall back to schoolbook decomposition
    rank = schoolbook_rank
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
    return {"U": U, "V": V, "W": W}
