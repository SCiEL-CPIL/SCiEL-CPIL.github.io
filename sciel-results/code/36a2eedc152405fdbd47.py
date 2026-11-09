"""Program to find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_elements = nm * mp * pn
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we're multiplying A[i, j] by B[j, k] to get C[i, k]
    # Actually, the tensor T is such that C[i, k] = sum_{j=1}^{m} A[i, j] * B[j, k]
    # The rank-1 decomposition corresponds to finding tensors U_c, V_c, W_c such that
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is n x m, V_c is m x p, W_c is p x n
    # In the flattened interface:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # with W indexed by W[k*n + i, r]
    
    # The tensor T[i, j, k] = delta_{j, j'} where we have A[i, j] * B[j, k]
    # More precisely, T is a 3-way tensor with T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, for matrix multiplication, the tensor is:
    # T[i, j, k] = 1 (for all i, j, k) but structured such that
    # The multiplication tensor has entries T[i, j, k] = 1 where i indexes rows of A/cols of C,
    # j indexes cols of A/rows of B, k indexes cols of B/rows of C... 
    # Wait, let me reconsider.
    
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T is of size n x m x p with T[i, j, k] = 1 for all i, j, k
    # This is because each term A[i,j]*B[j,k] appears in the sum for C[i,k].
    
    # So T is simply an all-ones tensor of shape (n, m, p).
    # But that's not quite right either. Let me think again.
    
    # Actually, the matrix multiplication tensor T_{mult} is defined by:
    # T_{i, j, k} = 1 for all i in [1,n], j in [1,m], k in [1,p]
    # This is because the bilinear form is sum_{i,j,k} T_{i,j,k} x_i y_j z_k
    # where x_i = A[i,j]... no.
    
    # Let me use the standard definition. The tensor for multiplying (n x m) by (m x p)
    # has dimensions n x m x p and entry T[i, j, k] = 1 for all i, j, k.
    # The reason is that the product C[i,k] = sum_j A[i,j] B[j,k],
    # so the coefficient of A[i,j] B[j,k] in the expression for C[i,k] is 1.
    
    # So T is an all-ones tensor of shape (n, m, p).
    
    T = np.ones((n, m, p))
    
    # Now we want to find a rank-r decomposition:
    # T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is shape (n,), v_c is shape (m,), w_c is shape (p,)
    
    # In the interface:
    # U is (n*m, rank): U[i*m + j, c] = u_c[i] * v_c[j]... no.
    # Let me re-read the interface.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # with W indexed by W[k*n + i, r]
    
    # For a rank-1 term c:
    # U[:, c] is a vector of length n*m, V[:, c] is a vector of length m*p, W[:, c] is a vector of length p*n
    # The contribution to the tensor is:
    # For each i, j, k: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should sum to T[i, j, k] = 1
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    # This is equivalent to saying that the tensor T can be decomposed into rank-1 tensors
    # where the rank-1 tensor for column c is the outer product of three vectors:
    # u_c of length n*m (but structured as n x m), v_c of length m*p (structured as m x p),
    # w_c of length p*n (structured as p x n)
    
    # Actually, the standard way is:
    # U is n x m x rank, V is m x p x rank, W is p x n x rank
    # Then T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Wait, W is p x n, so W[k, i] where k indexes rows of W (size p) and i indexes cols of W (size n)
    # But the problem says W[k*n + i, r], so W is stored as (p*n, rank) with row index k*n + i.
    # So W[k*n + i, c] = W_flat[c, k*n + i] if we think of it as (rank, p*n), or W_flat[k*n + i, c] as (p*n, rank).
    
    # Let me just work with the flat arrays as given.
    
    # For rank 1 decomposition with vectors:
    # Let a_c be a vector of length n*m (flattened n x m matrix)
    # Let b_c be a vector of length m*p (flattened m x p matrix)  
    # Let d_c be a vector of length p*n (flattened p x n matrix, with d_c[k*n + i])
    # Then we need: sum_c a_c[i*m + j] * b_c[j*p + k] * d_c[k*n + i] = 1 for all i, j, k
    
    # This is a system of equations. For small n, m, p, we can try to find low-rank solutions.
    
    # Known lower bounds:
    # For 2x3 x 3x5, the rank is at least max(nm, mp, pn) = max(6, 15, 10) = 15? No, that's not right.
    # The rank is at least max(n*p, n*m, m*p) for some bounds, but actually the rank of the 
    # matrix multiplication tensor is at least the maximum of the sizes of the three matricizations.
    
    # The three matricizations of T (shape n x m x p) are:
    # T_{(nm) x p}: reshape T to (n*m, p), rank is at most min(nm, p)
    # T_{(mp) x n}: reshape T to (m*p, n), rank is at most min(mp, n)
    # T_{(pn) x m}: reshape T to (p*n, m), rank is at most min(pn, m)
    
    # But the tensor rank is at least the rank of any matricization.
    # Since T is all ones:
    # T_{(nm) x p} is an (n*m) x p all-ones matrix, which has rank 1.
    # Similarly for the others. So this lower bound is only 1, which is weak.
    
    # Better lower bounds exist, but let's just try to find a good decomposition.
    
    # The schoolbook rank is n*m*p = 2*3*5 = 30.
    
    # Let's try to find a rank less than 30.
    
    # One approach: use random initialization and gradient descent, or use a more structured approach.
    
    # Given the time constraint, let's try a few different approaches:
    # 1. Try rank 29, 28, ... down to some minimum, and check if we can find a solution.
    # 2. Use a structured decomposition.
    
    # For now, let's return the schoolbook decomposition as a fallback,
    # but first try to find something better.
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using a structured approach.
    # We'll use the fact that the tensor is all-ones and try to exploit symmetry.
    
    # Let's try rank 28 first.
    # We can use a simple iterative method: initialize U, V, W randomly or with structure,
    # then use alternating least squares or similar.
    
    # Given the time limit, let's just try a few fixed ranks with a simple solver.
    
    def try_rank(rank):
        """Try to find a rank decomposition. Returns (U, V, W) or None."""
        if time.time() > deadline:
            return None
        
        # Initialize U, V, W with small random values
        np.random.seed(42)
        U_mat = np.random.randn(n * m, rank) * 0.1
        V_mat = np.random.randn(m * p, rank) * 0.1
        W_mat = np.random.randn(p * n, rank) * 0.1
        
        # Use alternating minimization
        for iteration in range(100):
            if time.time() > deadline:
                return None
            
            # Update U: for each i, j, and rank c, minimize
            # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
            # This is linear in U when V and W are fixed.
            # For fixed j, k, i: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
            # So U[i*m+j, :] * diag(V[j*p+k, :] * W[k*n+i, :]) ... this is getting complex.
            
            # Let's use a different approach: treat it as a linear system.
            # For each (i, j, k), we have one equation.
            # The unknowns are U[:, :], V[:, :], W[:, :].
            # This is bilinear/quadratic, so we use alternating least squares.
            
            # Update U: for each column c and row (i*m+j), we want to minimize
            # sum_{i,j,k} (sum_c' U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] - 1)^2
            # This is complex. Let's use a simpler heuristic.
            
            # Actually, let's just check if the current decomposition is good.
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                        error += (val - 1.0) ** 2
            
            if error < 1e-6:
                # Convert to list format
                U_list = [[U_mat[i, c] for c in range(rank)] for i in range(n*m)]
                V_list = [[V_mat[i, c] for c in range(rank)] for i in range(m*p)]
                W_list = [[W_mat[i, c] for c in range(rank)] for i in range(p*n)]
                return U_list, V_list, W_list
            
            # Simple update: adjust U to reduce error
            # For each i, j, c: U[i*m+j, c] += alpha * sum_k V[j*p+k, c] * W[k*n+i, c] * (1 - current_val)
            # This is a gradient step.
            alpha = 0.1
            for i in range(n):
                for j in range(m):
                    for c in range(rank):
                        grad = 0.0
                        for k in range(p):
                            val = 0.0
                            for c2 in range(rank):
                                val += U_mat[i*m+j, c2] * V_mat[j*p+k, c2] * W_mat[k*n+i, c2]
                            residual = 1.0 - val
                            grad += V_mat[j*p+k, c] * W_mat[k*n+i, c] * residual
                        U_mat[i*m+j, c] += alpha * grad
            
            if time.time() > deadline:
                return None
        
        return None
    
    # Try ranks from 29 down to 25
    for rank in range(29, 24, -1):
        result = try_rank(rank)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
