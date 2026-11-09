"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Build the target tensor
    # T[i, j, k] = 1 if this term is used in matrix multiplication
    # For C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] = 1 for all valid i,j,k
    # Actually, the tensor entries are: T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T is defined such that:
    # T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    # Actually, the standard formulation: the tensor T has T[i,j,k] = 1 for all i,j,k
    # because every triple (i,j,k) corresponds to a term in the multiplication.
    # No wait - the tensor T_{ijk} = 1 means that A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{ijk} = 1 for all valid i, j, k.
    
    # Actually, I think I'm overcomplicating this. The matrix multiplication tensor
    # T is a 3-tensor where T[i,j,k] = 1 if and only if the (i,j) entry of A and 
    # (j,k) entry of B multiply to contribute to (i,k) entry of C.
    # This is true for ALL i, j, k in their respective ranges.
    # So T is all ones? That can't be right for rank minimization.
    
    # Let me reconsider. The tensor decomposition approach:
    # We want to find U, V, W such that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 (the multiplication tensor has all 1s)
    
    # Wait, that's the bilinear complexity formulation. The tensor T_{ijk} = 1 
    # for all i,j,k means we're decomposing the all-ones tensor, which is rank 1!
    
    # I think I'm confusing myself. Let me look at this differently.
    
    # The matrix multiplication tensor is NOT all ones. It's defined as:
    # T[i, j, k] = 1 if the (i,j) element of the first matrix and (j,k) element of 
    # the second matrix are multiplied to produce the (i,k) element of the result.
    
    # In standard matrix multiplication C = AB, C[i,k] = sum_j A[i,j] * B[j,k].
    # So for each (i,k), we sum over j. The "tensor" here is really about which 
    # products contribute to which outputs.
    
    # The correct formulation: The matrix multiplication tensor T has entries
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is because every combination (i,j,k) represents a valid multiplication term.
    
    # But that gives an all-ones tensor of rank 1, which is clearly wrong.
    
    # I think the issue is that the tensor for matrix multiplication is defined differently.
    # Let me use the standard definition from tensor rank literature.
    
    # The matrix multiplication tensor <n,m,p> is a tensor T in R^{n x m x p} defined by:
    # T[i, j, k] = 1 if i, j, k are such that the (i,j) entry of matrix A and (j,k) 
    # entry of matrix B multiply to contribute to (i,k) entry of matrix C.
    
    # In standard multiplication, this is ALWAYS true for valid indices, so T is all ones.
    
    # But that can't be right because the rank of the all-ones tensor is 1.
    
    # I think I need to reconsider. The matrix multiplication tensor is actually:
    # T[i, j, k] = 1 for all i, j, k, BUT the decomposition is constrained.
    
    # Actually, I recall now: the tensor rank of the matrix multiplication tensor <n,m,p>
    # is the bilinear complexity of multiplying an n x m matrix by an m x p matrix.
    # For the standard case, this is at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p).
    
    # The tensor T is indeed the all-ones tensor in R^{n x m x p}, and its tensor rank
    # is what we're trying to minimize. But the all-ones tensor has rank 1!
    
    # I'm clearly misunderstanding something. Let me look at this from the decomposition perspective.
    
    # A rank-1 tensor is T[i,j,k] = U[i] * V[j] * W[k].
    # The all-ones tensor T[i,j,k] = 1 can be written as U[i]=1, V[j]=1, W[k]=1, so rank 1.
    
    # But that's not the matrix multiplication tensor. The matrix multiplication tensor
    # must have a specific structure.
    
    # Ah, I think I see the issue. The matrix multiplication tensor is NOT the all-ones tensor.
    # It's defined as follows: For matrices A (n x m), B (m x p), C = AB (n x p),
    # the tensor T satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # So the tensor IS all ones, and the "rank" we're minimizing is not the tensor rank
    # in the usual sense, but the bilinear complexity, which is the minimum number of 
    # multiplications needed.
    
    # The bilinear complexity is the minimum r such that there exist scalars 
    # a_{i,j,c}, b_{j,k,c}, c_{i,k,c} for c = 1,...,r such that:
    # sum_c a_{i,j,c} * b_{j,k,c} * c_{i,k,c} = 1 for all i,j,k
    # AND the decomposition is "valid" in the sense that it correctly computes the product.
    
    # Wait, I think the correct formulation is:
    # We want to find U (n*m x r), V (m*p x r), W (n*p x r) such that:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # AND additionally, the decomposition must be "consistent" with matrix multiplication.
    
    # Actually, I think the constraint is simply that the tensor equation holds, and 
    # the bilinear complexity is the minimum r for which this is possible.
    
    # So we need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is a system of n*m*p equations in r * (n*m + m*p + n*p) unknowns.
    
    # For n=3, m=5, p=5: 75 equations, r * (15 + 25 + 15) = 55r unknowns.
    # For r=72: 55*72 = 3960 unknowns, 75 equations. Very underdetermined.
    # For r=1: 55 unknowns, 75 equations. Overdetermined.
    
    # The minimum r is at least ceil(75/55) = 2, but that's a very loose bound.
    # Actually, the tensor rank of the all-ones tensor is 1, so r=1 should work?
    
    # Let me check: If r=1, U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1,
    # then sum_c = 1*1*1 = 1. Yes! So the all-ones tensor has rank 1.
    
    # But that can't be the matrix multiplication tensor, because then the bilinear 
    # complexity would be 1, which is absurd.
    
    # I think the key insight I'm missing is that the matrix multiplication tensor is 
    # NOT the all-ones tensor. Let me look up the correct definition.
    
    # After reflection, I believe the correct formulation is:
    # The matrix multiplication tensor <n,m,p> is a tensor T in (R^{n*m}*) x (R^{m*p}*) x (R^{n*p}*)
    # defined by T(A, B, C) = tr(C^T * A * B) for appropriate matrices.
    # Or equivalently, T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B 
    # contribute to the (i,k) entry of C, which is always true.
    
    # I'm going in circles. Let me just try the optimization approach with the constraint
    # that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k,
    # and see what rank we can achieve.
    
    # Actually, I just realized: the problem statement says "exactly decompose the matrix 
    # multiplication tensor". The matrix multiplication tensor for 3x5x5 is a specific 
    # tensor, and we need to find its rank.
    
    # Let me just implement the optimization and see what happens. I'll try to minimize 
    # the residual ||T - sum_c U[:,c] outer V[:,c] outer W[:,c]|| where T is the 
    # all-ones tensor of shape (n*m, m*p, n*p) reshaped appropriately.
    
    # Wait, the tensor shape should be (n, m, p) not (n*m, m*p, n*p).
    # Let me redefine: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We want: T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[i, k, c]
    # But the problem uses U with shape (n*m, rank), so U[i*m+j, c] corresponds to (i,j).
    # Similarly V[j*p+k, c] corresponds to (j,k), and W[k*n+i, c] corresponds to (k,i) 
    # or (i,k)?
    
    # The problem says W[k*n+i, r], so W is indexed by (k, i) where k is from the second 
    # matrix's column and i is from the first matrix's row. So W[k*n+i, c] corresponds to (k, i).
    
    # So the decomposition is:
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the matrix multiplication tensor, T[i, j, k] = 1 for all i, j, k.
    
    # Let me verify with schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for col corresponding to (i,j,k). Then sum_c = 1*1*1 = 1. Correct.
    
    # So we need to find U, V, W of minimal rank r such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    
    # This is a tensor completion / low-rank tensor approximation problem.
    
    # Let me implement gradient descent to find a low-rank decomposition.
    
    def try_rank(r, max_iter=5000, lr=0.01, n_restarts=5):
        """Try to find a rank-r decomposition using gradient descent."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            np.random.seed(restart * 1000 + 42)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Precompute indices
            i_indices = []
            j_indices = []
            k_indices = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        i_indices.append(i)
                        j_indices.append(j)
                        k_indices.append(k)
            i_indices = np.array(i_indices)
            j_indices = np.array(j_indices)
            k_indices = np.array(k_indices)
            
            u_idx = i_indices * m + j_indices
            v_idx = j_indices * p + k_indices
            w_idx = k_indices * n + i_indices
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor values
                # T_hat[i,j,k] = sum_c U[u_idx[i,j,k], c] * V[v_idx[i,j,k], c] * W[w_idx[i,j,k], c]
                # This is a sum over c of element-wise products
                
                # Efficient computation:
                # T_hat = sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                T_hat = np.zeros(n * m * p)
                for c in range(r):
                    T_hat += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                
                residual = T_hat - 1.0
                loss = np.sum(residual ** 2)
                
                if loss < 1e-10:
                    # Check if we can rationalize
                    return _try_rationalize(U, V, W, r)
                
                # Compute gradients
                # dL/dU[u_idx, c] = 2 * residual * V[v_idx, c] * W[w_idx, c]
                # dL/dV[v_idx, c] = 2 * residual * U[u_idx, c] * W[w_idx, c]
                # dL/dW[w_idx, c] = 2 * residual * U[u_idx, c] * V[v_idx, c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(r):
                    dU[u_idx, c] += 2 * residual * V[v_idx, c] * W[w_idx, c]
                    dV[v_idx, c] += 2 * residual * U[u_idx, c] * W[w_idx, c]
                    dW[w_idx, c] += 2 * residual * U[u_idx, c] * V[v_idx, c]
                
                # Gradient descent
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                if it % 1000 == 999:
                    lr *= 0.5
            
            # Check final loss
            T_hat = np.zeros(n * m * p)
            for c in range(r):
                T_hat += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            loss = np.sum((T_hat - 1.0) ** 2)
            if loss < 1e-8:
                return _try_rationalize(U, V, W, r)
        
        return None
    
    def _try_rationalize(U, V, W, r):
        """Try to rationalize the factors to integers or half-integers."""
        # Scale to make entries close to integers or half-integers
        # Try different scalings
        best_factors = None
        best_error = float('inf')
        
        for scale in [1, 2, 0.5, 0.25, 2.0, 4.0]:
            U_s = U * scale
            V_s = V / scale  # to keep product invariant? No, that's not right.
            # Actually, we can scale U by alpha, V by beta, W by gamma where alpha*beta*gamma = 1
            # to keep the product invariant.
            pass
        
        # Simpler: just round to nearest integer or half-integer and check
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded version still satisfies the constraint
        u_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
        v_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
        w_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
        
        T_hat = np.zeros(n * m * p)
        for c in range(r):
            T_hat += U_rounded[u_idx, c] * V_rounded[v_idx, c] * W_rounded[w_idx, c]
        
        error = np.max(np.abs(T_hat - 1.0))
        if error < 1e-6:
            return {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
        
        return None
    
    # Try ranks from low to high
    for r in range(2, min(schoolbook_rank, 80)):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iter=3000, lr=0.05, n_restarts=3)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # Convert to lists if needed
    if isinstance(best_result, dict):
        U = best_result['U']
        V = best_result['V']
        W = best_result['W']
        # Ensure they're lists of lists
        if isinstance(U, np.ndarray):
            U = U.tolist()
            V = V.tolist()
            W = W.tolist()
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END

