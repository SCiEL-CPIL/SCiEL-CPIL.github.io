"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin for evaluation
    
    # Build Strassen-like decomposition for 3x3 * 3x8
    # We can use a 3x3 Strassen decomposition (rank 7) for the 3x3 part,
    # then extend to handle the 8 columns.
    
    # Strategy: Use a known low-rank decomposition for 3x3 matrix multiplication
    # Strassen's algorithm gives rank 7 for 2x2, but for 3x3 we need to be more careful.
    
    # Actually, let's use a different approach:
    # For 3x3 * 3x8, we can think of it as 8 separate 3x3 * 3x1 multiplications.
    # But that's not how tensor rank works.
    
    # Let's use the fact that we can decompose the 3x3 matrix multiplication tensor
    # and then replicate it for each of the 8 columns of B.
    
    # A simpler approach: use a Kronecker-like structure.
    # For 3x3 * 3x8, we can view this as multiplying by 8 independent 3x3 matrices
    # (each column of B gives one 3x1 result).
    
    # Actually, the cleanest approach for 3x3 * 3x8:
    # We can use the schoolbook method but try to find a lower rank via structured search.
    
    # Let me try a direct approach: construct a candidate decomposition and verify it.
    
    # For 3x3 * 3x8, one approach is to use a "block" structure.
    # Think of A as 3x3 and B as 3x8. We can partition B into two 3x4 blocks.
    # Then we need to compute A*B1 and A*B2 where B1, B2 are 3x4.
    
    # For each 3x3 * 3x4 multiplication, we can use a structured decomposition.
    
    # Let's try a different strategy: use the known rank bounds.
    # For m x n * n x p, the rank is at most mn + np + mp - n (Landsberg's bound is different).
    # Actually, the best known upper bound for 3x3 * 3x3 is 23 (Landsberg), but we don't know for 3x3 * 3x8.
    
    # Let's just do a practical search: try to find a decomposition with rank < 72.
    
    # First, let's construct a simple structured decomposition:
    # Use the fact that we can share computation across the 8 columns.
    
    # For each row i of A (i=0,1,2) and each column j of A (j=0,1,2),
    # we have A[i,j] which multiplies with each row j of B (B[j,k] for k=0..7).
    
    # The schoolbook uses 3*3*8 = 72 terms.
    
    # Can we do better? Let's try a "factored" approach:
    # For each j (column of A / row of B), compute S_j = sum_i A[i,j] * e_i (a 3-vector)
    # and T_j = sum_k B[j,k] * e_k (an 8-vector)
    # Then C = sum_j outer(S_j, T_j)
    
    # This is just the schoolbook written differently, still rank 72.
    
    # Let's try to use a low-rank factorization of the identity-like structure.
    
    # Actually, for the tensor T[i,j,k] = 1 for all i,j,k, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a 3rd-order tensor equation. Let's try to solve it.
    
    # For the all-ones tensor, one solution is U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
    # which gives rank 1. But this doesn't work for matrix multiplication!
    
    # I think I've been confusing two different problems. Let me re-read the task.
    
    # The task says: "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 8 matrix."
    # The interface is: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And the decomposition should satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Wait, that IS the all-ones tensor! So rank 1 should work?
    
    # Let me verify with the actual matrix multiplication:
    # If U[i*m+j,r] = 1, V[j*p+k,r] = 1, W[k*n+i,r] = 1 for r=0,
    # then for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the decomposition: C[i,k] = sum_r sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
    # = sum_j 1 * A[i,j] * 1 * B[j,k] * 1 = sum_j A[i,j] * B[j,k] ✓
    
    # So rank 1 actually works?! Let me double-check the indexing.
    
    # The output factor follows W[k*n+i, r]. So W is indexed by (k, i) pairs.
    # For the all-ones decomposition:
    # U[i*m+j, 0] = 1 for all i, j
    # V[j*p+k, 0] = 1 for all j, k  
    # W[k*n+i, 0] = 1 for all k, i
    
    # Then: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1*1*1 = 1 ✓
    
    # This seems too good to be true. Let me verify with a concrete example.
    
    # A = [[1,2,3],[4,5,6],[7,8,9]], B = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]]
    # C = A * B = [[1,2,3,0,0,0,0,0],[4,5,6,0,0,0,0,0],[7,8,9,0,0,0,0,0]]
    
    # Using rank-1 decomposition:
    # C[i,k] = sum_j U[i*m+j,0] * A[i,j] * V[j*p+k,0] * B[j,k] * W[k*n+i,0]
    # = sum_j 1 * A[i,j] * 1 * B[j,k] * 1
    # = sum_j A[i,j] * B[j,k] ✓
    
    # So rank 1 works! But this contradicts the known lower bounds for matrix multiplication.
    
    # I think the issue is that the "rank" in the tensor decomposition sense is different
    # from the "arithmetic complexity" (number of multiplications).
    
    # The tensor rank of the matrix multiplication tensor T_{m,n,p} is indeed the minimum
    # number of scalar multiplications needed. For 3x3 * 3x3, it's at least 19 (Landsberg).
    
    # But wait, if the all-ones tensor has rank 1, then the matrix multiplication tensor
    # should also have rank 1, which is impossible.
    
    # I must be making an error. Let me look at this more carefully.
    
    # The matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k.
    # The rank of this tensor as a 3rd-order tensor is the minimum r such that
    # T[i,j,k] = sum_{r'=1}^r U[i,r'] * V[j,r'] * W[k,r']
    
    # For the all-ones tensor, U[i,1]=1, V[j,1]=1, W[k,1]=1 gives rank 1.
    
    # But this is the tensor for scalar multiplication, not matrix multiplication!
    
    # For matrix multiplication, the tensor has a more complex structure.
    # The correct tensor is: T[(i,j), (j,k), (i,k)] = 1 if the indices match, 0 otherwise.
    
    # In our case, U is indexed by (i,j), V by (j,k), W by (k,i) [with the given convention].
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k, and we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This IS the all-ones tensor, and it has rank 1!
    
    # I think the confusion is that the "rank" in the context of matrix multiplication
    # tensor decompositions refers to something different. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition" - this should be the tensor rank.
    # If the tensor is all-ones, then rank 1 is correct.
    
    # But this seems wrong because it would mean we can multiply any matrices with just 1
    # scalar multiplication, which is impossible.
    
    # I think the key insight I'm missing is that the U, V, W matrices must have a specific
    # structure that corresponds to actual linear combinations of the input entries.
    
    # In a proper matrix multiplication algorithm, each "term" in the sum corresponds to
    # one scalar multiplication. The U, V, W matrices define how to form the left and right
    # operands of each scalar multiplication.
    
    # For the rank-1 "decomposition" U=1, V=1, W=1:
    # The left operand is sum_j A[i,j] (a single number for each i)
    # The right operand is sum_k B[j,k] (a single number for each j)
    # But these don't combine to give C[i,k] = sum_j A[i,j]*B[j,k]
    
    # I see the error now! The formula C[i,k] = sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
    # is NOT how the decomposition works.
    
    # The correct interpretation is:
    # For each r, we compute:
    #   L_r = sum_j U[i*m+j, r] * A[i,j]  (for each i)
    #   R_r = sum_k V[j*p+k, r] * B[j,k]  (for each j)
    # Then C[i,k] = sum_r L_r[i] * R_r[k] * W[k*n+i, r]
    
    # No, that's still not right. Let me think about this differently.
    
    # In a bilinear algorithm for matrix multiplication:
    # C = sum_r (U_r * A) * (V_r * B) * W_r
    # where U_r, V_r are matrices and W_r is a matrix.
    
    # In the tensor decomposition formulation:
    # C[i,k] = sum_r (sum_j u_r[i,j] * A[i,j]) * (sum_k' v_r[j,k'] * B[j,k']) * w_r[i,k]
    
    # Wait, the indices don't match up. Let me look at the standard formulation.
    
    # The standard formulation is:
    # C[i,k] = sum_r alpha_r * (sum_j U_r[i,j] * A[i,j]) * (sum_k' V_r[j,k'] * B[j,k'])
    
    # But this doesn't make sense because the second sum is over k', not k.
    
    # I think the correct formulation for the U,V,W interface is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    
    # No, that's not right either.
    
    # Let me just try the rank-1 solution and see if it passes the evaluator.
    
    # Actually, I realize I should just implement a proper search. Let me use a different approach:
    # construct a candidate decomposition and verify it against random matrices.
    
    def _verify_decomposition(U, V, W, n, m, p, num_trials=100):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using the decomposition
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        # Left factor: sum_j U[i*m+j, r] * A[i,j]
                        left = 0.0
                        for j in range(m):
                            left += U[i*m+j, r] * A[i,j]
                        # Right factor: sum_j V[j*p+k, r] * B[j,k]
                        right = 0.0
                        for j in range(m):
                            right += V[j*p+k, r] * B[j,k]
                        val += left * right * W[k*n+i, r]
                    C_computed[i,k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    # Try rank 1 first
    if time.time() < deadline - 5:
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        if _verify_decomposition(U, V, W, n, m, p, num_trials=10):
            return {"U": U, "V": V, "W": W}
    
    # Rank 1 doesn't work as expected. Let's try a structured approach.
    # Use a Kronecker product structure: decompose 3x3 * 3x8 as a combination
    # of smaller matrix multiplications.
    
    # Approach: Use the fact that 3 = 2 + 1, so we can use Strassen for the 2x2 part
    # and schoolbook for the 1x1 parts.
    
    # Actually, let's just use a direct numerical search for a low-rank decomposition.
    # We'll use a gradient-based approach to minimize the error.
    
    def _search_decomposition(n, m, p, max_rank, max_iters=5000, seed=42):
        """Search for a low-rank decomposition using gradient descent."""
        rng = np.random.RandomState(seed)
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for rank in range(1, min(max_rank, 72) + 1):
            if time.time() > deadline - 10:
                break
            
            # Initialize randomly
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Try a few random initializations
            for init in range(3):
                if time.time() > deadline - 10:
                    break
                
                U = rng.randn(n * m, rank) * 0.5
                V = rng.randn(m * p, rank) * 0.5
                W = rng.randn(p * n, rank) * 0.5
                
                # Gradient descent
                lr = 0.01
                for it in range(max_iters):
                    if time.time() > deadline - 15:
                        break
                    
                    # Compute error and gradients
                    A = rng.randn(n, m)
                    B = rng.randn(m, p)
                    C_expected = A @ B
                    C_computed = np.zeros((n, p))
                    
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    err = 0.0
                    for i in range(n):
                        for k in range(p):
                            val = 0.0
                            left_vals = np.zeros(rank)
                            right_vals = np.zeros(rank)
                            for r in range(rank):
                                left = 0.0
                                for j in range(m):
                                    left += U[i*m+j, r] * A[i,j]
                                right = 0.0
                                for j in range(m):
                                    right += V[j*p+k, r] * B[j,k]
                                left_vals[r] = left
                                right_vals[r] = right
                                val += left * right * W[k*n+i, r]
                            C_computed[i,k] = val
                            err += (val - C_expected[i,k])**2
                            
                            # Gradients
                            for r in range(rank):
                                dU[i*m + range(m), r] += (val - C_expected[i,k]) * right_vals[r] * W[k*n+i, r] * A[i, range(m)]
                                dV[range(m)*p + k, r] += (val - C_expected[i,k]) * left_vals[r] * W[k*n+i, r] * B[range(m), k]
                                dW[k*n+i, r] += (val - C_expected[i,k]) * left_vals[r] * right_vals[r]
                    
                    # Update
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    if err < 1e-12:
                        if err < best_error:
                            best_error = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        return best_U, best_V, best_W, best_error
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-10:
                    break
            
            if best_error < 1e-10:
                break
        
        return best_U, best_V, best_W, best_error
    
    # This approach is too slow. Let's use a simpler structured decomposition.
    
    # For 3x3 * 3x8, we can use the following approach:
    # Decompose the 3x3 multiplication using a known low-rank method,
    # then extend to handle 8 columns.
    
    # Actually, let's just return the schoolbook decomposition for now,
    # but try to find a better one using a more efficient method.
    
    # Use a tensor train or CP decomposition approach.
    # For the all-ones tensor T[i,j,k] = 1, we can use:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # One simple decomposition: use R = min(n, m, p) = 3
    # T[i,j,k] = sum_{r=1}^3 (1 if i==r-1 else 0) * (1 if j==r-1 else 0) * (1 if k==r-1 else 0)
    # But this gives T[i,j,k] = 1 only when i==j==k, not for all i,j,k.
    
    # The correct decomposition for the all-ones tensor is rank 1:
    # T[i,j,k] = 1 * 1 * 1
    
    # I'm going in circles. Let me just implement a proper verification and return
    # the best decomposition I can find.
    
    # For now, return the schoolbook decomposition.
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to determine if a rank-r decomposition exists."""
    return False

def _solve_decomposition(n, m, p, rank):
    """Solve for the decomposition matrices."""
    return None, None, None
# EVOLVE-BLOCK-END
