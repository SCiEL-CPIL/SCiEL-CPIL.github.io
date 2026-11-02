"""Rewritten program for 3x5x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i < n and j < m and k < p) else 0.0
                # Actually T[i,j,k] = 1 if this is a valid multiplication term
                # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # Wait, that's not right. Let me reconsider.
                
                # Matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor T has T[i,j,k] such that (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
                # This means T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
                # Actually no - the tensor for matrix multiplication is:
                # T[i,j,k] where the contraction is: result[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
                # So T[i,j,k] = 1 for all valid i,j,k
                # 
                # Wait, that would mean the tensor is all ones, which has rank 1?
                # No, that's not right either.
                
                # Let me reconsider. The matrix multiplication tensor T is defined such that:
                # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
                # This can be written as: (A*B)[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r] * A[i,j] * B[j,k]
                # No, that's not the standard formulation.
                
                # Standard formulation: The tensor T has components T[i,j,k] where:
                # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                # The "multiplication tensor" is actually just the identity in the middle index.
                # 
                # Actually, I think the correct formulation is:
                # We want to compute C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor decomposition represents this as:
                # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
                # No, that's not standard either.
                
                # Let me look at the seed code again:
                # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
                # So for each (i,j,k) triple, there's a rank-1 term.
                # The reconstruction is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
                # = sum_{c} [delta_{i,j,k,c}] * A[i,j] * B[j,k]
                # where delta is 1 only when c corresponds to (i,j,k)
                # = A[i,j] * B[j,k] for the specific (i,j,k)
                # But we need sum_j A[i,j]*B[j,k], so we sum over j for fixed i,k.
                # So the total is sum_j A[i,j]*B[j,k] which is correct!
                
                # So the verification is: for each (i,j,k), the sum over r of 
                # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                # (since we're just checking the tensor value, not the full multiplication)
                
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook and try to improve
    # Schoolbook rank = n*m*p = 105
    # Try to find lower rank
    
    # First, let's try a structured approach:
    # For matrix multiplication, we can sometimes reduce rank by grouping.
    
    # Strategy: Try ranks from a lower bound up to schoolbook
    # Lower bound for n x m x p matrix mult is max(n*m, m*p, n*p, (n*m*p)^(2/3)) roughly
    # But let's just try to find any valid decomposition with rank < 105
    
    best_result = None
    
    # Try schoolbook first as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Now try to find better decompositions
    # For small cases, we can try random initializations and optimize
    
    # Try various rank values below schoolbook
    # The minimum rank for 3x5x7 is unknown to me, but let's try to find it
    
    # Approach: Use a greedy/structured method
    # For each possible rank r, try to construct a valid decomposition
    
    # Actually, let me try a different approach:
    # Use the fact that matrix multiplication can be done with fewer than n*m*p operations
    # by exploiting linearity.
    
    # For now, return schoolbook as a safe baseline
    # The key insight: for 3x5x7, we might be able to do better than 105
    
    # Let me try a simple optimization: randomly initialize and use gradient descent
    # to find a valid decomposition with lower rank.
    
    def try_random_decomposition(rank, num_trials=100, seed=42):
        """Try to find a valid decomposition with given rank using random init + local search."""
        np.random.seed(seed)
        
        for trial in range(num_trials):
            # Random initialization with small integers/half-integers
            U = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook-1
    # Lower bound: at least max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # But likely higher. Let's try from 50 to 104
    
    for rank in range(50, rank_schoolbook):
        if time.time() > deadline - 5.0:
            break
        
        result = try_random_decomposition(rank, num_trials=50, seed=rank)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            # Found a valid decomposition, try to go lower
            # But for now, return this
            break
    
    return best_result
