"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 100  # Leave 20s safety margin for 120s limit
    
    # Start with schoolbook rank (100)
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank), dtype=np.float64)
    best_V = np.zeros((m * p, best_rank), dtype=np.float64)
    best_W = np.zeros((p * n, best_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1.0
                best_V[j * p + k, col] = 1.0
                best_W[k * n + i, col] = 1.0
                col += 1
    
    # Try block decomposition to reduce rank
    # Strategy: Partition A (4x5) into rows, B (5x5) into columns
    # Use known fast matrix multiplication for sub-blocks where possible
    
    # For 4x5 * 5x5, we can try:
    # - Split A into 2x5 blocks (2 blocks of 2 rows each)
    # - Split B into 5x2 and 5x3 blocks
    # This gives us 4 sub-products, each of form (2x5)*(5xk) for k=2 or 3
    
    # Actually, let's try a more structured approach:
    # Use the fact that 5x5 matrix multiplication has known algorithms
    # We can compute C = A*B by:
    # 1. Compute a 5x5x5 product using rank-r algorithm (if we had one)
    # 2. But we only need rows 0-3 of the result
    
    # Alternative: Use Kronecker product structure
    # A is 4x5, B is 5x5, C is 4x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try a simple optimization: since n=4 < m=p=5,
    # we might be able to use a decomposition that exploits this asymmetry
    
    # For now, let's try to use a rank reduction via structured ansatz
    # We'll use the schoolbook as baseline and try to find lower rank via random search
    
    # Random search for lower rank decompositions
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # The tensor for matrix multiplication T[i,j,k] = delta(i,j,k) means:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition says: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right. Let me re-read the task.
        
        # The tensor is: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 if the multiplication involves these indices
        # Actually for matrix multiplication, the tensor is:
        # T[i,j,k] = 1 for all i,j,k (it's the identity in the appropriate sense)
        # No wait, matrix multiplication tensor has T[i,j,k] = delta_{i,j,k} in some sense
        
        # Let me think again. The matrix multiplication tensor M_{n,m,p} has:
        # M[i,j,k] = 1 if the product C[i,k] uses A[i,j] and B[j,k]
        # So M[i,j,k] = 1 for all valid i,j,k where j is the inner dimension
        
        # A rank-r decomposition: M[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, the indices need to match. Let me use the standard formulation.
        
        # For C = AB where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        
        # The tensor T[i,j,k] = 1 (for all i,j,k in valid range)
        # Rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify by checking: for random A, B, does sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i,j,k]?
        # T[i,j,k] should be 1 for all i,j,k
        
        # Actually, I think the tensor is defined as:
        # T[i,j,k] = 1 if (i,j,k) is a valid triple, which it always is
        # So T is just the all-ones tensor? That doesn't seem right for matrix multiplication.
        
        # Let me reconsider. The matrix multiplication tensor is:
        # T[i,j,k] where i indexes rows of A/C, j indexes columns of A/rows of B, 
        # k indexes columns of B/C
        # T[i,j,k] = 1 (scalar) for all i in [n], j in [m], k in [p]
        
        # The rank of this tensor is what we're looking for.
        # A rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[i,k,r]
        # Note: W indices are (i,k) not (k,i) based on the task description W[k*n+i, r]
        
        # Wait, the task says W has shape (p*n, rank) with index W[k*n+i, r]
        # So W is indexed by (k, i) where k is in [0,p) and i is in [0,n)
        
        # Let's verify: for the schoolbook, U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        # for r = i*m*p + j*p + k (or some bijection)
        # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if there's an r where all three are 1
        # For schoolbook, each (i,j,k) triple has its own r, so the sum is 1 for each (i,j,k)
        # And T[i,j,k] = 1, so it works.
        
        # To verify a proposed decomposition, we check:
        # For all i in [n], j in [m], k in [p]:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        U_vec = U_mat[i*m+j, :]  # shape (rank,)
        V_vec = V_mat[j*p+k, :]  # shape (rank,)
        W_vec = W_mat[k*n+i, :]  # shape (rank,)
        return np.dot(U_vec, V_vec * W_vec) == 1.0
    
    # Actually, the verification above is wrong. Let me fix it.
    # For a given (i,j,k), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me implement proper verification
    def verify_full(U_mat, V_mat, W_mat, rank):
        """Verify the full decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    result = np.dot(u_val, v_val * w_val)
                    if abs(result - 1.0) > 1e-10:
                        return False
        return True
    
    # Try random low-rank decompositions
    # For rank r, we need U (20 x r), V (25 x r), W (20 x r)
    # such that for all i,j,k: sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    # This is a hard constraint satisfaction problem. Let's try a different approach.
    # We'll use the fact that the tensor is multi-linear and try to find a structured solution.
    
    # Actually, let's just return the schoolbook solution for now, as it's guaranteed correct.
    # The score is 1.0 which is the baseline.
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
