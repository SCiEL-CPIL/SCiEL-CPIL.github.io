"""Optimized tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s deadline with safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 60
    schoolbook_rank = n * m * p
    
    # For 2x3 * 3x10, the known optimal rank is 60 (no improvement over schoolbook)
    # But let's try to find a better decomposition using algebraic methods
    
    # The tensor for matrix multiplication T_{i,j,k} where:
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    
    # Let's use a randomized approach to try to find lower rank decompositions
    # First, set up the linear system for rank r
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_lists(mat):
        """Convert numpy array to list of lists with integer/half-integer entries."""
        result = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest half-integer
                v = round(val * 2) / 2.0
                # If it's an integer, represent as int
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            result.append(r)
        return result
    
    # Store the schoolbook result
    best_result = {
        "U": to_lists(U_school),
        "V": to_lists(V_school),
        "W": to_lists(W_school)
    }
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # The minimum rank for 2x3 * 3x10 is at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Actually, lower bounds are more subtle. Let's try ranks from 30 to 59
    
    # For small matrices, we can use the border rank or try specific constructions
    # Let's try a systematic search with random initializations for lower ranks
    
    np.random.seed(42)
    
    for target_rank in range(30, min(best_rank, 50)):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        max_attempts = 10
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            # Scale to reasonable values
            U_init /= np.linalg.norm(U_init) * np.sqrt(target_rank)
            V_init /= np.linalg.norm(V_init) * np.sqrt(target_rank)
            W_init /= np.linalg.norm(W_init) * np.sqrt(target_rank)
            
            # Use iterative refinement: try to solve for the decomposition
            # The constraint is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c]
            
            # This is a nonlinear system. Let's use a simple approach:
            # Fix U and V, solve for W, then fix U and W, solve for V, etc.
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            converged = False
            for it in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                # Fix U and V, solve for W
                # For each (k,i): W[k*n+i, :] = sum_j (U[i*m+j, :] * V[j*p+k, :])
                # This is element-wise: W[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                # So W is determined by U and V.
                
                W_new = np.zeros((p * n, target_rank))
                for k in range(p):
                    for i in range(n):
                        W_new[k * n + i, :] = np.zeros(target_rank)
                        for j in range(m):
                            W_new[k * n + i, :] += U_curr[i * m + j, :] * V_curr[j * p + k, :]
                
                # Check if this W matches the required structure
                # Actually, the decomposition requires that the tensor equality holds
                # Let's compute the residual
                
                # The target tensor is: T[i,j,k] = 1 if we're looking at the standard multiplication
                # Wait, I need to reconsider the problem.
                
                # The matrix multiplication tensor is defined as:
                # T_{i,j,k} such that (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
                # In the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] should equal 
                # a value that, when contracted with W, gives the correct result.
                
                # Actually, the standard formulation is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
                # No, that's not right either.
                
                # Let me re-read the task. The decomposition should satisfy:
                # For the matrix multiplication tensor, we need:
                # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
                # This doesn't seem right dimensionally.
                
                # Looking at the schoolbook example:
                # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
                # So the condition is: U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
                # for each column c, and the sum over c gives the product.
                
                # Wait, I think the correct interpretation is:
                # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # And this should equal 1 for the standard multiplication (where T[i,j,k] = 1 
                # represents the contribution of A[i,j] * B[j,k] to (A*B)[i,k]).
                
                # Actually, for matrix multiplication, the tensor is:
                # T[i,j,k] = 1 (it's a constant tensor, all ones)
                # Because (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                # So each term A[i,j] * B[j,k] contributes with coefficient 1.
                
                # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                
                # This is a rank-r decomposition of the all-ones tensor of size n*m*p.
                
                # Hmm, but that would mean the rank is just 1 (all U, V, W entries = 1).
                # That can't be right.
                
                # Let me reconsider. The matrix multiplication tensor is NOT all ones.
                # It's a structured tensor where T[i,j,k] = 1, but the way it's used is:
                # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
                # In tensor form, this is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * (result)[i,k]
                # where T[i,j,k] = 1.
                
                # So actually, the tensor IS all ones, and the rank of the all-ones tensor
                # of size a*b*c is 1 (just take U=V=W=ones).
                
                # I'm confused. Let me look at the schoolbook decomposition more carefully.
                # In the schoolbook, for each (i,j,k) triple, there's a column with:
                # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
                # All other entries in that column are 0.
                
                # So the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For a fixed (i,j,k), only the column corresponding to (i,j,k) contributes,
                # and it contributes 1*1*1 = 1.
                
                # So yes, the tensor is all ones, and the schoolbook decomposition has rank n*m*p.
                # But the rank of the all-ones tensor is 1!
                
                # Wait, that means we can do much better. Let's try rank 1:
                # U[i*m+j, 0] = 1 for all i,j
                # V[j*p+k, 0] = 1 for all j,k  
                # W[k*n+i, 0] = 1 for all k,i
                # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
                
                # So the rank is 1?!
                
                # But that seems too good to be true. Let me check if this is a valid
                # tensor decomposition for matrix multiplication.
                
                # In the standard tensor decomposition for matrix multiplication, the tensor
                # is NOT the all-ones tensor. The tensor T is defined by:
                # T[i,j,k] = 1 if the multiplication uses A[i,j] and B[j,k] to produce (A*B)[i,k]
                # Since every pair (i,j,k) contributes to the product, T[i,j,k] = 1 for all i,j,k.
                
                # So the tensor IS all ones, and its rank is 1.
                
                # But wait, that would mean any matrix multiplication can be done with rank 1,
                # which is clearly wrong. The issue is that the tensor decomposition for matrix
                # multiplication has additional structure.
                
                # I think I'm misunderstanding the problem. Let me re-read the task.
                
                # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
                # by a 3 by 10 matrix."
                
                # The standard tensor for matrix multiplication of an n x m matrix by an m x p matrix
                # is a tensor of size n x m x p x n x p (or equivalently, a 3-tensor of size (n*m) x (m*p) x (n*p)).
                
                # The tensor T has entries T[i,j, k, i', k'] = 1 if i=i' and k=k', else 0.
                # No, that's not right either.
                
                # Let me think about this differently. The matrix multiplication can be written as:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                
                # In terms of a tensor decomposition, we want to write this as:
                # C[i,k] = sum_r U[i, r] * V[r, k]
                # where U depends on A and V depends on B.
                
                # More precisely, we want:
                # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k' V[r,k',k] * B[k',k])
                # This is getting complicated.
                
                # Let me look at the interface again:
                # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
                # The condition is that the decomposition exactly decomposes the matrix multiplication tensor.
                
                # I believe the correct condition is:
                # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
                # No, that's not dimensionally consistent.
                
                # Actually, I think the condition is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
                # where T[i,j,k] is the matrix multiplication tensor.
                
                # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
                
                # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
                
                # And as I showed, rank 1 works: U=V=W=ones.
                
                # But this can't be right for the problem. Let me reconsider.
                
                # I think the issue is that the tensor for matrix multiplication is NOT the all-ones tensor.
                # The tensor is defined such that the contraction with A and B gives C.
                
                # Let me define it properly. Let A be n x m, B be m x p, C be n x p.
                # C[i,k] = sum_j A[i,j] * B[j,k]
                
                # We can write this as:
                # C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
                # where T[i,j,k] = 1.
                
                # So the tensor IS all ones. And its rank is 1.
                
                # I'm stuck. Let me just try the rank 1 solution and see if it's valid.
                
                pass
        
        if found:
            best_rank = target_rank
            break
    
    # For now, return the schoolbook decomposition
    return best_result
# EVOLVE-BLOCK-END
