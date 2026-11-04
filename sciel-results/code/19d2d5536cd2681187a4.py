"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank), dtype=object)
    V = np.zeros((m * p, schoolbook_rank), dtype=object)
    W = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1

    def tensor_value(U_mat, V_mat, W_mat, rank):
        """Compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    T[(i,j,k)] = s
        return T

    def verify(U_mat, V_mat, W_mat, rank):
        T = tensor_value(U_mat, V_mat, W_mat, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[(i,j,k)] != Fraction(1):
                        return False
        return True

    # Verify schoolbook
    if not verify(U, V, W, schoolbook_rank):
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}

    # Strategy: Try to find rank reductions by exploiting algebraic structure
    # For matrix multiplication, we can use the following identity:
    # (a+b)(c+d) = ac + ad + bc + bd
    # This suggests that some columns might be combined.
    
    # More systematically: try to find subsets of columns whose sum of tensor contributions
    # equals the tensor contribution of a single column, allowing us to merge them.
    
    # For a rank-r decomposition, column r contributes:
    # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # We can try: for each pair of columns (a,b), check if there exists a column c
    # such that T_a + T_b - T_c = 0 (i.e., T_c = T_a + T_b)
    # This would allow us to replace columns a and b with a single column c.
    
    # But this is a nonlinear constraint. Let's try a different approach.
    
    # Known result: For 3x3x4 matrix multiplication, the best known rank is 30.
    # Let's try to construct a rank-30 decomposition using known algorithms.
    
    # One approach: use the "lattice reduction" idea from the directive.
    # Start with rank 36 and try to reduce by finding linear dependencies.
    
    # Convert to float for numerical work
    U_float = U.astype(float)
    V_float = V.astype(float)
    W_float = W.astype(float)
    
    # Try rank reduction by removing columns that are "redundant"
    # A column r is redundant if its tensor contribution can be expressed
    # as a linear combination of other columns' contributions.
    
    # For each column, compute its tensor contribution vector
    def column_tensor(U_mat, V_mat, W_mat, r, rank):
        """Return the tensor contribution of column r as a flat vector"""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    vec[idx] = U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
        return vec
    
    # Try to find a rank-30 decomposition by structured search
    # Use the fact that for 3x3x4, we can partition the problem
    
    # Let's try a more direct approach: search for rank r decompositions
    # using the structure of the tensor.
    
    # For now, let's try to verify if rank 30 is achievable with a specific construction.
    # The key insight: for 3x3x4, we can use block methods.
    
    # Partition A (3x3) into 1x3 row blocks, B (3x4) into 3x1 column blocks
    # This gives 3*3 = 9 subproblems of 1x3 * 3x1 = scalar products
    # Each scalar product has rank 1, total rank 9? No, that's not right.
    
    # Actually, let's try a randomized approach with better initialization.
    # Initialize with the schoolbook solution and perturb.
    
    np.random.seed(42)
    
    def try_rank_reduction(start_rank, target_rank, max_attempts=100):
        """Try to reduce rank from start_rank to target_rank"""
        if time.time() > deadline - 10:
            return None
        
        # Start with schoolbook and try to eliminate (start_rank - target_rank) columns
        U_curr = U_float.copy()
        V_curr = V_float.copy()
        W_curr = W_float.copy()
        current_rank = start_rank
        
        # Try to eliminate columns one by one
        for elim in range(start_rank - target_rank):
            if time.time() > deadline - 5:
                return None
            
            # For each pair of columns, try to merge them
            found = False
            for r1 in range(current_rank):
                if found:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() > deadline - 5:
                        return None
                    
                    # Try to find a replacement column that equals the sum
                    # T_r1 + T_r2 = T_new
                    # This requires: for all (i,j,k):
                    # U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + 
                    # U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2] = 
                    # U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i]
                    
                    # This is generally impossible, so let's try a different approach.
                    # Instead, try to set one column to zero and see if we can adjust
                    # the others to compensate.
                    
                    # Skip for now - this approach is too complex
                    pass
            
            # Alternative: try to find a column that can be set to zero
            # by adjusting the other columns
            for r in range(current_rank):
                if time.time() > deadline - 5:
                    return None
                
                # Save original
                U_save = U_curr.copy()
                V_save = V_curr.copy()
                W_save = W_curr.copy()
                
                # Set column r to zero
                U_curr[:, r] = 0
                V_curr[:, r] = 0
                W_curr[:, r] = 0
                
                # Check if the tensor is still correct
                T = tensor_value(U_curr.astype(object), V_curr.astype(object), W_curr.astype(object), current_rank)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if T[(i,j,k)] != Fraction(1):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Column r was redundant! Remove it
                    U_curr = np.delete(U_curr, r, axis=1)
                    V_curr = np.delete(V_curr, r, axis=1)
                    W_curr = np.delete(W_curr, r, axis=1)
                    current_rank -= 1
                    found = True
                    break
                else:
                    # Restore
                    U_curr = U_save
                    V_curr = V_save
                    W_curr = W_save
            
            if not found:
                return None
        
        # Verify final result
        U_final = U_curr.astype(object)
        V_final = V_curr.astype(object)
        W_final = W_curr.astype(object)
        if verify(U_final, V_final, W_final, current_rank):
            return U_final, V_final, W_final, current_rank
        return None
    
    # Try to reduce from 36 to various target ranks
    best_result = None
    for target in [30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 15:
            break
        
        result = try_rank_reduction(schoolbook_rank, target)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U_final, V_final, W_final, final_rank = best_result
        return {"U": [[float(x) for x in row] for row in U_final], 
                "V": [[float(x) for x in row] for row in V_final], 
                "W": [[float(x) for x in row] for row in W_final]}
    
    # If no reduction found, return schoolbook
    return {"U": [[float(x) for x in row] for row in U], 
            "V": [[float(x) for x in row] for row in V], 
            "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
