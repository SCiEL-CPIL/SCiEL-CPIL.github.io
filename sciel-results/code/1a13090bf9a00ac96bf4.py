"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (index into A, i in [0,n), j in [0,m))
    # b = j*p + k (index into B, j in [0,m), k in [0,p))
    # c = k*n + i (index into C, using trace convention W[k*n+i])
    # T[a,b,c] = 1 if the indices are consistent, else 0
    # For a given (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(recon, T, atol=1e-10)
    
    # Trivial decomposition
    U = np.zeros((n*m, trivial_rank), dtype=np.float64)
    V = np.zeros((m*p, trivial_rank), dtype=np.float64)
    W = np.zeros((p*n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    best_rank = trivial_rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Strategy: Try to find lower rank decompositions using structured approaches
    # Key idea: For matrix multiplication, we can share intermediate computations
    
    # Approach 1: Try structured decompositions based on matrix multiplication properties
    # For M_{n,m,p}, the rank is at most n*m*p but can be lower
    
    # Let's try a systematic approach:
    # 1. Start with trivial rank
    # 2. Try to reduce by finding linear dependencies among the rank-1 terms
    
    # For small cases, we can try to find decompositions with rank < trivial_rank
    # by solving a system of equations
    
    # Key insight: The matrix multiplication tensor has structure that allows
    # sharing of intermediate products. Specifically, for each intermediate index j,
    # we can share the computation across different i and k.
    
    # Let's try a structured decomposition:
    # For each j in [0,m), we have a "slice" of the multiplication
    # C[:,k] = sum_j A[:,j] * B[j,k]
    
    # A structured approach: use the fact that for fixed j, the contribution
    # to all C[i,k] for that j can be computed with shared terms
    
    # Actually, let's try a more direct approach:
    # Use the standard matrix multiplication algorithm but look for redundancies
    
    # For M_{3,4,7}, let's try ranks from 83 down to some minimum
    # and use a local search approach
    
    # Better approach: Use the fact that we can represent the multiplication
    # using fewer multiplications by exploiting the structure
    
    # Let's try a decomposition based on the following idea:
    # Instead of computing each C[i,k] = sum_j A[i,j]*B[j,k] separately,
    # we can group terms that share the same A[i,j] or B[j,k]
    
    # For a more systematic approach, let's try to find a decomposition
    # by solving for the coefficients directly
    
    # Start with the trivial decomposition and try to merge columns
    # Two columns r1 and r2 can be merged if there exists a column r3
    # such that the contribution of r1 + r2 equals the contribution of r3
    # (in the tensor sense)
    
    # Actually, a better approach: try to find a decomposition with rank < trivial_rank
    # by using the structure of matrix multiplication
    
    # For small matrices, we can try all possible structured decompositions
    
    # Let's try a specific approach: use the fact that for each pair (i,k),
    # the sum over j can be computed with shared intermediate values
    
    # Strategy: Try to find a decomposition where some rank-1 terms are shared
    # across multiple (i,j,k) triples
    
    # For now, let's try a randomized local search approach
    # that starts from the trivial decomposition and tries to reduce rank
    
    # First, let's try a simple approach: check if we can reduce by 1
    # by finding a linear combination of columns that gives the same result
    
    # Actually, let's try a more principled approach:
    # Use the Strassen-like idea of computing matrix products with fewer multiplications
    
    # For 2x2 matrices, Strassen uses 7 multiplications instead of 8
    # For larger matrices, we can use similar ideas
    
    # Let's try to implement a search that looks for structured decompositions
    
    # Given the time constraint, let's focus on finding a decomposition
    # with rank slightly below trivial_rank
    
    # Try to find a decomposition with rank = trivial_rank - 1, trivial_rank - 2, etc.
    # by solving a linear system
    
    # For rank R, we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    
    # This is a non-linear system, but we can linearize it by fixing two matrices
    # and solving for the third
    
    # Let's try a simple approach: fix U and V to be sparse and solve for W
    
    # Actually, for exact solutions with integer/half-integer entries,
    # let's try a different approach: construct the decomposition explicitly
    
    # For M_{3,4,7}, let's try to use the following structured decomposition:
    # Group the terms by the intermediate index j
    
    # For each j, we have contributions to C[i,k] for all i,k
    # The contribution from index j is: A[i,j] * B[j,k] for all i,k
    
    # We can write this as a sum of rank-1 terms in the (i,k) space
    # But we need to be careful about the tensor structure
    
    # Let me try a direct construction:
    # Use the standard algorithm but look for opportunities to share terms
    
    # For the specific case of 3x4x7, let's try to find a decomposition
    # by brute force over structured patterns
    
    # Given the complexity, let's try a heuristic approach:
    # Start with trivial and try to merge columns that have overlapping support
    
    # Two columns can potentially be merged if their supports overlap significantly
    # and their values can be combined
    
    # Let's implement a simple local search:
    # 1. Start with trivial decomposition
    # 2. Try to remove one column at a time by adjusting others
    # 3. Use least squares to find the adjustment
    
    # Actually, let's try a more direct approach:
    # For rank R, we can set up the problem as finding R vectors u_r, v_r, w_r
    # such that the tensor equation is satisfied
    
    # Let's try a simple iterative method:
    # Alternate between fixing U,V and solving for W, etc.
    
    # But for exact solutions, this is tricky. Let's try a different approach.
    
    # Key insight: For matrix multiplication, the rank is related to the
    # bilinear complexity. For small matrices, we can often find explicit
    # decompositions.
    
    # Let's try to construct a decomposition with rank = n*m*p - m*(n-1)*(p-1)
    # or some similar formula that accounts for shared intermediate products
    
    # Actually, let's just try to find a decomposition by solving the system
    # for decreasing ranks
    
    # For each candidate rank R, try to find U, V, W with integer/half-integer entries
    
    # This is a hard problem, so let's use a heuristic:
    # 1. Start with the trivial decomposition
    # 2. Try to reduce rank by finding linear dependencies
    
    # A column r can be removed if there exist coefficients c_s (s != r)
    # such that the tensor contribution of column r equals the sum of
    # c_s * (contribution of column s)
    
    # This is equivalent to checking if column r is in the span of the other columns
    # in the tensor product space
    
    # Let's implement this check
    
    def try_reduce_rank(U, V, W, current_rank):
        """Try to reduce rank by removing one column"""
        if current_rank <= 1:
            return None, None, None, current_rank
        
        # For each column, check if it can be expressed as a combination of others
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 5:
                return None, None, None, current_rank
            
            # Get the other columns
            other_cols = [c for c in range(current_rank) if c != col_to_remove]
            
            # The contribution of column col_to_remove is:
            # u = U[:, col_to_remove], v = V[:, col_to_remove], w = W[:, col_to_remove]
            # We need to check if this can be written as sum_s c_s * (U[:,s] outer V[:,s] outer W[:,s])
            
            # This is a non-linear problem, but we can try to solve it
            # by fixing the structure
            
            # Actually, for a rank-1 term to be in the span of other rank-1 terms,
            # it's not straightforward. Let's try a different approach.
            
            # Instead, let's try to find a new decomposition with rank-1
            # by solving a least squares problem
            
            # Set up the problem: find coefficients c_s such that
            # sum_s c_s * (U[:,s] outer V[:,s] outer W[:,s]) ≈ U[:,col_to_remove] outer V[:,col_to_remove] outer W[:,col_to_remove]
            
            # This is a linear problem in c_s if we fix the structure
            # But the tensor product makes it non-linear
            
            # Let's try a simpler approach: just check if the column is redundant
            # by trying to remove it and seeing if the decomposition still works
            # with adjusted coefficients
            
            # For now, let's skip this and try a different strategy
        
        return U, V, W, current_rank
    
    # Given the difficulty of the general problem, let's try a specific approach
    # for this small case: use a structured decomposition based on the matrix
    # multiplication algorithm
    
    # For M_{3,4,7}, let's try to use the following idea:
    # The rank is at most 3*4*7 = 84, but we can do better by sharing
    # intermediate products
    
    # A simple sharing strategy: for each intermediate index j, compute
    # all A[i,j]*B[j,k] products together
    
    # Actually, let's try a very simple approach: just return the trivial
    # decomposition, but try to find a slightly better one if possible
    
    # Let's try to find a decomposition with rank 83 by checking if
    # any two columns can be merged
    
    # For two columns r1 and r2, their combined contribution is:
    # u1 outer v1 outer w1 + u2 outer v2 outer w2
    # This can be written as a single rank-1 term if and only if
    # u1 outer v1 outer w1 + u2 outer v2 outer w2 = u outer v outer w
    # for some u, v, w
    
    # This is a non-trivial condition. Let's check it numerically
    
    def can_merge_columns(U, V, W, r1, r2):
        """Check if columns r1 and r2 can be merged into a single column"""
        # The combined tensor is:
        T_combined = np.einsum('a,b,c->abc', U[:, r1], V[:, r1], W[:, r1]) + \
                     np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
        
        # Check if this is a rank-1 tensor
        # A tensor is rank-1 if it can be written as u outer v outer w
        # This is equivalent to checking if all 2x2 minors of the unfolded matrices are zero
        
        # Unfold along first mode: T_combined.reshape(n*m, -1)
        # If rank-1, this matrix should have rank 1
        
        # Actually, for a 3-way tensor, rank-1 means it's an outer product
        # We can check by trying to factor it
        
        # Simple check: if the tensor is rank-1, then for any two indices
        # in different modes, the ratio should be consistent
        
        # Let's try to factor it using SVD-like approach
        # Reshape to a matrix: (n*m) x (m*p * p*n)
        mat = T_combined.reshape(n*m, -1)
        
        # If rank-1, the SVD should have only one non-zero singular value
        try:
            # Use a tolerance-based rank check
            singular_values = np.linalg.svd(mat, compute_uv=False)
            if len(singular_values) > 0 and singular_values[0] > 1e-10:
                if len(singular_values) > 1 and singular_values[1] > 1e-8 * singular_values[0]:
                    return False
            # Check if it's actually rank-1 by trying to reconstruct
            if singular_values[0] < 1e-10:
                return True  # Zero tensor, trivially rank-1
            # Try to find u, v, w such that T = u outer v outer w
            # This is underdetermined, so let's just check if the numerical rank is 1
            return True
        except:
            return False
    
    # Try to merge pairs of columns
    current_rank = trivial_rank
    U_cur = best_U.copy()
    V_cur = best_V.copy()
    W_cur = best_W.copy()
    
    merged_any = True
    while merged_any and time.time() < deadline - 10:
        merged_any = False
        for r1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 10:
                    break
                if can_merge_columns(U_cur, V_cur, W_cur, r1, r2):
                    # Try to merge them
                    # The merged column should be the sum, but we need to find
                    # a rank-1 representation
                    # For now, just keep the first column and remove the second
                    # (this is a simplification; in general, we'd need to
                    # find the actual merged rank-1 term)
                    
                    # Remove column r2
                    mask = [c for c in range(current_rank) if c != r2]
                    U_cur = U_cur[:, mask]
                    V_cur = V_cur[:, mask]
                    W_cur = W_cur[:, mask]
                    current_rank -= 1
                    merged_any = True
                    break
            if merged_any:
                break
        
        # Verify the decomposition still works
        if verify_decomposition(U_cur, V_cur, W_cur, current_rank):
            if current_rank < best_rank:
                best_rank = current_rank
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
        else:
            # Roll back
            U_cur = best_U.copy()
            V_cur = best_V.copy()
            W_cur = best_W.copy()
            current_rank = best_rank
            merged_any = False
    
    # Convert to lists for output
    U_out = best_U[:, :best_rank].tolist()
    V_out = best_V[:, :best_rank].tolist()
    W_out = best_W[:, :best_rank].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
