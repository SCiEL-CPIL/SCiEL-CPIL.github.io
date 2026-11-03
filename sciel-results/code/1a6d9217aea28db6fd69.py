"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook decomposition as baseline (guaranteed valid, rank = n*m*p)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to improve using algebraic approach with half-integer entries
    # Strategy: For each row i, try to find a rank-r decomposition of the 1 x 4 x 7 tensor
    # such that the total rank is less than 84.
    # We use a constructive approach based on splitting the 4x7 multiplication.
    
    # Let's try a hybrid approach:
    # For 3x4x7, we can try to use rank 14 per row (total 42) or similar.
    # But first, let's see if we can achieve rank < 84 with a simple structural improvement.
    
    # Known: R(1, 4, 7) >= 4*7 = 28 if we use standard multiplication per row.
    # Total would be 3 * 28 = 84.
    
    # However, there might be cross-term optimizations.
    # Let's attempt a search for a valid decomposition with rank < 84.
    # We will use a randomized approach with exact arithmetic (fractions) to ensure validity.
    
    from fractions import Fraction
    
    def try_random_decomposition(rank_target):
        """
        Try to find a decomposition of rank rank_target.
        Uses random linear combinations and checks validity exactly.
        """
        # For each rank target, we need U (nm x r), V (mp x r), W (pn x r)
        # such that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(i,j,k) ?
        # No, the condition is:
        # The tensor M_{i,j,k} = 1 if we are multiplying A_{i,j} * B_{j,k} to get C_{i,k}
        # The decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
        # Wait, the standard form is:
        # C_{i,k} = sum_c (sum_j U[i,j,c] * A_{i,j}) * (sum_j V[j,k,c] * B_{j,k})
        # This is not quite right. The standard bilinear decomposition is:
        # C_{i,k} = sum_c U_c(A) * V_c(B)
        # where U_c is a linear form on A and V_c is a linear form on B.
        # U_c(A) = sum_{i,j} U[i,j,c] * A_{i,j}
        # V_c(B) = sum_{j,k} V[j,k,c] * B_{j,k}
        # And W is not directly in the product, it maps to C?
        # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
        # And W[k*n+i, r] follows trace convention.
        
        # Let's re-read the task carefully.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix."
        # "The output factor follows the official trace convention W[k*n+i,r]."
        
        # The tensor for matrix multiplication C = AB is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor T has indices (i, j, k) and T_{i,j,k} = 1 if we are forming C_{i,k} from A_{i,j} and B_{j,k}?
        # No, the tensor is T_{i, j, k} = delta_{i,k}? No.
        
        # The matrix multiplication tensor T_{n,m,p} has entries T_{i,j,k} = 1 if i, j, k are valid indices?
        # Actually, the tensor is defined such that the bilinear form is:
        # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k} * W_{i,k} ?
        
        # Let's look at the seed program:
        # U[i*m+j, c] = 1
        # V[j*p+k, c] = 1
        # W[k*n+i, c] = 1
        # for each (i, j, k).
        
        # This corresponds to the term A_{i,j} * B_{j,k} contributing to C_{i,k}.
        # The decomposition is:
        # C_{i,k} = sum_c U_c(A) * V_c(B)
        # where U_c(A) = sum_{i,j} U[i,j,c] A_{i,j}
        # and V_c(B) = sum_{j,k} V[j,k,c] B_{j,k}
        
        # But W is involved. How?
        # Usually, the decomposition is given by three sets of matrices:
        # U_c (n x m), V_c (m x p), and the result is C = sum_c U_c(A) V_c(B)?
        # No, that would be a rank-1 update.
        
        # The standard definition of rank of the matrix multiplication tensor is:
        # We seek matrices U_c (n x m), V_c (m x p) and scalars or matrices W_c (n x p) such that:
        # T_{i,j,k} = sum_c U_c[i,j] V_c[j,k] W_c[k,i] ? No.
        
        # Let's check the seed again.
        # For a fixed (i,j,k), we have:
        # U[i*m+j, c] = 1
        # V[j*p+k, c] = 1
        # W[k*n+i, c] = 1
        
        # This looks like the tensor is decomposed as:
        # T = sum_c u_c \otimes v_c \otimes w_c
        # where u_c is a vector of length nm, v_c is a vector of length mp, w_c is a vector of length pn.
        
        # The condition for exact decomposition is:
        # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Let's verify this with the seed.
        # For a specific (i,j,k), say (0,0,0).
        # T_{0,0,0} should be 1 (since A_{0,0} * B_{0,0} contributes to C_{0,0}).
        # Sum over c:
        # Only one c has U[0, c]=1, V[0, c]=1, W[0, c]=1.
        # That c corresponds to (i=0, j=0, k=0).
        # The product is 1*1*1 = 1.
        # For other (i',j',k'), T_{i',j',k'} = 0?
        # If (i',j',k') != (i,j,k), then there is no c such that all three are 1.
        # So the sum is 0.
        
        # So the condition is:
        # For all i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a system of nm*p equations.
        # We want to minimize rank (number of columns c).
        
        # This is the standard definition of the rank of the matrix multiplication tensor.
        # For 3x4x7, the schoolbook rank is 3*4*7 = 84.
        
        # Can we do better?
        # Known lower bounds: R(n,m,p) >= nm, np, mp.
        # Here nm=12, np=21, mp=28.
        # So R >= 28.
        
        # It is known that R(1, n, m) = nm.
        # For 2x2x2, R=7 (Strassen).
        # For 3x4x7, it is likely that R=84, but let's try to find a better one.
        
        # We will use a randomized search with exact arithmetic.
        # We will try ranks from 28 upwards.
        
        # For each rank r, we will generate random matrices U, V, W with rational entries
        # and check if they satisfy the equations.
        
        # To make it efficient, we can fix U and V randomly and solve for W?
        # No, the equations are linear in W if U and V are fixed.
        # For each (i,j,k), we have:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let's define A_{i,j,k,c} = U[i*m+j, c] * V[j*p+k, c].
        # Then sum_c A_{i,j,k,c} * W[k*n+i, c] = 1.
        
        # This is a linear system in W.
        # There are nm*p equations and (pn)*r unknowns? No, W has pn * r elements.
        # But each equation involves only r terms (one for each c), and the index of W is k*n+i.
        # So for a fixed (k,i), the equations are:
        # For all j: sum_c A_{i,j,k,c} * W[k*n+i, c] = 1
        
        # This means that for each (k,i), the vector w_{k,i} of length r must satisfy:
        # sum_c A_{i,j,k,c} * w_{k,i,c} = 1 for all j.
        
        # This is a linear system: M_{k,i} * w_{k,i} = 1
        # where M_{k,i} is an m x r matrix with entries M_{j,c} = U[i*m+j, c] * V[j*p+k, c].
        
        # So the problem reduces to:
        # 1. Generate random U (nm x r) and V (mp x r).
        # 2. For each (k,i), form the matrix M_{k,i} (m x r).
        # 3. Check if there exists a solution w_{k,i} to M_{k,i} w = 1.
        # 4. If yes, we have a valid decomposition.
        
        # We want the solution to have rational (or half-integer) entries.
        # If M_{k,i} has rational entries, the solution will be rational if it exists.
        
        # To ensure half-integers, we can scale.
        
        # Let's implement this search.
        
        def is_valid_decomposition(U, V, W):
            # U: (nm, r), V: (mp, r), W: (pn, r)
            # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        # We will search for r from 28 to 83.
        # If we find one, we return it.
        
        # To make the search efficient, we can use the structure of the linear system.
        
        # For a given r, we generate random U and V.
        # Then for each (k,i), we solve M_{k,i} w = 1.
        # If the system is consistent, we have a solution.
        
        # We use numpy to solve the least squares problem and check the residual.
        
        import numpy as np
        
        for r in range(28, 84):
            if time.time() > deadline:
                break
            
            # Try a few random attempts for each r
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                # Generate random U and V with entries in {-1, 0, 1} or similar
                U_rand = np.random.randint(-1, 2, (n*m, r)).astype(np.float64)
                V_rand = np.random.randint(-1, 2, (m*p, r)).astype(np.float64)
                
                W_rand = np.zeros((p*n, r), dtype=np.float64)
                
                success = True
                for i in range(n):
                    for k in range(p):
                        # Form M_{k,i} (m x r)
                        M = np.zeros((m, r), dtype=np.float64)
                        for j in range(m):
                            for c in range(r):
                                M[j, c] = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                        
                        # Solve M w = 1
                        try:
                            w, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            # Check residual
                            if residuals.size > 0 and residuals[0] < 1e-8:
                                W_rand[k*n+i, :] = w
                            else:
                                success = False
                                break
                        except:
                            success = False
                            break
                    
                    if not success:
                        break
                
                if success:
                    # We have a valid decomposition with rank r
                    # Check if entries are close to half-integers
                    # If not, we can try to clean them up
                    best_rank = r
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_rand
                    break
            
            if best_rank < rank:
                break
        
        # Convert to lists of integers/half-integers if possible
        def to_half_integer_list(matrix):
            res = []
            for row in matrix:
                r_list = []
                for val in row:
                    # Round to nearest half-integer
                    rounded = round(val * 2) / 2.0
                    r_list.append(rounded)
                res.append(r_list)
            return res
        
        # Ensure the best decomposition is valid with the cleaned entries
        def validate_cleaned(U_list, V_list, W_list):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(len(U_list[0])):
                            s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                        if abs(s - 1.0) > 1e-6:
                            return False
            return True
        
        U_list = to_half_integer_list(best_U)
        V_list = to_half_integer_list(best_V)
        W_list = to_half_integer_list(best_W)
        
        if validate_cleaned(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fall back to schoolbook
            U_list = to_half_integer_list(U)
            V_list = to_half_integer_list(V)
            W_list = to_half_integer_list(W)
            return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
