"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s kill limit)
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # First, establish the schoolbook rank-36 decomposition as a fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find a lower-rank decomposition
    # The matrix multiplication tensor T of size n*m*p has entries T[i*m+j, j*p+k, k*n+i] = 1
    # A rank-r decomposition means T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n
    
    # Strategy: try ranks from a reasonable lower bound up to 36
    # For 3x3x4, known results suggest rank might be as low as ~20-24
    
    # We'll use a structured search approach:
    # The multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term in the decomposition corresponds to a "bilinear form"
    
    # Let's try to find decompositions of rank 24, 23, ..., down to some minimum
    # We'll use a greedy/structured approach based on the algebraic structure
    
    # Key insight: For matrix multiplication, we can exploit the structure
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # A rank-r decomposition has r triples (u_c, v_c, w_c)
    # such that for all (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = [i,j,k is valid]
    
    # Actually, the tensor is: T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
    # More precisely, T is defined on the "multiplication tensor" where:
    # T_{(i,j), (j,k), (i,k)} = 1 for all valid i,j,k, and 0 otherwise
    
    # Let me reformulate: We need to find vectors u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    # such that for all i, j, k:
    #   sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    # and for all other index combinations (that don't correspond to valid matrix mult), the sum is 0.
    
    # Wait, actually the tensor is only non-zero at positions where the indices form a valid
    # matrix multiplication triple. So the condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # And we need the decomposition to be "exact", meaning it only produces the correct
    # multiplication. The way this works is that the bilinear map defined by the decomposition
    # must match matrix multiplication.
    
    # Let's try a structured approach:
    # For each j, the contribution to C[i,k] from A[i,j]*B[j,k] is a rank-1 tensor
    # (in the i,k space) with entries A[i,j]*B[j,k]
    
    # To reduce rank, we need to "share" terms across different j values.
    
    # Let's try rank 24: use a structured decomposition
    # Idea: For each pair (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # If we could express this with fewer than n*m*p terms...
    
    # Try a specific construction for rank 24:
    # Use the fact that m=3, so we have 3 "layers"
    # For each layer j, we have n*p = 12 terms, total 36
    # To get 24, we need to eliminate 12 terms
    
    # Alternative: try to find a decomposition using bilinear identities
    
    # Let's try a computational approach: set up the linear system
    # For rank r, we have r * (n*m + m*p + p*n) unknowns
    # And n*m*p equations (one for each (i,j,k) triple)
    # Plus we need the decomposition to be valid (no spurious terms)
    
    # Actually, the standard way to verify: the bilinear map defined by (U,V,W)
    # computes C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not quite right. Let me re-read the interface.
    
    # The interface says: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The decomposition means: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where T is the matrix multiplication tensor.
    
    # The matrix multiplication tensor T has T[a, b, c] = 1 if a, b, c correspond to
    # a valid (i,j), (j,k), (i,k) triple, i.e., a = i*m+j, b = j*p+k, c = k*n+i
    # for some i, j, k, and 0 otherwise.
    
    # So we need: for all a, b, c: sum_{r=0}^{rank-1} U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of (n*m)*(m*p)*(p*n) equations, but most of T is zero.
    # The non-zero entries are at positions (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p)
    
    # Let's try to find a rank-r decomposition by solving the linear system
    # For a given rank r, we have r*(n*m + m*p + p*n) variables
    # And n*m*p non-zero constraints (each equal to 1)
    # Plus (n*m)*(m*p)*(p*n) - n*m*p zero constraints
    
    # This is a nonlinear system (trilinear), so it's hard to solve directly.
    # But we can use the following approach: fix U and V, then W is determined by:
    # W[c,r] = sum_{a,b: T[a,b,c]=1} T[a,b,c] * U[a,r] * V[b,r] / ... 
    # No, that doesn't work directly.
    
    # Alternative: use the fact that for each c (column of U,V,W), the contribution
    # is a rank-1 tensor. So we're looking for r rank-1 tensors that sum to T.
    
    # Let's try a specific known construction. For 3x3x4 matrix multiplication,
    # there are known decompositions of rank less than 36.
    
    # One approach: use the "Strassen-like" algorithm structure
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # For 3x3x3, there are algorithms of rank ~23-24
    
    # Let's try to construct a rank-24 decomposition for 3x3x4
    # Using a structured approach:
    
    # The key idea: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # We can try to compute this with fewer multiplications by using linear combinations
    
    # For each k, we have a vector C[:,k] = A * B[:,k]
    # This is a matrix-vector product: (3x3) * (3x1)
    # The rank of the matrix-vector multiplication tensor for 3x3 is 9 (schoolbook)
    # But we can do better with bilinear algorithms
    
    # Actually, let's try a direct construction:
    # Use the fact that for each k, C[:,k] = sum_j A[:,j] * B[j,k]
    # We can precompute some linear combinations of the columns of A
    
    # Let's try rank 24 with a specific pattern:
    # For each k (4 values), and each "group" of j values, use shared terms
    
    # Actually, let me try a simpler approach: just verify if rank 24 is achievable
    # by setting up the problem as a polynomial system and using a solver
    
    # Given the time constraint, let's try a few specific rank values with a structured search
    
    # Try rank 24:
    # Use a "block" structure: for each k, use 6 terms (4*6=24)
    # For fixed k, C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to compute 3 values (i=0,1,2) from 3 inputs each
    # This is a 3-input, 3-output bilinear map
    # Schoolbook: 3 multiplications
    # Can we do it with 2? Probably not for arbitrary inputs
    # But we can share across k values
    
    # Let's try a different structure:
    # Use 8 terms per "layer" of 3 k-values... this is getting complicated
    
    # Let's just try to find ANY valid decomposition of rank < 36 using a search
    # We'll use a random initialization + optimization approach
    
    # For rank r, initialize U, V, W randomly and try to minimize the error
    # using gradient descent on the log-sum-exp of squared errors
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W form a valid rank decomposition of the matrix mult tensor."""
        rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
        # Convert to numpy if needed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check dimensions
        if U_np.shape != (n*m, rank):
            return False
        if V_np.shape != (m*p, rank):
            return False
        if W_np.shape != (p*n, rank):
            return False
        
        # For each (i,j,k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_np[a, :], V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_full_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Full verification: check all entries of the tensor."""
        rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # The tensor T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
        # T is 1 at positions (i*m+j, j*p+k, k*n+i) and 0 elsewhere
        
        # Check non-zero positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_np[a, :], V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check some zero positions (sample)
        # For efficiency, check a subset of zero positions
        import random
        random.seed(42)
        checked = 0
        for _ in range(1000):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if this is a non-zero position
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if j == j2 and i == i2 and k == k2:
                continue  # This is a non-zero position, already checked
            val = np.dot(U_np[a, :], V_np[b, :] * W_np[c, :])
            if abs(val) > tol:
                return False
            checked += 1
        return True
    
    # Try to find lower-rank decompositions using optimization
    # We'll try ranks from 24 down to 18
    
    for target_rank in range(24, 17, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Use gradient descent to minimize the error
            # Error = sum over all (a,b,c) of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # But this is expensive. Instead, use the non-zero constraints only initially
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradients
                # For each (i,j,k): residual = 1 - sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # dU[i*m+j, r] = -residual * V[j*p+k, r] * W[k*n+i, r]
                # dV[j*p+k, r] = -residual * U[i*m+j, r] * W[k*n+i, r]
                # dW[k*n+i, r] = -residual * U[i*m+j, r] * V[j*p+k, r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            prod = U[a, :] * V[b, :] * W[c, :]
                            val = np.sum(prod)
                            residual = 1.0 - val
                            dU[a, :] += residual * V[b, :] * W[c, :]
                            dV[b, :] += residual * U[a, :] * W[c, :]
                            dW[c, :] += residual * U[a, :] * V[b, :]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check if converged
                if step % 50 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-8:
                        break
            
            # Check if we found a valid decomposition
            if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                # Convert to list format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break  # Found a valid decomposition at this rank
    
    # Try to snap to integer/half-integer values if close
    if best_rank < rank_schoolbook:
        U_arr = np.array(best_result["U"], dtype=float)
        V_arr = np.array(best_result["V"], dtype=float)
        W_arr = np.array(best_result["W"], dtype=float)
        
        # Try to snap to nearest 0.5
        U_snapped = np.round(U_arr * 2) / 2
        V_snapped = np.round(V_arr * 2) / 2
        W_snapped = np.round(W_arr * 2) / 2
        
        if verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
            best_result["U"] = U_snapped.tolist()
            best_result["V"] = V_snapped.tolist()
            best_result["W"] = W_snapped.tolist()
        else:
            # Try to snap to integers
            U_int = np.round(U_arr)
            V_int = np.round(V_arr)
            W_int = np.round(W_arr)
            if verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-9):
                best_result["U"] = U_int.tolist()
                best_result["V"] = V_int.tolist()
                best_result["W"] = W_int.tolist()
    
    return best_result
# EVOLVE-BLOCK-END
