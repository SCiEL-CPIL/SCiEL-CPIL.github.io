import numpy as np
import time

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    def verify_decomposition(U, V, W, rank):
        """
        Verify if U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        We check against a set of random matrices.
        """
        # Use a few random tests to be sure
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            
            C_decomp = np.zeros((n, p))
            # Efficient check: 
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
            # We can vectorize this check
            
            # U_A[r, i, j] = U[i*m+j, r] * A[i,j]
            # V_B[r, j, k] = V[j*p+k, r] * B[j,k]
            
            # Let's do it explicitly for small sizes to avoid indexing bugs in verification
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        u_val = 0.0
                        for j in range(m):
                            u_val += U[i*m+j, r] * A[i,j]
                        v_val = 0.0
                        for j in range(m):
                            v_val += V[j*p+k, r] * B[j,k]
                        val += W[k*n+i, r] * u_val * v_val
                    C_decomp[i,k] = val
            
            if not np.allclose(C_expected, C_decomp, atol=1e-8):
                return False
        return True

    def round_to_half_int(arr):
        """Round array to nearest half-integer (0.5, 1.0, 1.5, etc) or 0."""
        # Multiply by 2, round, divide by 2
        return np.round(arr * 2) / 2.0

    def try_rank(rank, max_iters=200, tol=1e-10):
        """
        Attempt to find a rank 'rank' decomposition using alternating minimization.
        """
        # Initialize with a random decomposition of a slightly higher rank, 
        # then project/prune? 
        # Better: Initialize with a known valid high-rank decomposition (e.g. rank 32) 
        # and try to merge columns or optimize towards rank.
        
        # Actually, let's start from a random initialization but constrain the structure?
        # No, random init for tensor rank is hard.
        
        # Strategy: Start from the known Rank 32 solution, and try to "compress" it.
        # Or, start from Rank 36 (schoolbook) and optimize?
        
        # Let's try a different approach: 
        # The tensor is fixed. We want U, V, W of shape (4, r), (18, r), (18, r).
        # We can fix U and V to be random and solve for W? No, it's nonlinear.
        
        # Let's use the "Bilinear" minimization:
        # Minimize || T - sum_r u_r v_r^T w_r^T ||_F
        # This is non-convex.
        
        # Given the small size, let's try to find a solution by:
        # 1. Generating a random rank-31 decomposition? No, that's not guaranteed to be close.
        
        # Alternative Structural Idea:
        # Is rank 31 possible?
        # The tensor is M_{2,2,9}.
        # Lower bound for M_{2,2,p} is often related to M_{2,2,2} (7) + (p-2)*4?
        # If we use Strassen for the first 2 columns (rank 7) and trivial for remaining 7 (rank 28), total 35.
        # The current solution uses Strassen for pairs of columns?
        # Current: 4 pairs of columns (0,1), (2,3), (4,5), (6,7) use rank 7 each -> 28.
        # Last column 8 uses rank 4. Total 32.
        
        # Can we do better?
        # What if we use a different algorithm for 2x2x3 or 2x2x4 blocks?
        # M_{2,2,3} rank? 
        # M_{2,2,4} rank?
        
        # Let's try to search for a rank 31 decomposition using local optimization on the coefficients
        # of a specific structured form.
        
        # However, a more robust "different strategy" is to use **Gröbner Bases** or **Solving Linear Systems** if we fix the structure?
        # No, let's stick to numerical optimization but with a twist:
        # We know the entries must be integers/half-integers.
        
        # Let's try to construct a rank 31 solution by perturbing the rank 32 solution 
        # and checking if one column can be eliminated or if two columns can be merged.
        
        # Actually, let's look at the "Inspiration" which achieved 1.125 (Rank 32 vs 36).
        # Score = 36/32 = 1.125.
        # If we find Rank 31, Score = 36/31 = 1.161.
        # If we find Rank 30, Score = 36/30 = 1.200.
        
        # Is Rank 31 possible?
        # The minimal rank of M_{2,2,9} is unknown in general, but likely > 31?
        # Actually, for M_{2,2,p}, the rank is at least 4p - something?
        # Lower bound: 4p - 4? No.
        # For M_{2,2,2}, rank 7.
        # For M_{2,2,3}, rank is 10? (7 + 3? No, 7 + 4 = 11? Or better?)
        # If M_{2,2,3} has rank 10, then 3 blocks of 3 would be 30.
        # 9 = 3+3+3.
        # If M_{2,2,3} rank is 10, then 3*10 = 30.
        # If M_{2,2,3} rank is 11, then 33.
        
        # Let's check M_{2,2,3} rank.
        # Standard: 4*3 = 12.
        # Strassen-like: 
        # Can we do better than 11 for 2x2x3?
        # A 2x2x3 tensor.
        # If we use Strassen for first 2 cols (rank 7) and 1 col (rank 4), total 11.
        # Is there a rank 10 decomposition for 2x2x3?
        # If yes, we can use 3 blocks of 2x2x3 to get rank 30.
        
        # Let's try to find a rank 10 decomposition for 2x2x3.
        # If we find it, we can build a rank 30 solution for 2x2x9.
        
        def build_2x2x3_rank10():
            # Search for a rank 10 decomposition of M_{2,2,3}
            # U: (4, 10), V: (6, 10), W: (6, 10)
            # We can try to solve this by assuming a specific structure or using random search.
            
            # Let's try to initialize with a random rank 11 solution (Strassen + Trivial) 
            # and try to minimize the rank by setting one column to zero?
            # No, we need to find a *different* solution.
            
            # Let's use a numerical solver.
            # We want U, V, W such that T = sum u v w.
            
            # Since the space is small, we can try random initializations of U, V, W 
            # and use gradient descent to minimize the error, then round.
            
            best_err = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for attempt in range(20):
                np.random.seed(attempt)
                # Initialize with random values
                U = np.random.randn(4, 10)
                V = np.random.randn(6, 10)
                W = np.random.randn(6, 10)
                
                # Normalize to avoid scaling issues
                for r in range(10):
                    norm_u = np.linalg.norm(U[:, r])
                    norm_v = np.linalg.norm(V[:, r])
                    norm_w = np.linalg.norm(W[:, r])
                    if norm_u > 0 and norm_v > 0 and norm_w > 0:
                        U[:, r] /= norm_u
                        V[:, r] /= norm_v
                        W[:, r] /= norm_w
                        # Distribute scale to W? Or keep balanced?
                        # Let's just do simple alternating least squares.
                        
                # Alternating Least Squares
                for it in range(100):
                    # Fix U, V, solve for W
                    # T_{i,k,j} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                    # This is linear in W for fixed U, V.
                    # For each r, W[:, r] is a vector.
                    # Actually, the equation is:
                    # T = sum_r (U_r outer V_r) * W_r^T? No.
                    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                    
                    # Let's vectorize the tensor.
                    # T is (2,2,3) -> (4, 6) matrix? No, it's a 3rd order tensor.
                    # Let's construct the linear system for W.
                    
                    # For each r, let A_r = U[:, r] (4x1), B_r = V[:, r] (6x1).
                    # The term is sum_r (A_r * B_r^T) * W_r? 
                    # No, the indices are mixed.
                    
                    # Let's stick to the explicit loop for solving W, it's small.
                    # For each r, W[:, r] is unknown.
                    # T_{i,k} (for fixed j? No, j is summed over in multiplication, but here j is an index of A and B).
                    # Wait, the tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
                    # So T is defined for all i,j,k.
                    # The decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
                    
                    # For fixed r, let u_r = U[:, r] (4x1), v_r = V[:, r] (6x1), w_r = W[:, r] (6x1).
                    # The contribution to T from rank r is:
                    # T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
                    
                    # We want to minimize || T - sum_r T_r ||^2.
                    # This is not linear in w_r if u_r and v_r are fixed?
                    # T_r[i,j,k] = (u_r[i,j] * v_r[j,k]) * w_r[k,i].
                    # Let M_r[i,j,k] = u_r[i,j] * v_r[j,k].
                    # Then T_r = M_r * w_r (element-wise product with w_r permuted?).
                    # w_r has indices k,i. M_r has i,j,k.
                    # So T_r[i,j,k] = M_r[i,j,k] * w_r[k,i].
                    
                    # We can solve for w_r by linear least squares.
                    # For a fixed r, we want to find w_r that minimizes 
                    # || T - (sum_{s!=r} T_s) - T_r(w_r) ||^2.
                    # Let R = T - sum_{s!=r} T_s.
                    # We want to minimize || R - M_r * w_r ||^2.
                    # This is a linear system for w_r.
                    
                    # Let's implement this.
                    
                    # Precompute M_r for all r?
                    # M_r is 2x2x3 = 12 elements.
                    # w_r is 2x2 = 4 elements (indices k,i).
                    # Wait, W shape is (p*n, rank) = (6, rank).
                    # Index k*n+i. So W[k*n+i, r].
                    # So w_r is a vector of length 6?
                    # No, W is (p*n, rank). p=3, n=2. So 6 rows.
                    # But in the formula T_{i,j,k} = ... W_{k,i,r}.
                    # The indices of W are k and i.
                    # In the matrix W (6xR), the row index is k*n+i.
                    # So W[k*n+i, r] is the element.
                    
                    # So for a fixed r, we have a vector w of length 6.
                    # T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w[k*n+i].
                    
                    # Let's construct the matrix A_sys for the linear system.
                    # We have 12 equations (for each i,j,k) and 6 unknowns (w).
                    # A_sys[row, col] corresponds to:
                    # row: (i,j,k)
                    # col: (k', i') -> w[k'*n+i']
                    # Coefficient is u_r[i,j] * v_r[j,k] if (k', i') == (k, i) else 0.
                    # Wait, w is indexed by k,i.
                    # So for a specific (i,j,k), the term is u_r[i,j] * v_r[j,k] * w[k*n+i].
                    # So the coefficient for w[k*n+i] is u_r[i,j] * v_r[j,k].
                    # All other w components are 0.
                    
                    # This means the system is diagonal? No, because w is a vector.
                    # Actually, for each (i,j,k), only one component of w is involved: w[k*n+i].
                    # So we have 12 equations for 6 variables.
                    # Each variable w[k*n+i] appears in equations for all j.
                    # For fixed k,i, the equations are for j=0,1.
                    # R[i,0,k] = u_r[i,0] * v_r[0,k] * w[k*n+i]
                    # R[i,1,k] = u_r[i,1] * v_r[1,k] * w[k*n+i]
                    
                    # So we can solve for each w[k*n+i] independently using least squares on these 2 equations.
                    
                    # Let's do this.
                    
                    R = np.zeros((2,2,3))
                    for s in range(10):
                        if s == r: continue
                        for i in range(2):
                            for j in range(2):
                                for k in range(3):
                                    R[i,j,k] -= U[i*2+j, s] * V[j*3+k, s] * W[k*2+i, s]
                    
                    # Subtract from T
                    # T[i,j,k] = A[i,j]*B[j,k] is the target tensor.
                    # We need to represent T in the same shape.
                    # T_target[i,j,k] = A[i,j]*B[j,k]
                    # But we are decomposing the *multiplication tensor*, which is defined by the operation.
                    # The "tensor" is the linear map.
                    # So T_target[i,j,k] is 1 if the operation maps A[i,j] and B[j,k] to C[i,k].
                    # Actually, the tensor T has entries T_{i,j,k} = 1 if j is the shared index?
                    # No, the tensor for matrix multiplication M_{n,m,p} is defined by:
                    # T_{i,j,k} = delta_{j,j} ? No.
                    # The tensor is T such that (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k} T_{i,j,k}?
                    # No, standard definition: T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
                    # So T_{i,j,k} = 1 for all i,j,k?
                    # No, the tensor is the *structure* of the multiplication.
                    # T_{i,j,k} = 1.
                    # And the decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}.
                    # So the target tensor is all 1s?
                    # Yes! The matrix multiplication tensor has all entries 1.
                    # Because C_{i,k} = sum_j A_{i,j} B_{j,k}.
                    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
                    # Thus T_{i,j,k} = 1.
                    
                    # So R should be (1 - sum_{s!=r} ...).
                    
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                R[i,j,k] += 1.0 # Target is 1
                    
                    # Now solve for w
                    w_new = np.zeros(6)
                    for k in range(3):
                        for i in range(2):
                            idx = k*2+i
                            # Equations:
                            # R[i,0,k] = U[i*2+0, r] * V[0*3+k, r] * w[idx]
                            # R[i,1,k] = U[i*2+1, r] * V[1*3+k, r] * w[idx]
                            
                            coeff0 = U[i*2+0, r] * V[0*3+k, r]
                            coeff1 = U[i*2+1, r] * V[1*3+k, r]
                            target0 = R[i,0,k]
                            target1 = R[i,1,k]
                            
                            # Least squares for w[idx]:
                            # Minimize (coeff0 * w - target0)^2 + (coeff1 * w - target1)^2
                            # Derivative: 2*coeff0*(coeff0*w - target0) + 2*coeff1*(coeff1*w - target1) = 0
                            # w * (coeff0^2 + coeff1^2) = coeff0*target0 + coeff1*target1
                            
                            denom = coeff0**2 + coeff1**2
                            if denom > 1e-12:
                                w_new[idx] = (coeff0*target0 + coeff1*target1) / denom
                            else:
                                w_new[idx] = 0.0
                                
                    W[:, r] = w_new

                # After ALS, check error
                err = 0.0
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            val = 0.0
                            for r in range(10):
                                val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                            err += (val - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-10:
                    break

            if best_err < 1e-8:
                # Round to half-integers
                U_r = round_to_half_int(best_U)
                V_r = round_to_half_int(best_V)
                W_r = round_to_half_int(best_W)
                
                # Verify rounded version
                if verify_decomposition_2x2x3(U_r, V_r, W_r, 10):
                    return U_r, V_r, W_r
            return None

        def verify_decomposition_2x2x3(U, V, W, rank):
            for seed in [1, 2]:
                np.random.seed(seed)
                A = np.random.rand(2, 2)
                B = np.random.rand(2, 3)
                C_expected = A @ B
                C_decomp = np.zeros((2, 3))
                for i in range(2):
                    for k in range(3):
                        val = 0.0
                        for r in range(rank):
                            u_val = U[i*2+0, r]*A[i,0] + U[i*2+1, r]*A[i,1]
                            v_val = V[0*3+k, r]*B[0,k] + V[1*3+k, r]*B[1,k]
                            val += W[k*2+i, r] * u_val * v_val
                        C_decomp[i,k] = val
                if not np.allclose(C_expected, C_decomp, atol=1e-8):
                    return False
            return True

        # Try to find rank 10 for 2x2x3
        sol_2x2x3 = build_2x2x3_rank10()
        
        if sol_2x2x3 is not None:
            U_blk, V_blk, W_blk = sol_2x2x3
            # Build full 2x2x9 solution using 3 blocks
            rank = 30
            U = np.zeros((4, rank))
            V = np.zeros((18, rank))
            W = np.zeros((18, rank))
            
            for b in range(3):
                offset = b * 10
                U[:, offset:offset+10] = U_blk
                # V block: rows j*p+k. p=9.
                # Block b covers columns 3*b to 3*b+2.
                for j in range(2):
                    for k in range(3):
                        V[j*9 + (3*b+k), offset:offset+10] = V_blk[j*3+k]
                # W block: rows k*n+i.
                for k in range(3):
                    for i in range(2):
                        W[(3*b+k)*2 + i, offset:offset+10] = W_blk[k*2+i]
            
            if verify_decomposition(U, V, W, rank):
                return U.tolist(), V.tolist(), W.tolist()
                
        return None

    # Try to find a better rank than 32
    result_30 = try_rank(30)
    if result_30 is not None:
        return result_30

    # Fallback to known rank 32
    def build_rank32():
        rank = 32
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        col = 0

        for pair_idx in range(4):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1

            # m1
            U[0, col] = 1; U[3, col] = 1
            V[0*9+k1, col] = 1; V[1*9+k2, col] = 1
            W[k1*2+0, col] = 1
            W[k2*2+1, col] = 1
            col += 1

            # m2
            U[2, col] = 1; U[3, col] = 1
            V[0*9+k1, col] = 1
            W[k1*2+1, col] = 1
            W[k2*2+1, col] = -1
            col += 1

            # m3
            U[0, col] = 1; U[1, col] = -1
            V[0*9+k2, col] = 1; V[1*9+k2, col] = 1
            W[k2*2+0, col] = 1
            W[k2*2+1, col] = -1
            col += 1

            # m4
            U[2, col] = 1; U[3, col] = 1
            V[1*9+k1, col] = 1
            W[k1*2+0, col] = -1
            W[k2*2+0, col] = 1
            col += 1

            # m5
            U[0, col] = 1; U[1, col] = 1
            V[0*9+k1, col] = 1
            W[k1*2+1, col] = 1
            W[k2*2+1, col] = 1
            col += 1

            # m6
            U[2, col] = 1; U[0, col] = -1
            V[0*9+k1, col] = 1; V[1*9+k1, col] = 1
            W[k1*2+0, col] = 1
            col += 1

            # m7
            U[1, col] = 1; U[3, col] = -1
            V[0*9+k2, col] = 1; V[1*9+k2, col] = 1
            W[k1*2+0, col] = 1
            col += 1

        # Last column
        k = 8
        U[0, col] = 1; V[0*9+k, col] = 1; W[k*2+0, col] = 1; col += 1
        U[1, col] = 1; V[1*9+k, col] = 1; W[k*2+0, col] = 1; col += 1
        U[2, col] = 1; V[0*9+k, col] = 1; W[k*2+1, col] = 1; col += 1
        U[3, col] = 1; V[1*9+k, col] = 1; W[k*2+1, col] = 1; col += 1

        assert col == rank
        return U.tolist(), V.tolist(), W.tolist()

    U, V, W = build_rank32()
    return {"U": U, "V": V, "W": W}

