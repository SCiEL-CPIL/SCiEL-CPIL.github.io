import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are half-integers (integer or integer+0.5)."""
    for row in M:
        for x in row:
            if abs(x * 2 - round(x * 2)) > tol:
                return False
    return True

def _try_strassen_style(n, m, p, deadline):
    """
    Attempt to construct a low-rank decomposition using Strassen-like block structure.
    For 3x3 x 3x5, we can try to exploit the rectangular structure.
    """
    # For a 3x3 x 3x5 matrix multiplication, the tensor has shape (3,3,5)
    # We can try to decompose it using a combination of:
    # 1. Direct multiplication for some entries
    # 2. Strassen-like tricks for others
    
    # One approach: treat the 3x5 output as three 3x(5/3) blocks, but 5 isn't divisible by 3
    # So we need a more careful approach.
    
    # Alternative: use the fact that we can write C = A*B where A is 3x3 and B is 3x5
    # The (i,k) entry of C is sum_j A[i,j]*B[j,k]
    
    # Let's try a rank-20 decomposition using a structured approach:
    # We'll use 15 "Strassen-like" terms plus 5 correction terms
    
    # Actually, let's try a simpler approach: use the known rank bound
    # For m x n x p matrix multiplication, the rank is at most m*n*p
    # and at least max(m*n, m*p, n*p)
    
    # For 3x3x5: lower bound is max(9, 15, 10) = 15
    # Let's try to find a rank-15 or rank-20 decomposition
    
    # Structured approach: 
    # The matrix multiplication tensor T[i,j,k] = delta[i,j,k] (in the appropriate basis)
    # We want T = sum_r U_r * V_r * W_r
    
    # For rank 15, we need 15 terms. Let's try a systematic construction.
    
    # One idea: use the fact that 3x3 matrix multiplication has rank 21 (Strassen)
    # and 3x5 can be decomposed into 3x3 + 3x2
    
    # Let's try a different approach: solve the linear system exactly for small ranks
    
    return None

def _try_rank_r_exact(R, n, m, p, deadline, max_restarts=500):
    """
    Try to find a rank-R decomposition using exact rational arithmetic.
    Fix U and V, solve for W using rational linear algebra.
    """
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 7919 + R * 104729)
        
        # Initialize U and V with small integers from {-2, -1, 0, 1, 2}
        # Use structured patterns to increase chances of success
        
        if restart % 5 == 0:
            # Pattern 1: sparse with structure
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            for i in range(n*m):
                num_nonzero = min(R, np.random.randint(1, 4))
                cols = np.random.choice(R, size=num_nonzero, replace=False)
                U[i, cols] = np.random.choice([-1, 1], size=num_nonzero)
            for j in range(m*p):
                num_nonzero = min(R, np.random.randint(1, 4))
                cols = np.random.choice(R, size=num_nonzero, replace=False)
                V[j, cols] = np.random.choice([-1, 1], size=num_nonzero)
        elif restart % 5 == 1:
            # Pattern 2: denser
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
        elif restart % 5 == 2:
            # Pattern 3: very sparse
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            for i in range(n*m):
                col = np.random.randint(R)
                U[i, col] = np.random.choice([-1, 1])
            for j in range(m*p):
                col = np.random.randint(R)
                V[j, col] = np.random.choice([-1, 1])
        elif restart % 5 == 3:
            # Pattern 4: block structure
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            for r in range(R):
                # Each column of U and V has a specific pattern
                U[np.random.choice(n*m, size=min(3, n*m), replace=False), r] = np.random.choice([-1, 1], size=min(3, n*m))
                V[np.random.choice(m*p, size=min(3, m*p), replace=False), r] = np.random.choice([-1, 1], size=min(3, m*p))
        else:
            # Pattern 5: mixed
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            # Ensure no all-zero rows
            for i in range(n*m):
                if np.all(U[i] == 0):
                    U[i, np.random.randint(R)] = 1
            for j in range(m*p):
                if np.all(V[j] == 0):
                    V[j, np.random.randint(R)] = 1
        
        # Build the linear system A * W_vec = b using exact fractions
        # A is (n*m*p) x (p*n*R)
        # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Use numpy for the linear system, but check with fractions for verification
        A = np.zeros((n*m*p, p*n*R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    for r in range(R):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            var_idx = (k*n + i) * R + r
                            A[eq_idx, var_idx] = float(coeff)
        
        b = np.ones(n*m*p)
        
        # Solve using least squares
        try:
            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact
        error = np.linalg.norm(A @ W_vec - b)
        if error > 1e-8:
            continue
        
        W_mat = W_vec.reshape(p*n, R)
        
        # Check if W is close to half-integers
        W_scaled = W_mat * 2
        W_rounded = np.round(W_scaled)
        if np.allclose(W_scaled, W_rounded, atol=1e-6):
            W_final = W_rounded / 2.0
            # Final verification with exact arithmetic
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_frac = np.round(W_final * 2).astype(int) / 2.0
            
            # Verify using integer arithmetic where possible
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # All should be exact
                            term = U_int[i*m+j, r] * V_int[j*p+k, r] * W_frac[k*n+i, r]
                            val += term
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U_int, V_int, W_frac
        
        # Try null space perturbations
        if rank < p*n*R and rank > 0:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
            null_space = Vt[rank:]
            
            for trial in range(30):
                if time.time() > deadline:
                    break
                c = np.random.randn(null_space.shape[0])
                W_trial = W_vec + null_space.T @ c
                W_trial_mat = W_trial.reshape(p*n, R)
                W_trial_scaled = W_trial_mat * 2
                W_trial_rounded = np.round(W_trial_scaled)
                if np.allclose(W_trial_scaled, W_trial_rounded, atol=1e-6):
                    W_trial_final = W_trial_rounded / 2.0
                    U_int = np.round(U).astype(int)
                    V_int = np.round(V).astype(int)
                    W_frac = np.round(W_trial_final * 2).astype(int) / 2.0
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for r in range(R):
                                    term = U_int[i*m+j, r] * V_int[j*p+k, r] * W_frac[k*n+i, r]
                                    val += term
                                if abs(val - 1) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return U_int, V_int, W_frac
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Keep margin before 150s limit
    
    R_max = n * m * p  # 45
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n*m, R_max))
    V_sb = np.zeros((m*p, R_max))
    W_sb = np.zeros((p*n, R_max))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = R_max
    
    # Try to find lower rank solutions
    # Theoretical lower bound for 3x3 x 3x5 is at least max(3*3, 3*5, 3*3) = 15
    
    # Try ranks from 15 up to R_max-1
    # Start with smaller ranks first to find the minimum
    for R in range(15, R_max):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_r_exact(R, n, m, p, deadline - 10)
        if result is not None:
            U, V, W = result
            best_U = U
            best_V = V
            best_W = W
            best_rank = R
            break
    
    # If we didn't find a solution with rank < R_max, try some higher ranks
    # that might be easier to find
    if best_rank == R_max:
        for R in range(R_max - 1, 14, -1):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_r_exact(R, n, m, p, deadline - 5)
            if result is not None:
                U, V, W = result
                best_U = U
                best_V = V
                best_W = W
                best_rank = R
                break
    
    # Convert to lists
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
