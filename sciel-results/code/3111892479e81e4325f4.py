"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the multiplication tensor T[i,j,k] = 1
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the tensor."""
        r = len(U[0])
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape for computation
        # T[i,j,k] should equal 1
        # T = sum_c outer(U[:,c], outer(V[:,c], W[:,c])) reshaped appropriately
        # U[:,c] is vector of length nm, V[:,c] length mp, W[:,c] length pn
        # T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Build T by summing rank-1 tensors
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        
        return np.allclose(T, np.ones((n,m,p)), atol=1e-9)
    
    def to_half_int_matrix(M):
        """Convert matrix to half-integer representation, return None if not half-integer."""
        # Check if all entries are half-integers (multiples of 0.5)
        M_float = np.array(M, dtype=float)
        M_2x = M_float * 2
        if not np.allclose(M_2x, np.round(M_2x), atol=1e-9):
            return None
        return [[int(round(M_2x[i][j] / 2 * 2) / 2) for j in range(len(M_2x[0]))] for i in range(len(M_2x))]
    
    # Try ranks from a lower bound up to schoolbook
    # Known lower bound for 3x3x3 is 19 (Strassen-like), but for 3x3x5 it's different
    # Let's try from rank 20 upward
    
    # First, try the schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(nm)]
    V_school = [[0] * rank_schoolbook for _ in range(mp)]
    W_school = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Use a randomized approach: fix some entries, solve for others
    
    def try_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization with small integers/half-integers
            # Use a structured approach: initialize with random values in {-1, -0.5, 0, 0.5, 1}
            
            # We'll use a simple iterative approach
            # Initialize U, V, W randomly
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, r)).astype(float)
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, r)).astype(float)
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, r)).astype(float)
            
            # Ensure at least some non-zero entries
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue
            
            # Compute current tensor
            T_current = np.zeros((n, m, p))
            for c in range(r):
                u_c = U[:, c].reshape(n, m)
                v_c = V[:, c].reshape(m, p)
                w_c = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_current[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            
            # Check if we're close to target
            if np.allclose(T_current, np.ones((n,m,p)), atol=1e-6):
                # Verify exactly with fractions
                U_frac = [[Fraction(int(round(U[i][c]*2)))/2 for c in range(r)] for i in range(nm)]
                V_frac = [[Fraction(int(round(V[i][c]*2)))/2 for c in range(r)] for i in range(mp)]
                W_frac = [[Fraction(int(round(W[i][c]*2)))/2 for c in range(r)] for i in range(pn)]
                
                # Exact verification
                exact_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = Fraction(0)
                            for c in range(r):
                                val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                            if val != 1:
                                exact_ok = False
                                break
                        if not exact_ok:
                            break
                    if not exact_ok:
                        break
                
                if exact_ok:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
            
            # Simple gradient-like update: nudge entries to reduce error
            error = T_current - np.ones((n,m,p))
            
            # Compute gradients (simplified)
            # dT/dU[i,j,c] = V[j*p+k,c] * W[k*n+i,c] summed over k
            # This is getting complex; let's use a simpler local search
            
            # Local search: perturb one entry at a time
            improved = False
            for _ in range(50):  # local search steps
                if time.time() > deadline:
                    return None
                
                # Pick a random matrix and entry
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    mat = U
                    mat_name = 'U'
                elif mat_choice == 'V':
                    mat = V
                    mat_name = 'V'
                else:
                    mat = W
                    mat_name = 'W'
                
                idx = random.randint(0, mat.shape[0]-1)
                col = random.randint(0, mat.shape[1]-1)
                
                old_val = mat[idx, col]
                # Try small perturbations
                for delta in [-1, -0.5, 0.5, 1]:
                    new_val = old_val + delta
                    if new_val not in [-1, -0.5, 0, 0.5, 1]:
                        continue
                    mat[idx, col] = new_val
                    
                    # Recompute error
                    T_new = np.zeros((n, m, p))
                    for c in range(r):
                        u_c = U[:, c].reshape(n, m)
                        v_c = V[:, c].reshape(m, p)
                        w_c = W[:, c].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_new[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
                    
                    new_err = np.sum((T_new - 1)**2)
                    old_err = np.sum((T_current - 1)**2)
                    
                    if new_err < old_err - 1e-9:
                        improved = True
                        T_current = T_new
                        break
                    else:
                        mat[idx, col] = old_val
            
            if not improved:
                break
            
            # Check convergence
            if np.allclose(T_current, np.ones((n,m,p)), atol=1e-8):
                # Verify exactly
                U_frac = [[Fraction(int(round(U[i][c]*2)))/2 for c in range(r)] for i in range(nm)]
                V_frac = [[Fraction(int(round(V[i][c]*2)))/2 for c in range(r)] for i in range(mp)]
                W_frac = [[Fraction(int(round(W[i][c]*2)))/2 for c in range(r)] for i in range(pn)]
                
                exact_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = Fraction(0)
                            for c in range(r):
                                val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                            if val != 1:
                                exact_ok = False
                                break
                        if not exact_ok:
                            break
                    if not exact_ok:
                        break
                
                if exact_ok:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    # Try ranks from 20 up to schoolbook-1
    for r in range(20, rank_schoolbook):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_attempts=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # Convert to list of lists with integer/half-integer values
    def convert_to_lists(mat_frac):
        result = []
        for row in mat_frac:
            new_row = []
            for val in row:
                # Convert Fraction to int or float (half-integer)
                if val.denominator == 1:
                    new_row.append(val.numerator)
                else:
                    new_row.append(float(val))
            result.append(new_row)
        return result
    
    U_out = convert_to_lists(best_result["U"])
    V_out = convert_to_lists(best_result["V"])
    W_out = convert_to_lists(best_result["W"])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
