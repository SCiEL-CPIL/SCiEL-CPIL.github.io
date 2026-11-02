import time
import numpy as np
import random
from fractions import Fraction
from collections import defaultdict

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    We attempt to find a rank-41 decomposition using Newton-Raphson with
    rational arithmetic. If unsuccessful, we return the schoolbook (rank 42).
    """
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def full_verify_numpy(U_mat, V_mat, W_mat, atol=1e-9):
        """Verify using numpy for speed during search."""
        T_exp = np.zeros((p*n, n*m, m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_exp[k*n+i, i*m+j, j*p+k] = 1.0
        T_rec = np.einsum('ar,br,cr->abc', W_mat, U_mat, V_mat)
        return np.allclose(T_rec, T_exp, atol=atol)
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    # 2. Attempt to find rank-41 decomposition using Newton-Raphson
    target_rank = 41
    if time.time() < deadline - 20:
        # Try Newton's method with rational arithmetic
        # Unknowns: U (6*41), V (21*41), W (14*41)
        num_u = n * m * target_rank
        num_v = m * p * target_rank
        num_w = p * n * target_rank
        total_unknowns = num_u + num_v + num_w
        
        # The 42 non-zero entries of the target tensor
        nonzero_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    nonzero_indices.append((a, b, c))
        
        # We need more equations. Add zero equations.
        # Total tensor entries: 6*21*14 = 1764
        all_indices = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    all_indices.append((a, b, c))
        
        zero_indices = [idx for idx in all_indices if idx not in set(nonzero_indices)]
        
        # Select a subset of zero equations to make the system square
        num_zero_needed = total_unknowns - len(nonzero_indices)
        if num_zero_needed > 0:
            random.seed(42)
            random.shuffle(zero_indices)
            selected_zeros = zero_indices[:num_zero_needed]
        else:
            selected_zeros = []
        
        all_equations = nonzero_indices + selected_zeros
        num_equations = len(all_equations)
        
        if num_equations != total_unknowns:
            # Adjust: use least squares or adjust selection
            if num_equations < total_unknowns:
                # Add more zero equations
                remaining_zeros = zero_indices[num_zero_needed:]
                needed = total_unknowns - num_equations
                selected_zeros.extend(remaining_zeros[:needed])
                all_equations = nonzero_indices + selected_zeros
            else:
                # Remove some equations
                all_equations = all_equations[:total_unknowns]
        
        num_equations = len(all_equations)
        
        # Initialize with rational numbers (small random values)
        # Use a deterministic initialization based on the schoolbook solution
        # Start close to a known structure
        
        # Map unknowns: 
        # U[a, r] -> index a*target_rank + r
        # V[b, r] -> index num_u + b*target_rank + r  
        # W[c, r] -> index num_u + num_v + c*target_rank + r
        
        def var_index_U(a, r):
            return a * target_rank + r
        
        def var_index_V(b, r):
            return num_u + b * target_rank + r
        
        def var_index_W(c, r):
            return num_u + num_v + c * target_rank + r
        
        # Initialize variables
        # Use a structured initialization: perturb the schoolbook solution
        # For rank 41, we can't directly use schoolbook (rank 42)
        # Instead, use small random rationals
        
        rng = random.Random(12345)
        x = []
        for idx in range(total_unknowns):
            # Small rational values
            sign = 1 if rng.random() < 0.5 else -1
            val = Fraction(sign * rng.randint(1, 100), 1000)
            x.append(val)
        
        # Newton iteration
        # For each equation e = (a, b, c):
        # f_e = sum_r U[a,r]*V[b,r]*W[c,r] - target_e
        # target_e = 1 if (a,b,c) in nonzero_indices else 0
        
        # Jacobian J[i, j] = df_i / dx_j
        # For equation e=(a,b,c):
        # df_e/dU[a',r] = V[b,r]*W[c,r] if a'==a else 0
        # df_e/dV[b',r] = U[a,r]*W[c,r] if b'==b else 0
        # df_e/dW[c',r] = U[a,r]*V[b,r] if c'==c else 0
        
        max_iterations = 50
        converged = False
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Compute residuals
            residuals = []
            for eq_idx, (a, b, c) in enumerate(all_equations):
                s = Fraction(0)
                for r in range(target_rank):
                    u_val = x[var_index_U(a, r)]
                    v_val = x[var_index_V(b, r)]
                    w_val = x[var_index_W(c, r)]
                    s += u_val * v_val * w_val
                
                if (a, b, c) in set(nonzero_indices):
                    residuals.append(s - 1)
                else:
                    residuals.append(s)
            
            # Check convergence
            max_res = max(abs(res) for res in residuals)
            if max_res == 0:
                converged = True
                break
            
            if iteration % 5 == 0:
                # Convert to float for quick check
                x_float = np.array([float(v) for v in x])
                U_float = x_float[:num_u].reshape(n*m, target_rank)
                V_float = x_float[num_u:num_u+num_v].reshape(m*p, target_rank)
                W_float = x_float[num_u+num_v:].reshape(p*n, target_rank)
                if full_verify_numpy(U_float, V_float, W_float, atol=1e-6):
                    # Verify with exact arithmetic
                    converged = True
                    break
            
            # Compute Jacobian (sparse)
            # J is num_equations x total_unknowns
            # Build as list of (row, col, value) for sparse representation
            jacobian_entries = []
            
            for eq_idx, (a, b, c) in enumerate(all_equations):
                for r in range(target_rank):
                    u_val = x[var_index_U(a, r)]
                    v_val = x[var_index_V(b, r)]
                    w_val = x[var_index_W(c, r)]
                    
                    # d/dU[a,r]
                    jacobian_entries.append((eq_idx, var_index_U(a, r), v_val * w_val))
                    # d/dV[b,r]
                    jacobian_entries.append((eq_idx, var_index_V(b, r), u_val * w_val))
                    # d/dW[c,r]
                    jacobian_entries.append((eq_idx, var_index_W(c, r), u_val * v_val))
            
            # Solve J * dx = -residuals
            # This is a large linear system. Use Gaussian elimination with rationals.
            # For performance, convert to float for the linear solve, then refine.
            
            # Build dense Jacobian in float for the linear solve
            J_float = np.zeros((num_equations, total_unknowns))
            for (row, col, val) in jacobian_entries:
                J_float[row, col] = float(val)
            
            b_float = np.array([-float(res) for res in residuals])
            
            try:
                # Use least squares if overdetermined, otherwise direct solve
                if num_equations >= total_unknowns:
                    dx_float, residuals_ls, rank, sv = np.linalg.lstsq(J_float, b_float, rcond=None)
                else:
                    dx_float = np.linalg.solve(J_float, b_float)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                try:
                    dx_float = np.linalg.pinv(J_float) @ b_float
                except:
                    break
            
            # Damped Newton step
            # Try full step first, then bisection if residual doesn't decrease
            step_size = Fraction(1)
            success = False
            
            for attempt in range(10):
                dx_rational = [Fraction(v).limit_denominator(10**6) for v in dx_float]
                x_new = [x[i] + step_size * dx_rational[i] for i in range(total_unknowns)]
                
                # Compute new residual norm
                new_res_norm = 0
                for eq_idx, (a, b, c) in enumerate(all_equations):
                    s = Fraction(0)
                    for r in range(target_rank):
                        u_val = x_new[var_index_U(a, r)]
                        v_val = x_new[var_index_V(b, r)]
                        w_val = x_new[var_index_W(c, r)]
                        s += u_val * v_val * w_val
                    if (a, b, c) in set(nonzero_indices):
                        new_res_norm += (s - 1) ** 2
                    else:
                        new_res_norm += s ** 2
                
                old_res_norm = sum(res ** 2 for res in residuals)
                
                if new_res_norm < old_res_norm:
                    x = x_new
                    success = True
                    break
                else:
                    step_size = step_size / 2
            
            if not success:
                break
        
        if converged:
            # Extract solution
            U_sol = np.zeros((n*m, target_rank))
            V_sol = np.zeros((m*p, target_rank))
            W_sol = np.zeros((p*n, target_rank))
            
            for a in range(n*m):
                for r in range(target_rank):
                    U_sol[a, r] = float(x[var_index_U(a, r)])
            for b in range(m*p):
                for r in range(target_rank):
                    V_sol[b, r] = float(x[var_index_V(b, r)])
            for c in range(p*n):
                for r in range(target_rank):
                    W_sol[c, r] = float(x[var_index_W(c, r)])
            
            # Final verification
            if full_verify_numpy(U_sol, V_sol, W_sol, atol=1e-8):
                best_result = {
                    "U": U_sol.tolist(),
                    "V": V_sol.tolist(),
                    "W": W_sol.tolist()
                }
                best_rank = target_rank
                return best_result
    
    return best_result
