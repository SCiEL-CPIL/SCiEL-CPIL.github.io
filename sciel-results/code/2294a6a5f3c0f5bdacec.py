import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Total constraints: n * m * p * m = 3 * 4 * 5 * 4 = 240
    # Unknowns: R * dim_w = 15 * R
    
    # We want to find R < 60
    # Start from 59 down to 40
    # Time limit 150s, keep safety margin
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Pre-compute indices for the linear system
    # Equation index: e = i * (m * p * m) + k * (m * m) + j * m + j_prime
    # Wait, the constraint is for each i, k, j, j_prime:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j_prime*p+k, r] = delta[j, j_prime]
    
    # We can flatten W into a vector w of size R * dim_w
    # w[r * dim_w + (k*n+i)] = W[k*n+i, r]
    
    # The coefficient for w[idx] in equation (i, k, j, j_prime) is:
    # U[i*m+j, r] * V[j_prime*p+k, r] if idx corresponds to (r, k*n+i)
    
    # Let's build the matrix A (size 240 x (R * dim_w))
    # And vector b (size 240)
    
    def build_system(R, U, V):
        A = np.zeros((n * m * p * m, R * dim_w))
        b = np.zeros(n * m * p * m)
        
        eq_idx = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        # Target: delta[j, jp]
                        b[eq_idx] = 1.0 if j == jp else 0.0
                        
                        # Coefficients for each r
                        for r in range(R):
                            u_val = U[i * m + j, r]
                            v_val = V[jp * p + k, r]
                            if u_val != 0 and v_val != 0:
                                # Index in w: r * dim_w + (k * n + i)
                                w_idx = r * dim_w + (k * n + i)
                                A[eq_idx, w_idx] = u_val * v_val
                        eq_idx += 1
        return A, b

    def check_solution(W, U, V, tol=1e-9):
        # Verify exact decomposition
        # C_ik = sum_j A_ij B_jk
        # Check: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_ij) * (sum_jp V[jp*p+k, r] B_jp*k)
        # This is equivalent to checking the tensor identity.
        # For speed, we can just check the residual of the linear system used to find it,
        # but re-verifying is safer.
        
        # Actually, since we solved the linear system, the residual is the error.
        # But floating point errors might accumulate.
        # Let's check a few random instances.
        
        # Better: check the tensor identity directly on the coefficients.
        # We already built the system. The residual is the error.
        pass

    best_result = None
    best_rank = 60  # Default to schoolbook if search fails
    
    # Generate schoolbook as fallback
    U_fb = [[0] * 60 for _ in range(dim_u)]
    V_fb = [[0] * 60 for _ in range(dim_v)]
    W_fb = [[0] * 60 for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Search for better rank
    for R in range(59, 39, -1):
        if time.time() > deadline:
            break
            
        # Try several random initializations
        max_attempts = 500
        attempts = 0
        found = False
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random sparse U and V with entries -1, 0, 1
            # To help find integer solutions, keep them sparse
            U = np.random.choice([-1, 0, 1], size=(dim_u, R))
            V = np.random.choice([-1, 0, 1], size=(dim_v, R))
            
            # Ensure non-zero columns? Not strictly necessary, but helps rank.
            # If a column of U is 0, the corresponding term in W is unused.
            
            A, b = build_system(R, U, V)
            
            # Solve least squares
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check residual
            if residuals and residuals[0] < 1e-9:
                # Reshape w to W
                W = w.reshape(R, dim_w).T  # W is dim_w x R
                W = W.tolist()
                
                # Check if W entries are close to integers
                # If not, we might still accept if they are simple fractions, 
                # but the task says "integer/half-integer". 
                # Let's check if 2*W is integer.
                is_valid = True
                for i in range(dim_w):
                    for r in range(R):
                        val = W[i][r]
                        # Check if close to integer
                        if abs(val - round(val)) > 1e-7:
                            # Check if close to half-integer
                            if abs(val * 2 - round(val * 2)) > 1e-7:
                                is_valid = False
                                break
                    if not is_valid:
                        break
                
                if is_valid:
                    # Snap to nearest integer or half-integer
                    W_snapped = [[0] * R for _ in range(dim_w)]
                    for i in range(dim_w):
                        for r in range(R):
                            val = W[i][r]
                            if abs(val - round(val)) < 1e-7:
                                W_snapped[i][r] = int(round(val))
                            else:
                                # Half integer
                                W_snapped[i][r] = round(val * 2) / 2.0
                                # Store as float or fraction? 
                                # Task says "integer/half-integer entries".
                                # Python floats are fine if they are exact.
                                
                    # Final verification with snapped values
                    # Rebuild system with snapped W? No, just verify the identity.
                    # Let's verify one random case to be sure.
                    
                    # Create random A, B
                    A_mat = np.random.randint(-5, 5, size=(n, m))
                    B_mat = np.random.randint(-5, 5, size=(m, p))
                    C_true = A_mat @ B_mat
                    
                    # Compute using decomposition
                    C_calc = np.zeros((n, p))
                    for r in range(R):
                        # term_r = (U_r dot A) * (V_r dot B) * W_r
                        # U_r is vector of length n*m
                        # V_r is vector of length m*p
                        # W_r is scalar for each (k,i)
                        
                        u_vec = np.array([U[i * m + j, r] for i in range(n) for j in range(m)]).reshape(n, m)
                        v_vec = np.array([V[j * p + k, r] for j in range(m) for k in range(p)]).reshape(m, p)
                        
                        # The decomposition is:
                        # C_ik += W[k*n+i, r] * (sum_j U[i*m+j, r] A_ij) * (sum_jp V[jp*p+k, r] B_jp*k)
                        # This is equivalent to:
                        # C += W_r * (A * U_r) * (V_r * B) ?
                        # No, U_r and V_r are coefficient matrices.
                        # Let's just iterate.
                        
                        for i in range(n):
                            for k in range(p):
                                w_val = W_snapped[k * n + i][r]
                                if w_val != 0:
                                    sum_u = 0
                                    for j in range(m):
                                        sum_u += U[i * m + j, r] * A_mat[i, j]
                                    sum_v = 0
                                    for jp in range(m):
                                        sum_v += V[jp * p + k, r] * B_mat[jp, k]
                                    C_calc[i, k] += w_val * sum_u * sum_v
                                        
                    if np.allclose(C_true, C_calc, atol=1e-7):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W_snapped}
                        
            if time.time() > deadline - 10:
                break
                
    # If search failed, return schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}

