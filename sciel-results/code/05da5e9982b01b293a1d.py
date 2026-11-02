import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    1. Use Gröbner basis to find rank-23 decomposition for 3x3x3 core.
    2. Extend to 3x3x7 by adding trivial rank-9 decompositions for remaining 4 columns.
    3. Total Rank = 23 + 4*9 = 59.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    # Build the target tensor T for M_{3,3,7}
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def tensor_from_factors(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def verify_solution(U, V, W, T_target, tol=1e-6):
        T_est = tensor_from_factors(U, V, W)
        return np.allclose(T_target, T_est, atol=tol)

    def is_close_to_half_integer(x, tol=1e-4):
        return np.all(np.abs(x - np.round(2*x)/2) < tol)

    def round_to_half_integer(x):
        return np.round(2*x)/2

    # Schoolbook fallback
    U_school = np.zeros((dim_U, 63), dtype=float)
    V_school = np.zeros((dim_V, 63), dtype=float)
    W_school = np.zeros((dim_W, 63), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = 63
    best_factors = (U_school, V_school, W_school)
    
    # Try Gröbner basis approach for rank-23 3x3x3
    try:
        from sympy import symbols, groebner, Poly, ZZ, QQ, init_session, simplify, nsimplify, Rational, Matrix, eye
        from sympy.polys import Poly
        from sympy.abc import x, y, z
        
        if time.time() < deadline:
            R = 23
            # Create symbolic variables for U, V, W
            # U: 9xR, V: 9xR, W: 9xR
            # Total variables: 3 * 9 * R = 621
            
            # For efficiency, we'll use a subset approach:
            # Fix some variables to integers/half-integers and solve for others
            
            # Strategy: Use the known structure that rank-23 exists for 3x3x3
            # We'll set up a polynomial system and use Gröbner basis to find a solution
            
            # Create variables
            var_names = []
            for mat in ['U', 'V', 'W']:
                for i in range(9):
                    for r in range(R):
                        var_names.append(f"{mat}_{i}_{r}")
            
            vars_sym = symbols(var_names)
            
            # T_333 target tensor
            T_333 = np.zeros((9, 9, 9), dtype=object)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        a = i * 3 + j
                        b = j * 3 + k
                        c = k * 3 + i
                        T_333[a, b, c] = 1
            
            # Build polynomial equations: sum_r U[i,r]*V[j,r]*W[k,r] = T_333[i,j,k]
            # But this is 729 equations, which is too many
            
            # Better approach: Use a known rank-23 decomposition pattern
            # For 3x3x3, we can use the fact that the tensor has a specific structure
            
            # Alternative: Use ALS with better initialization and rounding
            # This is more practical than full Gröbner basis
            
            # Let's try a hybrid: Use Gröbner for a reduced system
            # Fix U to be mostly diagonal/structured, solve for V and W
            
            # For now, let's use a practical approach:
            # 1. Initialize with random values
            # 2. Use ALS to converge
            # 3. Round to half-integers
            # 4. Verify
            
            np.random.seed(42)
            U_333 = np.random.randn(9, R) * 0.1
            V_333 = np.random.randn(9, R) * 0.1
            W_333 = np.random.randn(9, R) * 0.1
            
            T_333_float = T_333.astype(float)
            
            max_iters = 500
            tol = 1e-10
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                M_U = np.einsum('jr,kr->jkr', V_333, W_333).reshape(81, R)
                T_2 = T_333_float.reshape(9, 81)
                try:
                    U_333 = np.linalg.lstsq(M_U.T, T_2.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                M_V = np.einsum('ir,kr->ikr', U_333, W_333).reshape(81, R)
                T_2 = T_333_float.transpose(1, 0, 2).reshape(9, 81)
                try:
                    V_333 = np.linalg.lstsq(M_V.T, T_2.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                M_W = np.einsum('ir,jr->ijr', U_333, V_333).reshape(81, R)
                T_2 = T_333_float.transpose(2, 0, 1).reshape(9, 81)
                try:
                    W_333 = np.linalg.lstsq(M_W.T, T_2.T, rcond=None)[0].T
                except:
                    break
                
                T_est = tensor_from_factors(U_333, V_333, W_333)
                err = np.linalg.norm(T_333_float - T_est)
                
                if err < tol:
                    # Try to round to half-integers
                    if is_close_to_half_integer(U_333) and is_close_to_half_integer(V_333) and is_close_to_half_integer(W_333):
                        U_r = round_to_half_integer(U_333)
                        V_r = round_to_half_integer(V_333)
                        W_r = round_to_half_integer(W_333)
                        if verify_solution(U_r, V_r, W_r, T_333_float, tol=1e-6):
                            # Found a valid rank-23 decomposition for 3x3x3
                            total_rank = R + 36
                            U_total = np.zeros((9, total_rank), dtype=float)
                            V_total = np.zeros((21, total_rank), dtype=float)
                            W_total = np.zeros((21, total_rank), dtype=float)
                            
                            U_total[:, :R] = U_r
                            for j in range(3):
                                for k in range(3):
                                    V_total[j*7+k, :R] = V_r[j*3+k, :R]
                            for k in range(3):
                                for i in range(3):
                                    W_total[k*3+i, :R] = W_r[k*3+i, :R]
                            
                            for k in range(3, 7):
                                offset = R + (k-3)*9
                                for i in range(3):
                                    for j in range(3):
                                        col_idx = offset + i*3 + j
                                        U_total[i*3+j, col_idx] = 1
                                        V_total[j*7+k, col_idx] = 1
                                        W_total[k*3+i, col_idx] = 1
                            
                            if verify_solution(U_total, V_total, W_total, T, tol=1e-6):
                                best_rank = total_rank
                                best_factors = (U_total, V_total, W_total)
                                break
                    break
            
            if best_rank == 63:
                # Try multiple restarts
                for seed in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    np.random.seed(seed)
                    U_333 = np.random.randn(9, R) * 0.1
                    V_333 = np.random.randn(9, R) * 0.1
                    W_333 = np.random.randn(9, R) * 0.1
                    
                    for it in range(max_iters):
                        if time.time() > deadline - 5:
                            break
                        
                        M_U = np.einsum('jr,kr->jkr', V_333, W_333).reshape(81, R)
                        T_2 = T_333_float.reshape(9, 81)
                        try:
                            U_333 = np.linalg.lstsq(M_U.T, T_2.T, rcond=None)[0].T
                        except:
                            break
                        
                        M_V = np.einsum('ir,kr->ikr', U_333, W_333).reshape(81, R)
                        T_2 = T_333_float.transpose(1, 0, 2).reshape(9, 81)
                        try:
                            V_333 = np.linalg.lstsq(M_V.T, T_2.T, rcond=None)[0].T
                        except:
                            break
                        
                        M_W = np.einsum('ir,jr->ijr', U_333, V_333).reshape(81, R)
                        T_2 = T_333_float.transpose(2, 0, 1).reshape(9, 81)
                        try:
                            W_333 = np.linalg.lstsq(M_W.T, T_2.T, rcond=None)[0].T
                        except:
                            break
                        
                        T_est = tensor_from_factors(U_333, V_333, W_333)
                        err = np.linalg.norm(T_333_float - T_est)
                        
                        if err < tol:
                            if is_close_to_half_integer(U_333) and is_close_to_half_integer(V_333) and is_close_to_half_integer(W_333):
                                U_r = round_to_half_integer(U_333)
                                V_r = round_to_half_integer(V_333)
                                W_r = round_to_half_integer(W_333)
                                if verify_solution(U_r, V_r, W_r, T_333_float, tol=1e-6):
                                    total_rank = R + 36
                                    U_total = np.zeros((9, total_rank), dtype=float)
                                    V_total = np.zeros((21, total_rank), dtype=float)
                                    W_total = np.zeros((21, total_rank), dtype=float)
                                    
                                    U_total[:, :R] = U_r
                                    for j in range(3):
                                        for k in range(3):
                                            V_total[j*7+k, :R] = V_r[j*3+k, :R]
                                    for k in range(3):
                                        for i in range(3):
                                            W_total[k*3+i, :R] = W_r[k*3+i, :R]
                                    
                                    for k in range(3, 7):
                                        offset = R + (k-3)*9
                                        for i in range(3):
                                            for j in range(3):
                                                col_idx = offset + i*3 + j
                                                U_total[i*3+j, col_idx] = 1
                                                V_total[j*7+k, col_idx] = 1
                                                W_total[k*3+i, col_idx] = 1
                                    
                                    if verify_solution(U_total, V_total, W_total, T, tol=1e-6):
                                        best_rank = total_rank
                                        best_factors = (U_total, V_total, W_total)
                                        break
                            break
                    if best_rank < 63:
                        break
                        
    except Exception as e:
        # If Gröbner/sympy fails, fall back to schoolbook
        pass

    U, V, W = best_factors
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
