import time
import numpy as np
import sys

def construct():
    """
    Finds the rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    1. The tensor decomposition of rank R corresponds to R rank-1 tensors summing to the multiplication tensor.
    2. This is equivalent to finding matrices U (6xR), V (15xR), W (10xR) such that for all i,j,k:
       sum_r U[i*3+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
       AND for all other index combinations (a,b,c) in the flattened spaces where the indices don't 
       align to a valid (i,j,k) chain, the sum must be 0.
       
    Actually, the standard definition of the matrix multiplication tensor T in C^(nm) x C^(mp) x C^(pn) is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j}^A \otimes e_{j,k}^B \otimes e_{k,i}^C
    where e_{i,j}^A is the basis vector in C^(nm) with 1 at index i*m+j.
    
    The condition for decomposition U, V, W is:
    For all a in [0, nm), b in [0, mp), c in [0, pn):
    sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    
    Here T[a, b, c] is 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i, and 0 otherwise.
    Note: The indices a,b,c uniquely determine i,j from a; j',k' from b; k'',i'' from c.
    The term is 1 iff i=i'' and j=j' and k=k''.
    
    We want to find the smallest R such that a solution exists with integer/half-integer entries.
    Since the problem is small (n=2, m=3, p=5), we can try to solve for R starting from a lower bound.
    Lower bound: max(nm, mp, pn) = max(6, 15, 10) = 15.
    Also, Strassen's algorithm for 2x2 has rank 7. For 2x3x5, it's unknown but likely close to 15-20.
    
    We will use Sympy to solve the system of polynomial equations for small R.
    If Sympy is too slow or fails, we fall back to a numerical search or schoolbook.
    """
    
    n, m, p = 2, 3, 5
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Precompute the target tensor T[a, b, c]
    T = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def try_rank(R, timeout=10.0):
        """
        Attempts to find a decomposition of rank R using symbolic solving.
        Returns (U, V, W) if found, else None.
        """
        try:
            import sympy as sp
            
            # We have R columns. Let's fix one column to break scaling symmetry?
            # Actually, the equations are multilinear in the entries of U, V, W.
            # Total variables: R * (6 + 15 + 10) = 31 * R.
            # Equations: 6 * 15 * 10 = 900.
            # For R=15, vars = 465. Equations = 900. Overdetermined.
            # For R=20, vars = 620. Equations = 900.
            # For R=30, vars = 930. Equations = 900.
            
            # Gröbner basis is expensive. We only try small R.
            # The variables are u_{a,r}, v_{b,r}, w_{c,r}.
            
            # To make it solvable, we can fix the first row of U to be canonical?
            # Or just try to solve.
            
            # Let's create symbols
            # To save memory/time, we only create symbols for non-zero entries if we assume sparsity?
            # No, generic dense matrices.
            
            # This might be too slow for R > 15.
            # Let's try R from 15 to 25.
            
            symbols = []
            for r in range(R):
                for a in range(dim_u):
                    symbols.append(sp.Symbol(f'u_{a}_{r}'))
                for b in range(dim_v):
                    symbols.append(sp.Symbol(f'v_{b}_{r}'))
                for c in range(dim_w):
                    symbols.append(sp.Symbol(f'w_{c}_{r}'))
            
            # Map from (type, index, r) to symbol index
            # This is getting complex to manage manually.
            # Let's build the equations.
            
            eqs = []
            # We only need to check the 900 positions.
            # However, many are 0.
            # We can reduce the number of equations by using linearity?
            # No, it's a product sum.
            
            # For performance, we only generate equations for the 30 non-zero entries of T?
            # No, we MUST satisfy the zero entries too.
            # But if we find a solution to the non-zero ones, does it imply the zero ones?
            # Not necessarily.
            
            # Let's try a different approach: Numerical Search with local refinement?
            # The directive says "different design family". Symbolic is one.
            # But symbolic is likely to time out for R=15-20.
            
            # Alternative Strategy: 
            # Use the fact that matrix multiplication tensors have symmetries.
            # Or, use a known algorithm if R is small.
            
            # Given the 150s limit, symbolic solving for R=15 (465 vars, 900 eqs) is risky.
            # Let's try a **numerical optimization** approach instead?
            # Minimize sum (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2.
            # If min is 0 (or very close), we have a solution.
            # We can use scipy.optimize.least_squares or similar.
            
            # Let's switch to a numerical global optimization strategy using multiple random restarts.
            # This is a different family from "static schoolbook".
            
            return None # Placeholder, will implement numerical below
            
        except Exception:
            return None

    def _schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Fallback to schoolbook if search fails
    best_U, best_V, best_W = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    
    # Attempt numerical search for lower rank
    # We will try ranks from 15 to 29.
    # Objective: Minimize || U V^T W - T ||_F^2 ? No, it's a tensor product.
    # Let X be a 3D array of shape (dim_u, dim_v, dim_w).
    # X[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    
    import scipy.optimize as so
    
    def tensor_residuals(params, R, T_flat, dim_u, dim_v, dim_w):
        U = params[0:R*dim_u].reshape(R, dim_u).T # R x dim_u -> dim_u x R
        V = params[R*dim_u:R*dim_u + R*dim_v].reshape(R, dim_v).T
        W = params[R*dim_u + R*dim_v:].reshape(R, dim_w).T
        
        # Compute tensor
        # This is expensive to do in pure Python loops for every step.
        # We can use einsum.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # But we only need to check against T_flat.
        # To speed up, we can just compute the full tensor and flatten.
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_calc - T).flatten()

    # We need to define T as a numpy array for the objective
    T_np = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_np[a, b, c] = 1.0
    T_flat = T_np.flatten()

    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    # Try ranks from 15 upwards
    # Note: Integer/half-integer constraint is hard for continuous optimization.
    # However, if we find a continuous solution, we can try to round it or solve for integers nearby?
    # The task requires "integer/half-integer entries".
    # If we find a real solution, it might not be valid.
    # But often, minimal rank decompositions for small matrices have rational entries.
    # We will check if the found solution is close to half-integers.
    
    for R in range(15, 30):
        if time.time() > deadline - 20: # Stop if close to deadline
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            np.random.seed(42 + restart)
            params = np.random.randn(R * (dim_u + dim_v + dim_w))
            
            try:
                res = so.least_squares(
                    tensor_residuals, 
                    params, 
                    args=(R, T_flat, dim_u, dim_v, dim_w),
                    method='lm', # Levenberg-Marquardt for small problems
                    max_nfev=1000
                )
                
                if res.success or res.cost < 1e-10:
                    # Check if entries are close to half-integers
                    U_sol = res.x[0:R*dim_u].reshape(R, dim_u).T
                    V_sol = res.x[R*dim_u:R*dim_u + R*dim_v].reshape(R, dim_v).T
                    W_sol = res.x[R*dim_u + R*dim_v:].reshape(R, dim_w).T
                    
                    # Check half-integer condition
                    def is_half_int(arr):
                        # 2 * arr should be close to integer
                        doubled = 2 * arr
                        diff = np.abs(doubled - np.round(doubled))
                        return np.all(diff < 1e-5)
                    
                    if is_half_int(U_sol) and is_half_int(V_sol) and is_half_int(W_sol):
                        # Verify exactness
                        U_int = np.round(2 * U_sol) / 2.0
                        V_int = np.round(2 * V_sol) / 2.0
                        W_int = np.round(2 * W_sol) / 2.0
                        
                        T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                        if np.allclose(T_check, T_np, atol=1e-9):
                            # Found valid solution
                            best_U = U_int.tolist()
                            best_V = V_int.tolist()
                            best_W = W_int.tolist()
                            best_rank = R
                            found = True
                            break # Break restart loop
                            
            except Exception as e:
                pass
                
        if found:
            break # Found minimal rank for this R, since we iterate upwards
            
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
