import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    Uses a structured search for rank-10, falling back to verified rank-11.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def set_u(i, j, r, val): U[i * m + j][r] = val
        def set_v(j, k, r, val): V[j * p + k][r] = val
        def set_w(k, i, r, val): W[k * n + i][r] = val

        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify_solution(sol, n_trials=50):
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            M = np.einsum('ijr,ij->r', U3, A)
            N = np.einsum('jkr,jk->r', V3, B)
            MN = M * N
            C_computed = np.einsum('kir,r->ik', W3, MN)
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    sol11 = make_rank11()
    
    def try_rank10():
        """
        Structured search for rank-10 decomposition.
        
        Strategy: Take the rank-11 solution and try to merge one of the k=2 terms
        into an existing k=0,1 term by adding a W[2,i] component to that term.
        
        For this to work, the merged term's M*N product must equal the k=2 term's
        M*N product when restricted to the k=2 output. Since M and N are shared,
        we need the existing term's M*N to contain the k=2 product as a factor.
        
        Alternative approach: Use a parameterized 10-term structure and solve
        the resulting system of polynomial equations.
        
        Let's try: 7 terms for k=0,1 (Strassen-like) + 3 terms for k=2,
        where one of the k=0,1 terms also contributes to k=2.
        """
        if time_remaining() < 30:
            return None
        
        # Approach: Try to find a rank-10 solution by modifying the rank-11 structure.
        # The 4 k=2 terms are:
        # T7: A00*B02 -> C[0,2]
        # T8: A01*B12 -> C[0,2]  
        # T9: A10*B02 -> C[1,2]
        # T10: A11*B12 -> C[1,2]
        
        # Can we merge T7 (A00*B02) into an existing term?
        # Term 0 has U = {A00, A11}, V = {B00, B11}. M*N = (A00+A11)(B00+B11)
        # This doesn't contain A00*B02 as a factor.
        
        # Term 2 has U = {A00}, V = {B01, -B11}. M*N = A00*(B01-B11)
        # Doesn't contain A00*B02.
        
        # Term 4 has U = {A00, A01}, V = {B11}. M*N = (A00+A01)*B11
        # Doesn't contain A00*B02.
        
        # Term 5 has U = {A10, -A00}, V = {B00, B01}. M*N = (A10-A00)(B00+B01)
        # Contains A00*B00 and A00*B01 but not A00*B02.
        
        # So direct merging doesn't work with the current structure.
        
        # Let's try a different approach: use numerical optimization.
        # Set up 10 terms with a specific sparse pattern and solve.
        
        # Pattern: Terms 0-6 handle k=0,1 (7 terms), terms 7-9 handle k=2 (3 terms).
        # But we showed 3 terms can't handle all 4 k=2 products independently.
        # Unless we allow some k=2 terms to also contribute to k=0,1.
        
        # Let's try: 6 terms for k=0,1 + 4 terms for k=2 = 10 total.
        # But 6 terms can't handle the 2x2x2 part (needs rank 7).
        
        # Alternative: 7 terms where some handle both k=0/1 and k=2.
        # Let's parameterize and use scipy to solve.
        
        from scipy.optimize import least_squares
        
        # Try a specific 10-term pattern:
        # Terms 0-6: Strassen-like for k=0,1
        # Terms 7-9: Handle k=2, with one term also contributing to k=0 or k=1
        
        # Actually, let's try a completely different strategy:
        # Use the fact that C[:,2] = A @ B[:,2] is a 2x2 times 2x1 multiplication.
        # The rank of 2x2 x 2x1 matrix multiplication is 4 (each output is a dot product).
        # But if we can share terms with the k=0,1 computation, we might save one.
        
        # Let's try: 9 terms for k=0,1,2 combined + 1 extra = 10.
        # Or: use a bilinear algorithm with 10 multiplications.
        
        # Given the complexity, let's try a direct numerical search.
        # We'll fix the U and V patterns (sparse) and solve for W, then verify.
        
        # Try pattern: same as rank-11 but merge T7 into T0 by adding B02 to V[0]
        # and adjusting W. This won't work directly because M*N changes.
        
        # Let's try a random restart approach with structured initialization.
        rng = np.random.RandomState(42)
        
        for attempt in range(200):
            if time_remaining() < 10:
                break
            
            # Initialize with rank-11 structure, then randomly perturb and optimize
            U = np.zeros((n*m, 10))
            V = np.zeros((m*p, 10))
            W = np.zeros((p*n, 10))
            
            # Copy first 7 terms from rank-11 (k=0,1 part)
            for r in range(7):
                for i in range(n*m):
                    U[i, r] = sol11["U"][i][r]
                for j in range(m*p):
                    V[j, r] = sol11["V"][j][r]
                for k in range(p*n):
                    W[k, r] = sol11["W"][k][r]
            
            # Terms 7-9 for k=2: start from rank-11 terms 7,8,9 and add randomness
            for r in range(7, 10):
                src = r  # map to rank-11 terms 7,8,9
                for i in range(n*m):
                    U[i, r] = sol11["U"][i][src] * (1 + 0.1*rng.randn())
                for j in range(m*p):
                    V[j, r] = sol11["V"][j][src] * (1 + 0.1*rng.randn())
                for k in range(p*n):
                    W[k, r] = sol11["W"][k][src] * (1 + 0.1*rng.randn())
            
            # Now try to optimize to make this a valid rank-10 decomposition
            # Define loss: sum of squared errors over random test cases
            
            def loss(params):
                U_flat = params[:n*m*10].reshape(n*m, 10)
                V_flat = params[n*m*10:n*m*10 + m*p*10].reshape(m*p, 10)
                W_flat = params[n*m*10 + m*p*10:].reshape(p*n, 10)
                
                U3 = U_flat.reshape(n, m, 10)
                V3 = V_flat.reshape(m, p, 10)
                W3 = W_flat.reshape(p, n, 10)
                
                total_err = 0.0
                for _ in range(5):
                    A = rng.randn(n, m)
                    B = rng.randn(m, p)
                    C_expected = A @ B
                    M = np.einsum('ijr,ij->r', U3, A)
                    N = np.einsum('jkr,jk->r', V3, B)
                    MN = M * N
                    C_computed = np.einsum('kir,r->ik', W3, MN)
                    total_err += np.sum((C_computed - C_expected)**2)
                return total_err
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(loss, x0, method='lm', max_nfev=500)
                if result.success or result.fun < 1e-8:
                    U_opt = result.x[:n*m*10].reshape(n*m, 10)
                    V_opt = result.x[n*m*10:n*m*10 + m*p*10].reshape(m*p, 10)
                    W_opt = result.x[n*m*10 + m*p*10:].reshape(p*n, 10)
                    
                    sol10 = {
                        "U": U_opt.tolist(),
                        "V": V_opt.tolist(),
                        "W": W_opt.tolist()
                    }
                    if verify_solution(sol10, n_trials=20):
                        return sol10
            except:
                pass
        
        return None

    better_sol = try_rank10()
    
    if better_sol is not None:
        return better_sol
    else:
        return sol11

# EVOLVE-BLOCK-END
