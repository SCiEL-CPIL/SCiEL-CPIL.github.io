import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        T_calc = np.zeros((n, m, p))
        for s in range(rank):
            u_col = np.array(U)[:, s].reshape(n, m)
            v_col = np.array(V)[:, s].reshape(m, p)
            w_col = np.array(W)[:, s].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        T_target = np.ones((n, m, p))
        max_err = np.max(np.abs(T_calc - T_target))
        return max_err < 1e-9

    def search_rank(R, n_sub, m_sub, p_sub, num_restarts=50, maxiter=2000):
        """
        Search for a rank-R decomposition of the n_sub x m_sub x p_sub multiplication tensor.
        Returns (U, V, W) as lists of lists if found, else None.
        """
        if time.time() > deadline:
            return None

        dim_U = n_sub * m_sub
        dim_V = m_sub * p_sub
        dim_W = p_sub * n_sub
        
        total_vars = dim_U * R + dim_V * R + dim_W * R

        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((n_sub, m_sub, p_sub))
            for s in range(R):
                u_col = U[:, s].reshape(n_sub, m_sub)
                v_col = V[:, s].reshape(m_sub, p_sub)
                w_col = W[:, s].reshape(p_sub, n_sub)
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n_sub, m_sub, p_sub))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        np.random.seed(42)
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
            params = np.random.uniform(-1, 1, size=total_vars)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': maxiter, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n_sub, m_sub, p_sub):
                return U_snapped, V_snapped, W_snapped
        return None

    def kronecker_21():
        """
        Kronecker product: Strassen (rank 7 for 2x2x2) ⊗ rank-3 decomposition for 1x1x6.
        
        The 12 columns are split into 3 blocks of 4. For each block, we use the Strassen
        structure on the 2x2 input, and a rank-3 decomposition handles the 1x1x4 output
        within each block.
        
        Actually, let me think more carefully. The tensor for 2x2x12 multiplication is:
        T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        
        We can write this as a Kronecker product if we decompose the 12-dimensional
        output as 3 blocks of 4, and use a rank-3 decomposition for 1x1x4.
        
        For 1x1x4: C[k] = B[k], so the tensor is T[j,k] = delta_{j,k}... wait, that's
        just the identity. Let me reconsider.
        
        Actually, for 1x1xp multiplication: C[0] = B[0] (scalar times scalar). The tensor
        is T[j,k] = 1 if j==k else 0. This has rank p (each output dimension needs its
        own term).
        
        Hmm, that doesn't give us rank 3 for 1x1x4. Let me reconsider the approach.
        
        Alternative: Think of 2x2x12 as having structure where the 12 output columns
        can be grouped. If we use Strassen for the 2x2 part (rank 7), and for each of
        the 7 Strassen terms, we need to handle the 12 output columns.
        
        For a single Strassen term: it computes a linear combination of A entries times
        a linear combination of B entries, contributing to specific C entries.
        
        Let me try a different factorization. Consider that C is 2x12. We can write:
        C = A @ B where A is 2x2 and B is 2x12.
        
        Using Strassen: 7 multiplications M_s = (linear combo of A) * (linear combo of B)
        Then C[i,k] = sum_s coeff_s[i,k] * M_s
        
        The tensor T[i,j,k] for the full multiplication can be decomposed as:
        T = sum_s U_s ⊗ V_s ⊗ W_s
        
        where U_s is 2x2, V_s is 2x12, W_s is 12x2.
        
        For Strassen, each s gives a 2x2 U_s and a 2x2 V_s (for the B part per block).
        For 12 columns, V_s becomes 2x12 by repeating the 2x2 pattern 6 times.
        Similarly W_s becomes 12x2.
        
        This gives rank 7 * 6 = 42 (the naive Strassen).
        
        To get rank 21, we need to find a better decomposition.
        
        Let me try: rank-7 Strassen ⊗ rank-3 for some sub-problem.
        
        If we group the 12 columns into 3 groups of 4, and for each group of 4,
        we use a rank-3 decomposition... but 1x1x4 has rank 4, not 3.
        
        Let me try a direct search for rank 21 using optimization.
        """
        if time.time() > deadline:
            return None
            
        R = 21
        dim_U = n * m  # 4
        dim_V = m * p  # 24
        dim_W = p * n  # 24
        
        total_vars = dim_U * R + dim_V * R + dim_W * R
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((n, m, p))
            for s in range(R):
                u_col = U[:, s].reshape(n, m)
                v_col = V[:, s].reshape(m, p)
                w_col = W[:, s].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n, m, p))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        np.random.seed(42)
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            params = np.random.uniform(-1, 1, size=total_vars)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-6:
                break

        if best_error < 1e-5 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
        return None

    # Strategy 1: Try Kronecker product / direct rank-21 search
    result = kronecker_21()
    if result:
        return result

    # Fallback: Rank 42 Strassen decomposition
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * m + j_loc][r] = val
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * p + (2 * b + k_loc)][r] = val
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * n + i_loc][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    return get_strassen_42()
