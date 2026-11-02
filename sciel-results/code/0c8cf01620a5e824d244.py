import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 145s deadline, 5s safety margin
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # 1. Strassen for first 2 columns (k=0,1)
    # Rank 7
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((12, 7))
    W_strassen = np.zeros((12, 7))
    
    for s in range(7):
        step = strassen_steps[s]
        for l in range(4):
            U_strassen[l, s] = step['u'][l]
            b_global = [0, 1, 6, 7][l]
            V_strassen[b_global, s] = step['v'][l]
            c_global = [0, 2, 1, 3][l]
            W_strassen[c_global, s] = step['w'][l]

    # 2. Rank 13 for next 4 columns (k=2,3,4,5)
    T_224 = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k_prime in range(4):
                a = 2*i + j
                b = 2*j + k_prime
                c = 4*i + k_prime
                T_224[a, b, c] = 1.0

    def objective(params):
        U = params[:52].reshape(13, 4)
        V = params[52:156].reshape(13, 8)
        W = params[156:260].reshape(13, 8)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        res = T_calc - T_224
        return res.flatten()

    def try_rank13(x0, max_nfev=3000):
        try:
            sol = opt.least_squares(objective, x0, method='lm', max_nfev=max_nfev)
            if sol.cost < 1e-10:
                U_13 = sol.x[:52].reshape(13, 4)
                V_13 = sol.x[52:156].reshape(13, 8)
                W_13 = sol.x[156:260].reshape(13, 8)
                
                # Try rounding to integers and half-integers
                for scale in [1, 2, 4]:
                    U_r = np.round(U_13 * scale) / scale
                    V_r = np.round(V_13 * scale) / scale
                    W_r = np.round(W_13 * scale) / scale
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T_224, atol=1e-8):
                        return U_r, V_r, W_r
                # Also try without rounding if already very close
                T_check = np.einsum('ar,br,cr->abc', U_13, V_13, W_13)
                if np.allclose(T_check, T_224, atol=1e-8):
                    return U_13, V_13, W_13
        except Exception:
            pass
        return None

    # Build structured initial guesses for T_{2,2,4} rank-13 decomposition
    # The tensor T_{2,2,4} has the Kronecker structure: T_{2,2,4} = T_{2,2,1} ⊗ I_4
    # where T_{2,2,1} is the 2x2 matrix multiplication tensor.
    # A rank-7 decomposition of T_{2,2,1} extended to 4 columns needs rank 7*4=28 naively,
    # but we can share across columns. The minimum rank of T_{2,2,4} is at least 7 (lower bound).
    # Let's try a structured approach: use 4 "base" Strassen-like terms (one per column pair)
    # plus 9 correction terms.
    
    def make_structured_guess_v2():
        """Better structured guess using the block structure."""
        U = np.zeros((13, 4))
        V = np.zeros((13, 8))
        W = np.zeros((13, 8))
        
        # Use 4 terms that handle one column each with a simple pattern
        # Term 0: handles column 0 with identity-like structure
        # For column k', we need: sum_a U[a,r] * V[2j+k',r] * W[4i+k',r] = delta_{a,2i+j}
        
        # Strategy: 4 terms for direct multiplication (rank 4 for each column = 16 total, too many)
        # Instead, use 7 Strassen terms replicated + 6 extra
        
        # Better: use the fact that T_{2,2,4} = sum_{k'=0}^{3} T_{2,2,1}^{(k')}
        # where each T_{2,2,1}^{(k')} has support on {a} x {2j+k'} x {4i+k'}
        
        # Use 7 Strassen steps, each contributing to all 4 columns
        for s in range(7):
            step = strassen_steps[s]
            for l in range(4):
                U[l, s] = step['u'][l]
            # For each Strassen step, the v and w values need to work for all 4 k'
            # Original: v[0] for (j=0,k=0), v[1] for (j=0,k=1), v[2] for (j=1,k=0), v[3] for (j=1,k=1)
            # For 4 columns: v[2j+k'] for j in {0,1}, k' in {0,1,2,3}
            # We need: sum_r U[a,r]*V[2j+k',r]*W[4i+k',r] = 1 if a=2i+j else 0
            # The Strassen formula gives: sum_s u_s(a)*v_s(j,k)*w_s(i,k) = delta_{a,2i+j}
            # where v_s(j,k) and w_s(i,k) depend on k. For 4 independent k' values,
            # we can set v_s(2j+k') = v_s_original(2j + (k' mod 2)) and similarly for w
            # But this won't work because the k' values are independent.
            
            # Alternative: for each k', use the same Strassen u values but different v,w
            # This means 7*4 = 28 terms, too many.
            
            # Let's try: v_s(2j+k') = v_s_orig(j, k'%2) for all k'
            # and w_s(4i+k') = w_s_orig(i, k'%2) for all k'
            # This gives correct results only if the Strassen formula is separable in k,
            # which it is: sum_s u(a)*v(j,k)*w(i,k) = delta_{a,2i+j}
            # For fixed k, this holds. So for each k', using the same formula with k=k'%2
            # should give delta_{a,2i+j} for that k'. But we need it to hold simultaneously
            # for all k', which it does since the sum is over s and the equation is
            # pointwise in (a,j,i,k').
            
            for k_prime in range(4):
                b_idx = k_prime  # j=0
                b_idx2 = 4 + k_prime  # j=1
                c_idx = k_prime  # i=0
                c_idx2 = 4 + k_prime  # i=1
                # v_orig[0] is for (j=0, k=0), v_orig[1] for (j=0, k=1)
                # v_orig[2] is for (j=1, k=0), v_orig[3] for (j=1, k=1)
                V[b_idx, s] = step['v'][0]   # j=0, use k=0 pattern
                V[b_idx2, s] = step['v'][2]  # j=1, use k=0 pattern
                W[c_idx, s] = step['w'][0]   # i=0, use k=0 pattern
                W[c_idx2, s] = step['w'][2]  # i=1, use k=0 pattern
        
        # Initialize remaining 6 terms with small random values
        np.random.seed(456)
        U[7:, :] = np.random.randn(6, 4) * 0.3
        V[7:, :] = np.random.randn(6, 8) * 0.3
        W[7:, :] = np.random.randn(6, 8) * 0.3
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    # Helper to assemble full rank-20 decomposition from Strassen + 13-term block
    def assemble_rank20(U_13_r, V_13_r, W_13_r):
        U = np.zeros((4, 20))
        V = np.zeros((12, 20))
        W = np.zeros((12, 20))
        U[:, :7] = U_strassen
        V[:, :7] = V_strassen
        W[:, :7] = W_strassen
        U[:, 7:20] = U_13_r
        for r in range(13):
            for lb in range(8):
                val = V_13_r[lb, r]
                if val != 0:
                    j = lb // 4
                    k_prime = lb % 4
                    k = 2 + k_prime
                    gb = 6*j + k
                    V[gb, 7+r] = val
            for lc in range(8):
                val = W_13_r[lc, r]
                if val != 0:
                    i = lc // 4
                    k_prime = lc % 4
                    k = 2 + k_prime
                    gc = 2*k + i
                    W[gc, 7+r] = val
        return U, V, W

    # Expected tensor for verification
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                T_exp[i*2+j, j*6+k, k*2+i] = 1.0

    def verify_full(U, V, W):
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_full, T_exp, atol=1e-6)

    # Try structured guess first
    if time.time() < DEADLINE:
        x0_struct = make_structured_guess_v2()
        result = try_rank13(x0_struct, max_nfev=5000)
        if result is not None:
            U_13_r, V_13_r, W_13_r = result
            U, V, W = assemble_rank20(U_13_r, V_13_r, W_13_r)
            if verify_full(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try multiple random restarts with diverse seeds
    for seed_val in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
                     7000, 8000, 9000, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
                     8888, 9999, 12345, 54321, 13579, 24680, 11111, 22222, 33333, 44444]:
        if time.time() >= DEADLINE:
            break
        np.random.seed(seed_val)
        x0 = np.random.randn(260) * 0.5
        result = try_rank13(x0, max_nfev=2000)
        if result is not None:
            U_13_r, V_13_r, W_13_r = result
            U, V, W = assemble_rank20(U_13_r, V_13_r, W_13_r)
            if verify_full(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 21
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U[l, r] = step['u'][l]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V[b_map[l], r] = step['v'][l]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W[c_map[l], r] = step['w'][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
