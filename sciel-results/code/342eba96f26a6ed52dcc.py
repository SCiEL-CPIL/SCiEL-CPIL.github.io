"""Matrix multiplication 3x4x8 tensor decomposition search."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    rank_trivial = n * m * p  # 96
    triples = [(i, k, j) for i in range(n) for k in range(m) for j in range(p)]
    T = len(triples)  # 96
    
    idx_U = np.array([i*m+k for i,k,j in triples])
    idx_V = np.array([k*p+j for i,k,j in triples])
    idx_W = np.array([j*n+i for i,k,j in triples])
    
    def _verify_fast(U, V, W, tol=1e-9):
        """Verify decomposition using vectorized operations."""
        r = U.shape[1]
        U_exp = U[idx_U, :]
        V_exp = V[idx_V, :]
        W_exp = W[idx_W, :]
        vals = np.einsum('tr,tr,tr->t', U_exp, V_exp, W_exp)
        return bool(np.all(np.abs(vals - 1.0) < tol))
    
    def _verify_exact(U, V, W):
        """Exact verification using Fraction arithmetic."""
        U_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in U]
        V_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in V]
        W_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in W]
        
        for t in range(T):
            s = Fraction(0)
            for r in range(len(U_frac[0])):
                s += U_frac[idx_U[t]][r] * V_frac[idx_V[t]][r] * W_frac[idx_W[t]][r]
            if s != Fraction(1):
                return False
        return True
    
    def _clean_output(U, V, W):
        """Convert to clean integer/half-integer representation."""
        def _clean_mat(M):
            result = []
            for row in M:
                new_row = []
                for val in row:
                    v2 = round(val * 2)
                    if v2 % 2 == 0:
                        new_row.append(int(v2 // 2))
                    else:
                        new_row.append(v2 / 2.0)
                result.append(new_row)
            return result
        return {"U": _clean_mat(U), "V": _clean_mat(V), "W": _clean_mat(W)}
    
    best_result = None
    best_rank = rank_trivial
    
    lower_bound = max(n*m, m*p, n*p)  # 32
    
    # ALS Strategy:
    # For a given rank R, initialize U, V, W randomly.
    # Iterate:
    #   1. Solve for U given V, W (linear system)
    #   2. Round U to half-integers
    #   3. Solve for V given U, W (linear system)
    #   4. Round V to half-integers
    #   5. Solve for W given U, V (linear system)
    #   6. Round W to half-integers
    # Check if exact.
    
    def _solve_for_U(V, W, R):
        # U[idx_U[t], r] * V[idx_V[t], r] * W[idx_W[t], r] = 1
        # Sum over r.
        # For a fixed U column r, it appears in terms where idx_U[t] == u_row.
        # Actually, it's easier to build the full linear system for all U entries.
        # Unknowns: U[0..nm-1, 0..R-1]
        # Equation t: sum_r U[idx_U[t], r] * (V[idx_V[t], r] * W[idx_W[t], r]) = 1
        # Let C[t, r] = V[idx_V[t], r] * W[idx_W[t], r]
        # Then sum_r U[idx_U[t], r] * C[t, r] = 1
        # This is a linear system A_U * U_vec = 1
        # A_U[t, u_idx] = C[t, r] if u_idx corresponds to (idx_U[t], r)
        
        A = np.zeros((T, n*m*R))
        b = np.ones(T)
        
        for t in range(T):
            for r in range(R):
                u_row = idx_U[t]
                col = u_row * R + r
                val = V[idx_V[t], r] * W[idx_W[t], r]
                A[t, col] = val
        
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            U_sol = sol.reshape(n*m, R)
            return U_sol
        except np.linalg.LinAlgError:
            return None

    def _solve_for_V(U, W, R):
        A = np.zeros((T, m*p*R))
        b = np.ones(T)
        
        for t in range(T):
            for r in range(R):
                v_row = idx_V[t]
                col = v_row * R + r
                val = U[idx_U[t], r] * W[idx_W[t], r]
                A[t, col] = val
        
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            V_sol = sol.reshape(m*p, R)
            return V_sol
        except np.linalg.LinAlgError:
            return None

    def _solve_for_W(U, V, R):
        A = np.zeros((T, p*n*R))
        b = np.ones(T)
        
        for t in range(T):
            for r in range(R):
                w_row = idx_W[t]
                col = w_row * R + r
                val = U[idx_U[t], r] * V[idx_V[t], r]
                A[t, col] = val
        
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            W_sol = sol.reshape(p*n, R)
            return W_sol
        except np.linalg.LinAlgError:
            return None

    def _round_to_half_int(M):
        return np.round(M * 2) / 2.0

    # Search ranks from lower_bound upwards
    for rank in range(lower_bound, min(rank_trivial + 1, 50)):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        # Number of attempts: more for smaller ranks
        if rank <= 35:
            attempts = 200
        elif rank <= 40:
            attempts = 100
        else:
            attempts = 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.default_rng(seed=rank * 100000 + attempt * 7)
            
            # Initialize with small random integers
            U = rng.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
            V = rng.choice([-1, 0, 1], size=(m*p, rank)).astype(float)
            W = rng.choice([-1, 0, 1], size=(p*n, rank)).astype(float)
            
            # ALS iterations
            converged = False
            for _iter in range(20):
                if time.time() > deadline - 5.0:
                    break
                
                # Solve U
                U_sol = _solve_for_U(V, W, rank)
                if U_sol is None: continue
                U_new = _round_to_half_int(U_sol)
                
                # Solve V
                V_sol = _solve_for_V(U_new, W, rank)
                if V_sol is None: continue
                V_new = _round_to_half_int(V_sol)
                
                # Solve W
                W_sol = _solve_for_W(U_new, V_new, rank)
                if W_sol is None: continue
                W_new = _round_to_half_int(W_sol)
                
                # Check convergence and validity
                if np.allclose(U_new, U) and np.allclose(V_new, V) and np.allclose(W_new, W):
                    if _verify_fast(U_new, V_new, W_new, 1e-8):
                        if _verify_exact(U_new, V_new, W_new):
                            best_result = _clean_output(U_new, V_new, W_new)
                            best_rank = rank
                            found = True
                            break
                    break # Converged but not valid, try next attempt
                
                U, V, W = U_new, V_new, W_new
            
            if found:
                break
        
        if found:
            break
    
    # Fallback: return schoolbook decomposition
    if best_result is None:
        U_out = [[0]*rank_trivial for _ in range(n*m)]
        V_out = [[0]*rank_trivial for _ in range(m*p)]
        W_out = [[0]*rank_trivial for _ in range(p*n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i*(m*p) + k*p + j
                    U_out[i*m+k][r] = 1
                    V_out[k*p+j][r] = 1
                    W_out[j*n+i][r] = 1
        best_result = {"U": U_out, "V": V_out, "W": W_out}
    
    return best_result
# EVOLVE-BLOCK-END
