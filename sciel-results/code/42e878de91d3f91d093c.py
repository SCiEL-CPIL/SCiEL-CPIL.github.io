"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # 1. Construct the target tensor T for 2x3x4
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # 2. Strategy: Solve the 3x3x3 problem first to get a "border rank" style initialization
    # The 3x3x3 tensor has rank 19.
    # We will find a rank-19 decomposition of 3x3x3.
    
    def solve_333():
        n3, m3, p3 = 3, 3, 3
        nm3, mp3, pn3 = n3*m3, m3*p3, p3*n3
        T333 = np.zeros((nm3, mp3, pn3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    u = i * m3 + j
                    v = j * p3 + k
                    w = k * n3 + i
                    T333[u, v, w] = 1.0
        
        rank333 = 19
        # Initialize with random + naive structure
        np.random.seed(42)
        U3 = np.random.randn(nm3, rank333) * 0.1
        V3 = np.random.randn(mp3, rank333) * 0.1
        W3 = np.random.randn(pn3, rank333) * 0.1
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    r = (i*m3+j)*p3 + j*p3+k # naive index
                    # Just add a bit of structure
                    U3[i*m3+j, r % rank333] += 1.0
                    V3[j*p3+k, r % rank333] += 1.0
                    W3[k*n3+i, r % rank333] += 1.0
        
        x0 = np.concatenate([U3.flatten(), V3.flatten(), W3.flatten()])
        
        def res333(params):
            Uf = params[:nm3*rank333].reshape(nm3, rank333)
            Vf = params[nm3*rank333: nm3*rank333+mp3*rank333].reshape(mp3, rank333)
            Wf = params[nm3*rank333+mp3*rank333:].reshape(pn3, rank333)
            D = np.einsum('ur,vr,wr->uvw', Uf, Vf, Wf)
            return (D - T333).flatten()
        
        try:
            sol = least_squares(res333, x0, method='lm', max_nfev=200)
            if sol.cost < 1e-6:
                U3_sol = sol.x[:nm3*rank333].reshape(nm3, rank333)
                V3_sol = sol.x[nm3*rank333: nm3*rank333+mp3*rank333].reshape(mp3, rank333)
                W3_sol = sol.x[nm3*rank333+mp3*rank333:].reshape(pn3, rank333)
                return U3_sol, V3_sol, W3_sol
        except Exception:
            pass
        return None

    # 3. Use the 3x3x3 solution to initialize the 2x3x4 problem
    # The 2x3x4 tensor is a sub-tensor of a 3x3x4 or similar? 
    # Actually, 2x3x4 is NOT a direct sub-tensor of 3x3x3 in a simple index mapping.
    # However, we can use the "structure" of the 3x3x3 solution (the relative magnitudes and patterns)
    # to guide the initialization of the 2x3x4 problem.
    
    # A better projection: 
    # Map the 2x3x4 indices to 3x3x3 indices by padding.
    # i in 0..1, j in 0..2, k in 0..3.
    # In 3x3x3, i in 0..2, j in 0..2, k in 0..2.
    # This doesn't map directly.
    
    # Alternative: Just use the 3x3x3 solution to generate a "smart" random seed for the 2x3x4 problem.
    # Or, simply use the 3x3x3 solution to verify that the optimizer works, then run the 2x3x4 optimizer with multiple restarts.
    
    # Let's try to extract a rank-18 initialization from the rank-19 3x3x3 solution.
    # This is not straightforward.
    
    # Instead, let's use the "Border Rank" idea directly on 2x3x4.
    # We know the rank is 18.
    # We will try to find a decomposition of rank 18.
    
    def residual(params, rank):
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        D = np.einsum('ur,vr,wr->uvw', U_flat, V_flat, W_flat)
        res = D - T
        return res.flatten()

    def check_decomposition(U, V, W, rank, tol=1e-8):
        D = np.einsum('ur,vr,wr->uvw', U, V, W)
        max_err = np.max(np.abs(D - T))
        return max_err < tol, max_err

    def to_list(arr):
        lst = []
        for row in arr:
            row_lst = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_lst.append(rounded)
                else:
                    row_lst.append(float(val))
            lst.append(row_lst)
        return lst

    def naive_decomposition(rank=None):
        if rank is None:
            rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * (m * p) + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}

    # Try to solve 3x3x3 first to "warm up" the optimizer and potentially get a good structure
    # But since the mapping is not direct, we will just use it to set the time budget wisely.
    # If 3x3x3 is solved quickly, we have more time for 2x3x4.
    
    # Actually, a distinct strategy is to use **SVD-based initialization**.
    # We can view the tensor T as a matrix M of size (nm) x (mp*pn).
    # The rank of the tensor is the minimum number of rank-1 terms.
    # This is not the same as the matrix rank.
    
    # Let's stick to the **Multiple Restart with Structured Init** but with a **Projection** step.
    # We will initialize the rank-18 factors by taking the naive rank-24 factors and 
    # performing a **random projection** to rank 18, then optimizing.
    
    rank_target = 18
    max_attempts = 10
    attempts_per_rank = max(1, int((deadline - 10 - time.time()) / max_attempts))
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(100 + attempt)
        
        # Naive factors
        U_naive = np.zeros((nm, 24))
        V_naive = np.zeros((mp, 24))
        W_naive = np.zeros((pn, 24))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * (m * p) + j * p + k
                    U_naive[i * m + j, c] = 1
                    V_naive[j * p + k, c] = 1
                    W_naive[k * n + i, c] = 1
        
        # Project to rank 18 using a random orthogonal projection
        # This creates a "compressed" version of the naive decomposition
        P = np.random.randn(24, rank_target)
        U_init = U_naive @ P
        V_init = V_naive @ P
        W_init = W_naive @ P
        
        # Add noise
        U_init += np.random.randn(nm, rank_target) * 0.1
        V_init += np.random.randn(mp, rank_target) * 0.1
        W_init += np.random.randn(pn, rank_target) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use BFGS for better convergence on smaller problems? 
            # Least squares is more robust.
            sol = least_squares(residual, x0, args=(rank_target,), method='lm', max_nfev=300)
            
            if sol.cost < 1e-6:
                U_sol = sol.x[:nm * rank_target].reshape(nm, rank_target)
                V_sol = sol.x[nm * rank_target: nm * rank_target + mp * rank_target].reshape(mp, rank_target)
                W_sol = sol.x[nm * rank_target + mp * rank_target:].reshape(pn, rank_target)
                
                valid, err = check_decomposition(U_sol, V_sol, W_sol, rank_target, tol=1e-6)
                if valid:
                    # Polish
                    x_polish = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
                    sol2 = least_squares(residual, x_polish, args=(rank_target,), method='lm', max_nfev=200)
                    if sol2.cost < 1e-10:
                        U_pol = sol2.x[:nm * rank_target].reshape(nm, rank_target)
                        V_pol = sol2.x[nm * rank_target: nm * rank_target + mp * rank_target].reshape(mp, rank_target)
                        W_pol = sol2.x[nm * rank_target + mp * rank_target:].reshape(pn, rank_target)
                        valid2, _ = check_decomposition(U_pol, V_pol, W_pol, rank_target, tol=1e-10)
                        if valid2:
                            return {"U": to_list(U_pol), "V": to_list(V_pol), "W": to_list(W_pol)}
                    
                    return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
        except Exception:
            continue

    return naive_decomposition()
# EVOLVE-BLOCK-END
