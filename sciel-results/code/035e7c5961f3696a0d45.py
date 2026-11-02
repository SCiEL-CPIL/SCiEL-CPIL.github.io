import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_solve_for_W(U, V, n, m, p, R):
    """
    Given U (n*m x R) and V (m*p x R), try to find W (p*n x R) with half-integer entries
    such that for all i,k: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    """
    W_sol = np.zeros((p * n, R))
    
    for i in range(n):
        for k in range(p):
            # Build A matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Solve A @ w = ones(m)
            try:
                w_ls, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
            except:
                return None
            
            # Check if LS solution is already half-integer
            w_rounded = np.round(w_ls * 2) / 2
            residual = np.max(np.abs(A @ w_rounded - np.ones(m)))
            
            if residual < 1e-9:
                W_sol[k * n + i, :] = w_rounded
            else:
                # Try to find a half-integer solution in the affine space
                if rank_A < m:
                    if rank_A == 0:
                        return None
                    U_svd, S, Vt = np.linalg.svd(A, full_matrices=True)
                    null_space = Vt[rank_A:, :].T  # (R x (R-rank_A))
                    
                    found = False
                    for attempt in range(200):
                        t = np.random.randint(-4, 5, size=(R - rank_A,)) * 0.5
                        w_candidate = w_ls + null_space @ t
                        w_candidate_rounded = np.round(w_candidate * 2) / 2
                        res = np.max(np.abs(A @ w_candidate_rounded - np.ones(m)))
                        if res < 1e-9:
                            W_sol[k * n + i, :] = w_candidate_rounded
                            found = True
                            break
                    
                    if not found:
                        return None
                else:
                    w_exact = np.round(w_ls * 2) / 2
                    res = np.max(np.abs(A @ w_exact - np.ones(m)))
                    if res < 1e-9:
                        W_sol[k * n + i, :] = w_exact
                    else:
                        return None
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :] * V[j * p + k, :], W_sol[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return None
    
    return W_sol

def _try_rational_decomposition(n, m, p, R, max_denom=4, time_limit=5.0):
    """
    Try to find a rational decomposition of rank R where all entries are rational
    with denominator dividing max_denom. If found, clear denominators to get integer/half-integer.
    """
    start = time.time()
    
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Strategy: use structured patterns based on Kronecker products or simple patterns
    # For now, try a more systematic approach with small rational entries
    
    # Entry pool: rationals with denominator up to max_denom
    entries = []
    for d in range(1, max_denom + 1):
        for num in range(-d, d + 1):
            if num == 0 and d > 1:
                continue
            val = Fraction(num, d)
            if val not in entries:
                entries.append(val)
    
    entries_np = np.array([float(e) for e in entries])
    
    # Try random rational entries
    np.random.seed(12345)
    
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        U_frac = np.zeros((n * m, R), dtype=object)
        V_frac = np.zeros((m * p, R), dtype=object)
        
        for i in range(n * m):
            for r in range(R):
                U_frac[i, r] = entries[np.random.randint(len(entries))]
        
        for j in range(m * p):
            for r in range(R):
                V_frac[j, r] = entries[np.random.randint(len(entries))]
        
        # Try to solve for W
        W_frac = np.zeros((p * n, R), dtype=object)
        success = True
        
        for i in range(n):
            for k in range(p):
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, R), dtype=object)
                for j in range(m):
                    for r in range(R):
                        A[j, r] = U_frac[i * m + j, r] * V_frac[j * p + k, r]
                
                # Solve A @ w = ones(m) using exact rational arithmetic
                # Convert to float for solving, then round to nearest entry
                A_float = np.array([[float(x) for x in row] for row in A])
                b = np.ones(m)
                
                try:
                    w_float, _, _, _ = np.linalg.lstsq(A_float, b, rcond=None)
                except:
                    success = False
                    break
                
                # Round to nearest entry in our pool
                w_exact = np.zeros(R)
                for r in range(R):
                    # Find closest entry
                    best_e = entries[0]
                    best_dist = abs(float(w_float[r]) - float(best_e))
                    for e in entries[1:]:
                        dist = abs(float(w_float[r]) - float(e))
                        if dist < best_dist:
                            best_dist = dist
                            best_e = e
                    w_exact[r] = float(best_e)
                
                # Check if this works
                if np.max(np.abs(A_float @ w_exact - b)) > 1e-9:
                    success = False
                    break
                
                W_frac[k * n + i, :] = [Fraction(int(round(w_exact[r] * max_denom)), max_denom) for r in range(R)]
        
        if not success:
            continue
        
        # Verify the full decomposition
        verify_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0, 1)
                    for r in range(R):
                        val += U_frac[i * m + j, r] * V_frac[j * p + k, r] * W_frac[k * n + i, r]
                    if val != 1:
                        verify_ok = False
                        break
                if not verify_ok:
                    break
            if not verify_ok:
                break
        
        if verify_ok:
            # Clear denominators to get half-integer entries
            # Find LCM of all denominators
            from math import gcd
            def lcm(a, b):
                return abs(a * b) // gcd(a, b)
            
            lcm_denom = 1
            for i in range(n * m):
                for r in range(R):
                    d = U_frac[i, r].denominator
                    lcm_denom = lcm(lcm_denom, d)
            for j in range(m * p):
                for r in range(R):
                    d = V_frac[j, p + k, r].denominator
                    lcm_denom = lcm(lcm_denom, d)
            for k in range(p * n):
                for r in range(R):
                    d = W_frac[k, r].denominator
                    lcm_denom = lcm(lcm_denom, d)
            
            # Scale to make entries half-integers (multiply by lcm_denom / 2 if needed)
            # Actually, we just need entries to be multiples of 0.5
            # So we need lcm_denom to divide 2, i.e., lcm_denom in {1, 2}
            # If lcm_denom > 2, we need to scale
            scale = lcm_denom
            U_out = [[float(U_frac[i, r]) * scale / 2 for r in range(R)] for i in range(n * m)]
            V_out = [[float(V_frac[j, r]) * scale / 2 for r in range(R)] for j in range(m * p)]
            W_out = [[float(W_frac[k, r]) * scale / 2 for r in range(R)] for k in range(p * n)]
            
            # Verify after scaling
            U_np = np.array(U_out)
            V_np = np.array(V_out)
            W_np = np.array(W_out)
            
            verify_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :])
                        if abs(val - 1.0) > 1e-9:
                            verify_ok = False
                            break
                    if not verify_ok:
                        break
                if not verify_ok:
                    break
            
            if verify_ok:
                return U_out, V_out, W_out
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    np.random.seed(42)
    
    # Entry sets to try
    entry_sets = [
        np.array([-1, -0.5, 0, 0.5, 1]),
        np.array([-2, -1, -0.5, 0, 0.5, 1, 2]),
        np.array([0, 1]),
        np.array([-1, 0, 1]),
    ]
    
    # Try ranks from lower_bound up to best_rank - 1
    ranks_to_try = list(range(lower_bound, best_rank))
    ranks_to_try.sort(key=lambda R: (abs(R - 60), R))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        if R >= best_rank:
            continue
        
        # First, try rational decomposition approach
        if time.time() < deadline - 30:
            result = _try_rational_decomposition(n, m, p, R, max_denom=2, time_limit=3.0)
            if result is not None:
                U_out, V_out, W_out = result
                if _verify(U_out, V_out, W_out, n, m, p):
                    best_rank = R
                    best_solution = {"U": U_out, "V": V_out, "W": W_out}
                    continue
        
        # Number of attempts
        max_attempts = 50 if R < 70 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Choose entry set
            es = entry_sets[attempt % len(entry_sets)]
            
            # Generate random U and V
            U_rand = np.random.choice(es, size=(n * m, R))
            V_rand = np.random.choice(es, size=(m * p, R))
            
            # Try to solve for W
            W_sol = _try_solve_for_W(U_rand, V_rand, n, m, p, R)
            
            if W_sol is not None:
                # Verify
                if _verify(U_rand.tolist(), V_rand.tolist(), W_sol.tolist(), n, m, p):
                    best_rank = R
                    best_solution = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_sol.tolist()
                    }
                    break  # Move to next rank
    
    return best_solution
