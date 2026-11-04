"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build list of support indices: (a, b, c) where T[a,b,c] = 1
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                support.append((a, b, c))
    
    # Build list of non-support indices (sample for efficiency)
    all_indices = set()
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                all_indices.add((a, b, c))
    
    support_set = set(support)
    nonsupport = [(a, b, c) for (a, b, c) in all_indices if (a, b, c) not in support_set]
    
    # Try rank 62 first
    for target_rank in [62, 61, 60]:
        if time.time() > deadline - 60:
            break
        
        r = target_rank
        # Initialize with schoolbook-like structure
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        W = np.zeros((pn, r), dtype=float)
        
        # Initialize with first r schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
                    else:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        
        def residuals(x):
            U_flat = x[:nm*r].reshape(nm, r)
            V_flat = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W_flat = x[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
            
            res = []
            # Support constraints: sum_r U[a,r]*V[b,r]*W[c,r] = 1
            for (a, b, c) in support:
                val = np.sum(U_flat[a,:] * V_flat[b,:] * W_flat[c,:])
                res.append(val - 1)
            
            # Nonsupport constraints: sum_r U[a,r]*V[b,r]*W[c,r] = 0
            # Use a subset for efficiency
            for (a, b, c) in nonsupport[:500]:
                val = np.sum(U_flat[a,:] * V_flat[b,:] * W_flat[c,:])
                res.append(val)
            
            return res
        
        try:
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            result = least_squares(residuals, x0, method='lm', max_nfev=1000,
                                   xtol=1e-10, ftol=1e-10)
            
            if result.success and np.max(np.abs(result.fun)) < 1e-6:
                U_sol = result.x[:nm*r].reshape(nm, r)
                V_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_sol = result.x[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
                
                # Verify with all constraints
                max_err = 0
                for (a, b, c) in support:
                    val = np.sum(U_sol[a,:] * V_sol[b,:] * W_sol[c,:])
                    max_err = max(max_err, abs(val - 1))
                for (a, b, c) in nonsupport:
                    val = np.sum(U_sol[a,:] * V_sol[b,:] * W_sol[c,:])
                    max_err = max(max_err, abs(val))
                
                if max_err < 1e-6:
                    # Round to half-integers
                    U_sol = np.round(U_sol * 2) / 2
                    V_sol = np.round(V_sol * 2) / 2
                    W_sol = np.round(W_sol * 2) / 2
                    
                    # Verify after rounding
                    max_err = 0
                    for (a, b, c) in support:
                        val = np.sum(U_sol[a,:] * V_sol[b,:] * W_sol[c,:])
                        max_err = max(max_err, abs(val - 1))
                    for (a, b, c) in nonsupport:
                        val = np.sum(U_sol[a,:] * V_sol[b,:] * W_sol[c,:])
                        max_err = max(max_err, abs(val))
                    
                    if max_err < 1e-6:
                        def to_list(mat):
                            result = []
                            for row in mat:
                                r = []
                                for val in row:
                                    v = round(val * 2) / 2
                                    r.append(v)
                                result.append(r)
                            return result
                        
                        return {
                            "U": to_list(U_sol),
                            "V": to_list(V_sol),
                            "W": to_list(W_sol)
                        }
        except Exception:
            pass
    
    # Fall back to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                v = round(val * 2) / 2
                r.append(v)
            result.append(r)
        return result
    
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
