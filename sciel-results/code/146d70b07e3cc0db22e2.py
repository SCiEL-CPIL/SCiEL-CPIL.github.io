"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # List of target (a,b,c) indices where T=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    def verify(U, V, W, tol=1e-10):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def to_list(M):
        """Convert numpy array to list of lists, trying to round to integers/half-integers"""
        L = []
        for row in M:
            L.append([])
            for val in row:
                v = float(val)
                if abs(v - round(v)) < 1e-9:
                    L[-1].append(int(round(v)))
                elif abs(v - (round(2*v)/2.0)) < 1e-9:
                    L[-1].append(round(2*v)/2.0)
                else:
                    L[-1].append(v)
        return L
    
    # Schoolbook decomposition (rank = n*m*p = 84)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # Strategy: Try to find lower-rank decompositions by grouping terms
    # For each possible rank r < 84, try to find U, V, W such that the tensor matches
    
    # Approach 1: Try structured decompositions based on index patterns
    # For small n, m, p, we can try specific patterns
    
    # Approach 2: Use the fact that the tensor is sparse and try to find
    # rank-1 terms that cover multiple target positions
    
    # Key insight: A rank-1 term u⊗v⊗w contributes u[a]*v[b]*w[c] to position (a,b,c)
    # For the multiplication tensor, we need specific (a,b,c) to have value 1
    # If we can find u, v, w such that u[a]*v[b]*w[c] = 1 for multiple (a,b,c) in target_indices
    # and 0 for others, we can reduce the rank
    
    # Let's try a systematic search for rank-1 terms with structured support
    
    # For each possible support pattern, solve for the values
    
    # Actually, let's try a different approach: use the bilinear complexity results
    # For 3x4x7, the minimum rank is known to be less than 84 in some cases
    
    # Let's try to find a decomposition by solving a system of equations
    
    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all (a,b,c) in target_indices: sum_c U[a,c]*V[b,c]*W[c,c] = 1
    # and for all other (a,b,c): sum_c U[a,c]*V[b,c]*W[c,c] = 0
    
    # This is a nonlinear system. Let's try to solve it using optimization
    
    from scipy.optimize import minimize
    
    def objective(params, r):
        """Objective function for rank-r decomposition"""
        Ur = params[:nm*r].reshape(nm, r)
        Vr = params[nm*r:nm*r+mp*r].reshape(mp, r)
        Wr = params[nm*r+mp*r:].reshape(pn, r)
        S = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
        return np.sum((S - T)**2)
    
    # Try reducing rank from high to low
    # Start from just below schoolbook rank and try to find valid decompositions
    
    # For efficiency, try ranks in a strategic order
    # First try ranks that are multiples or have special structure
    
    ranks_to_try = []
    # Try ranks from 83 down to a reasonable lower bound
    # Lower bound: max(nm, mp, pn) = max(12, 28, 21) = 28
    # But realistically, for matrix multiplication, the rank is at least n*m + m*p + p*n - something
    
    # Let's try a range of ranks
    for target_rank in range(best_rank - 1, max(40, best_rank - 20), -1):
        if time.time() > DEADLINE:
            break
        
        # Multiple random restarts
        for attempt in range(5):
            if time.time() > DEADLINE:
                break
            
            # Initialize with random values
            Ur = np.random.randn(nm, target_rank) * 0.1
            Vr = np.random.randn(mp, target_rank) * 0.1
            Wr = np.random.randn(pn, target_rank) * 0.1
            
            x0 = np.concatenate([Ur.flatten(), Vr.flatten(), Wr.flatten()])
            
            try:
                res = minimize(objective, x0, args=(target_rank,), method='BFGS',
                              options={'maxiter': 200, 'gtol': 1e-10})
                if res.fun < 1e-10:
                    Ur_ = res.x[:nm*target_rank].reshape(nm, target_rank)
                    Vr_ = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    Wr_ = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    if verify(Ur_, Vr_, Wr_):
                        best_rank = target_rank
                        best_U, best_V, best_W = Ur_, Vr_, Wr_
                        break
            except:
                pass
    
    # Also try direct low-rank optimization with random initializations for smaller ranks
    for target_rank in range(max(40, best_rank - 30), best_rank):
        if time.time() > DEADLINE:
            break
        if target_rank >= best_rank:
            continue
        
        for attempt in range(3):
            if time.time() > DEADLINE:
                break
            
            Ur = np.random.randn(nm, target_rank) * 0.1
            Vr = np.random.randn(mp, target_rank) * 0.1
            Wr = np.random.randn(pn, target_rank) * 0.1
            
            x0 = np.concatenate([Ur.flatten(), Vr.flatten(), Wr.flatten()])
            
            try:
                res = minimize(objective, x0, args=(target_rank,), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < 1e-10:
                    Ur_ = res.x[:nm*target_rank].reshape(nm, target_rank)
                    Vr_ = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    Wr_ = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    if verify(Ur_, Vr_, Wr_):
                        best_rank = target_rank
                        best_U, best_V, best_W = Ur_, Vr_, Wr_
                        break
            except:
                pass
    
    # Final polish: try to clean up the solution if we found a lower rank
    if best_rank < n * m * p:
        # Try to round to integers/half-integers and verify
        U_clean = np.round(best_U * 2) / 2
        V_clean = np.round(best_V * 2) / 2
        W_clean = np.round(best_W * 2) / 2
        if verify(U_clean, V_clean, W_clean):
            best_U, best_V, best_W = U_clean, V_clean, W_clean
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
