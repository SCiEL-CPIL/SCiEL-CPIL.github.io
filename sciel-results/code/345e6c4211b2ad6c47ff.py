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
    # This represents the matrix multiplication tensor under the trace convention W[k*n+i, r]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W, tol=1e-10):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        # Reconstruct tensor: S[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    def to_list(M):
        """Convert numpy array to list of lists, trying to round to integers/half-integers"""
        L = []
        for row in M:
            L.append([])
            for val in row:
                # Try to represent as integer or half-integer
                v = float(val)
                # Check if close to integer
                if abs(v - round(v)) < 1e-9:
                    L[-1].append(int(round(v)))
                # Check if close to half-integer
                elif abs(v - (round(2*v)/2.0)) < 1e-9:
                    L[-1].append(round(2*v)/2.0)
                else:
                    # Just use the float value
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

    # Try to find lower-rank decompositions using optimization
    # Strategy: Start from schoolbook and try to reduce rank
    # For each target rank, try to find U, V, W such that the tensor matches

    from scipy.optimize import minimize

    def objective(params, r):
        """Objective function for rank-r decomposition"""
        Ur_ = params[:nm*r].reshape(nm, r)
        Vr_ = params[nm*r:nm*r+mp*r].reshape(mp, r)
        Wr_ = params[nm*r+mp*r:].reshape(pn, r)
        S = np.einsum('ar,br,cr->abc', Ur_, Vr_, Wr_)
        return np.sum((S - T)**2)

    # Try reducing rank from high to low
    # Start from just below schoolbook rank
    for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
        if time.time() > DEADLINE:
            break
        
        # Multiple random restarts
        for attempt in range(3):
            if time.time() > DEADLINE:
                break
            
            # Initialize by taking a subset of schoolbook columns and adding noise
            if target_rank <= best_rank:
                # Pick random columns from schoolbook
                cols = np.random.choice(best_rank, target_rank, replace=False)
                Ur = U_s[:, cols].copy()
                Vr = V_s[:, cols].copy()
                Wr = W_s[:, cols].copy()
                # Add small noise
                Ur += np.random.randn(nm, target_rank) * 0.01
                Vr += np.random.randn(mp, target_rank) * 0.01
                Wr += np.random.randn(pn, target_rank) * 0.01
            else:
                Ur = np.random.randn(nm, target_rank) * 0.1
                Vr = np.random.randn(mp, target_rank) * 0.1
                Wr = np.random.randn(pn, target_rank) * 0.1
            
            x0 = np.concatenate([Ur.flatten(), Vr.flatten(), Wr.flatten()])
            
            try:
                res = minimize(objective, x0, args=(target_rank,), method='BFGS',
                              options={'maxiter': 100, 'gtol': 1e-8})
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
    for target_rank in range(max(20, best_rank - 30), best_rank):
        if time.time() > DEADLINE:
            break
        if target_rank >= best_rank:
            continue
        
        for attempt in range(2):
            if time.time() > DEADLINE:
                break
            
            Ur = np.random.randn(nm, target_rank) * 0.1
            Vr = np.random.randn(mp, target_rank) * 0.1
            Wr = np.random.randn(pn, target_rank) * 0.1
            
            x0 = np.concatenate([Ur.flatten(), Vr.flatten(), Wr.flatten()])
            
            try:
                res = minimize(objective, x0, args=(target_rank,), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
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
        U_clean = np.round(best_U * 2) / 2  # Round to half-integers
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
