import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition as fallback
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

    # Deadline guard: 140s to leave 10s for overhead/serialization
    deadline = time.time() + 140

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            # Skip if any factor is all zeros (though unlikely in valid decomp)
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)

    def round_and_verify(U, V, W):
        """
        Try to round U, V, W to integers or half-integers and verify.
        Returns (U_int, V_int, W_int) if successful, else None.
        """
        # Try integers first
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half):
            return U_half, V_half, W_half
            
        # Try mixed: maybe some are integers, some half-integers?
        # This is combinatorially expensive, so we stick to global rounding strategies
        # or rely on the optimizer to push values close to these grids.
        return None

    def try_rank(R, max_attempts=5):
        """
        Attempt to find a decomposition of rank R.
        Uses random linear combinations of schoolbook basis vectors as initialization.
        """
        if R >= best_rank:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize U, V, W with random linear combinations of schoolbook columns
            # This ensures the initial solution is "close" to the true manifold
            U_init = np.zeros((nm, R))
            V_init = np.zeros((mp, R))
            W_init = np.zeros((pn, R))
            
            # Pick R random columns from schoolbook and mix them
            # Actually, a better init is to take random linear combinations of the basis
            # Let's just take random subsets of the schoolbook basis and add noise
            # Or better: Generate random vectors and project?
            # Simplest robust init: Random Gaussian, then optimize.
            # But to improve success rate, we can start with a subset of schoolbook vectors
            # and add small random perturbations to break symmetry.
            
            # Strategy: Take R random columns from schoolbook, then optimize.
            # If R < 84, we are dropping some components.
            # To allow for "recombination", we should initialize with random linear combos.
            
            # Let's use a random projection of the schoolbook basis.
            # Create a random matrix M of size (84, R)
            # U_init = U_s @ M
            # V_init = V_s @ M  (This is not quite right, the basis is coupled)
            
            # Correct initialization for coupled basis:
            # The schoolbook decomposition is sum_{i,j,k} e_i e_j^T e_k^T.
            # A general rank R decomposition is sum_{r=1}^R u_r v_r w_r.
            # If we pick R random linear combinations of the 84 basis vectors:
            # u_r = sum_{i,j,k} c_{r,i,j,k} e_{i,j}
            # v_r = sum_{i,j,k} c_{r,i,j,k} e_{j,k}
            # w_r = sum_{i,j,k} c_{r,i,j,k} e_{k,i}
            # Then sum_r u_r v_r w_r = sum_{i,j,k} (sum_r c_{r,i,j,k}^2) e_{i,j} e_{j,k} e_{k,i}
            # This only recovers the tensor if sum_r c_{r,i,j,k}^2 = 1 for all i,j,k.
            # This is a valid way to initialize!
            
            C = np.random.randn(R, n, m, p)
            # Normalize so that sum_r C[r,i,j,k]^2 = 1
            # Actually, we want sum_r (C[r,i,j,k])^2 = 1 to match T exactly in the limit.
            # Let's just normalize the columns of C flattened?
            # No, we want the "weight" of each basis vector to sum to 1.
            # Let's just use random C and let the optimizer fix the scaling.
            
            U_init = np.zeros((nm, R))
            V_init = np.zeros((mp, R))
            W_init = np.zeros((pn, R))
            
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = C[r, i, j, k]
                            U_init[i*m+j, r] += val
                            V_init[j*p+k, r] += val
                            W_init[k*n+i, r] += val
                            
            # Add small noise to avoid symmetry breaking issues
            U_init += np.random.randn(nm, R) * 0.1
            V_init += np.random.randn(mp, R) * 0.1
            W_init += np.random.randn(pn, R) * 0.1

            # Optimize
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    Ur = params[:nm*R].reshape(nm, R)
                    Vr = params[nm*R:nm*R+mp*R].reshape(mp, R)
                    Wr = params[nm*R+mp*R:].reshape(pn, R)
                    
                    # Compute S = sum_r Ur[:,r] outer Vr[:,r] outer Wr[:,r]
                    S = np.zeros((nm, mp, pn))
                    for r in range(R):
                        S += np.einsum('a,b,c->abc', Ur[:, r], Vr[:, r], Wr[:, r])
                    return np.sum((S - T)**2)
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                # Use L-BFGS-B for larger problems, Nelder-Mead for smaller
                # nm=12, mp=28, pn=21. Total vars = 12R + 28R + 21R = 61R.
                # If R=80, vars=4880. L-BFGS is better.
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    Ur = res.x[:nm*R].reshape(nm, R)
                    Vr = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    Wr = res.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try to round
                    result = round_and_verify(Ur, Vr, Wr)
                    if result:
                        return result
                        
            except Exception:
                pass
                
        return None

    # Try to reduce rank
    # Start from best_rank - 1 and go down
    # Limit the search to a reasonable range to fit in time
    # Schoolbook is 84. Try down to 70.
    for R in range(best_rank - 1, 69, -1):
        if time.time() > deadline:
            break
            
        # Try a few attempts for each rank
        # More attempts for lower ranks? No, lower ranks are harder.
        # But we break as soon as we find one.
        res = try_rank(R, max_attempts=3)
        
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = R
            best_U, best_V, best_W = Ur, Vr, Wr
            # We found a lower rank, can we go lower?
            # Yes, continue the loop.
            
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
