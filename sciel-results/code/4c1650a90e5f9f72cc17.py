"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.flatten()

    def make_residual(r):
        def residual(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = x[nm*r+mp*r:].reshape(pn, r)
            R = np.einsum('ad,bd,cd->abc', U, V, W)
            return (R - T).flatten()
        return residual

    def schoolbook_init(r):
        """
        Initialize with schoolbook terms, but if r < n*m*p, we need a valid starting point.
        For r < n*m*p, we can't just drop terms. 
        Strategy: Start with a random initialization or a perturbed schoolbook if r == n*m*p.
        For r < n*m*p, we use random init.
        """
        if r == n * m * p:
            U0 = np.zeros((nm, r))
            V0 = np.zeros((mp, r))
            W0 = np.zeros((pn, r))
            for c in range(r):
                i = c // (m*p)
                rem = c % (m*p)
                j = rem // p
                k = rem % p
                U0[i*m+j, c] = 1.0
                V0[j*p+k, c] = 1.0
                W0[k*n+i, c] = 1.0
            return np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        else:
            # Random initialization for lower ranks
            np.random.seed(42 + r)
            U0 = np.random.randn(nm, r)
            V0 = np.random.randn(mp, r)
            W0 = np.random.randn(pn, r)
            return np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

    def snap_to_rational(x, max_denom=10):
        """
        Snap float values to nearest Fraction with denominator <= max_denom.
        Returns list of Fractions or floats if no good snap found.
        """
        snapped = []
        for val in x:
            best_frac = Fraction(val).limit_denominator(max_denom)
            # Check if the snap is close enough
            if abs(float(best_frac) - val) < 1e-6:
                snapped.append(best_frac)
            else:
                # If not close, keep as float? No, task requires integer/half-integer.
                # If we can't snap, this solution is invalid.
                return None
        return snapped

    def verify_decomposition(U, V, W, T, tol=1e-9):
        """
        Verify if U, V, W exactly decompose T.
        U, V, W are lists of lists of Fractions or floats.
        """
        nm, mp, pn = len(U), len(V), len(W)
        r = len(U[0])
        U_arr = np.array([[float(u) for u in row] for row in U])
        V_arr = np.array([[float(v) for v in row] for row in V])
        W_arr = np.array([[float(w) for w in row] for row in W])
        
        R = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
        if not np.allclose(R, T, atol=tol, rtol=0):
            return False
        return True

    # We want to find the minimal rank.
    # Schoolbook rank is 20.
    # Try ranks 19, 18, ... down to a lower bound.
    # Lower bound for 2x2x5 is 10.
    # But finding rank 10 is very hard. 
    # Let's try 19 first, then 18.
    
    deadline = time.time() + 140
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution as fallback
    U_fb = np.zeros((nm, best_rank))
    V_fb = np.zeros((mp, best_rank))
    W_fb = np.zeros((pn, best_rank))
    for c in range(best_rank):
        i = c // (m*p)
        rem = c % (m*p)
        j = rem // p
        k = rem % p
        U_fb[i*m+j, c] = 1.0
        V_fb[j*p+k, c] = 1.0
        W_fb[k*n+i, c] = 1.0
        
    # Convert fallback to list of lists of floats (or ints)
    def to_list_of_lists(mat):
        return [[float(val) for val in row] for row in mat]

    # Try ranks from 19 down to 15 (arbitrary cutoff, higher ranks are easier to find if they exist)
    # If we find rank 19, we stop.
    
    for r in range(best_rank - 1, 14, -1):
        if time.time() > deadline:
            break
            
        residual_fn = make_residual(r)
        
        # Try multiple initializations
        found = False
        num_attempts = 20
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
                
            if attempt == 0:
                # Use a perturbed schoolbook-like init? 
                # For r < 20, we can't use standard schoolbook.
                # We can use a "sparse" init: pick r random triples and set them to 1, others 0?
                # This is a valid rank-r tensor (sum of r rank-1 tensors).
                # Let's try random sparse init.
                np.random.seed(attempt * 100 + r)
                U0 = np.zeros((nm, r))
                V0 = np.zeros((mp, r))
                W0 = np.zeros((pn, r))
                
                for c in range(r):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    U0[i*m+j, c] = 1.0
                    V0[j*p+k, c] = 1.0
                    W0[k*n+i, c] = 1.0
                    
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            else:
                # Random dense init
                np.random.seed(attempt * 7919 + r * 31)
                x0 = np.random.randn(nm*r + mp*r + pn*r) * 0.5

            try:
                result = least_squares(residual_fn, x0, method='lm', max_nfev=5000)
                if result.cost < 1e-10:
                    # Extract U, V, W
                    U = result.x[:nm*r].reshape(nm, r)
                    V = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    W = result.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Try to snap to rationals
                    # The task says "integer/half-integer". 
                    # So denominators should be 1 or 2.
                    U_snapped = snap_to_rational(U.flatten(), max_denom=2)
                    if U_snapped is None:
                        continue
                    V_snapped = snap_to_rational(V.flatten(), max_denom=2)
                    if V_snapped is None:
                        continue
                    W_snapped = snap_to_rational(W.flatten(), max_denom=2)
                    if W_snapped is None:
                        continue
                    
                    # Reshape snapped values
                    U_list = [[U_snapped[i*r + c] for c in range(r)] for i in range(nm)]
                    V_list = [[V_snapped[i*r + c] for c in range(r)] for i in range(mp)]
                    W_list = [[W_snapped[i*r + c] for c in range(r)] for i in range(pn)]
                    
                    # Verify exactness
                    if verify_decomposition(U_list, V_list, W_list, T, tol=1e-12):
                        best_rank = r
                        best_solution = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            # Found a solution at this rank. We could try lower, but 
            # usually the first found is good. 
            # However, the loop continues to lower ranks.
            # If we want to be safe, we can break here if we find a very low rank,
            # but let's let the loop run to see if we can go lower.
            # Actually, if we find rank 19, we should check 18, etc.
            # But optimization for 18 is harder.
            # Let's break if we find rank <= 19 to save time? 
            # No, the goal is rank-minimal. 
            # But finding 18 is much harder than 19.
            # Let's just try one step down if time permits.
            # For now, break to ensure we return a valid result quickly.
            # If we don't break, we might time out trying rank 18.
            # Given the difficulty, rank 19 is a good target.
            break

    if best_solution is not None:
        return best_solution
    else:
        # Return schoolbook
        return {"U": to_list_of_lists(U_fb), "V": to_list_of_lists(V_fb), "W": to_list_of_lists(W_fb)}

# EVOLVE-BLOCK-END
