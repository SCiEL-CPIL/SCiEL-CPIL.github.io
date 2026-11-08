"""Tensor decomposition search for 2x4 x 4x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _tensor_coeff(i, j, k, n, m, p):
    """Return the coefficient of the (i,k) entry of the product matrix in the multiplication tensor.
    The multiplication tensor T is defined such that C[i,k] = sum_{j} A[i,j] * B[j,k].
    In the tensor formulation with factors U, V, W:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    where a indexes the A-matrix (n*m), b indexes the B-matrix (m*p), c indexes the C-matrix (p*n).
    The official trace convention for W is W[k*n+i, r].
    The tensor has T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k, and 0 otherwise.
    """
    a = i * m + j
    b = j * p + k
    c = k * n + i
    if 0 <= a < n * m and 0 <= b < m * p and 0 <= c < p * n:
        return 1
    return 0

def _build_tensor(n, m, p):
    """Build the full multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if the decomposition exactly reproduces the tensor."""
    # T_reconstructed = sum_r U[:,r] * V[:,r] * W[:,r] (outer product sum)
    # Using einsum: T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T_recon - T))
    return max_err < tol, max_err

def _snap_to_half_integer(x, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    half = 2.0 * x
    rounded = np.round(half)
    if np.all(np.abs(half - rounded) < tol):
        return rounded / 2.0
    return None

def _verify_exact(U, V, W, n, m, p):
    """Verify the decomposition exactly using fractions."""
    T_frac = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_frac[(a, b, c)] = Fraction(1, 1)
    
    # Check that all entries in U, V, W are half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, (int, np.integer)):
                    if val != int(val):
                        return False
                elif isinstance(val, (float, np.floating)):
                    if abs(val * 2 - round(val * 2)) > 1e-10:
                        return False
    
    rank = U.shape[1]
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                expected = T_frac.get((a, b, c), Fraction(0, 1))
                actual = Fraction(0, 1)
                for r in range(rank):
                    u_val = U[a, r] if isinstance(U[a, r], (int, np.integer)) else float(U[a, r])
                    v_val = V[b, r] if isinstance(V[b, r], (int, np.integer)) else float(V[b, r])
                    w_val = W[c, r] if isinstance(W[c, r], (int, np.integer)) else float(W[c, r])
                    # Convert to fraction
                    u_f = Fraction(u_val).limit_denominator(1000)
                    v_f = Fraction(v_val).limit_denominator(1000)
                    w_f = Fraction(w_val).limit_denominator(1000)
                    actual += u_f * v_f * w_f
                if actual != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_r(r, n, m, p, T, max_iterations=5000, seed=0):
    """Try to find a rank-r decomposition using local search."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random half-integers in {-1, -0.5, 0, 0.5, 1}
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf
    
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    
    # Try multiple random initializations
    for init in range(20):
        U = rng.choice(vals, size=(n * m, r))
        V = rng.choice(vals, size=(m * p, r))
        W = rng.choice(vals, size=(p * n, r))
        
        # Local search
        for it in range(max_iterations):
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
            
            # Pick a random element to perturb
            if r > 0:
                mat_choice = rng.randint(3)
                if mat_choice == 0:
                    row = rng.randint(n * m)
                    col = rng.randint(r)
                    old = U[row, col]
                    U[row, col] = rng.choice(vals)
                elif mat_choice == 1:
                    row = rng.randint(m * p)
                    col = rng.randint(r)
                    old = V[row, col]
                    V[row, col] = rng.choice(vals)
                else:
                    row = rng.randint(p * n)
                    col = rng.randint(r)
                    old = W[row, col]
                    W[row, col] = rng.choice(vals)
                
                # Compute new error
                T_recon_new = np.einsum('ar,br,cr->abc', U, V, W)
                err_new = np.sum((T_recon_new - T) ** 2)
                
                if err_new >= err:
                    # Revert
                    if mat_choice == 0:
                        U[row, col] = old
                    elif mat_choice == 1:
                        V[row, col] = old
                    else:
                        W[row, col] = old
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-10:
        # Try to snap to half-integers
        U_snap = _snap_to_half_integer(best_U)
        V_snap = _snap_to_half_integer(best_V)
        W_snap = _snap_to_half_integer(best_W)
        if U_snap is not None and V_snap is not None and W_snap is not None:
            return {"U": U_snap, "V": V_snap, "W": W_snap}, best_err
        else:
            return {"U": best_U, "V": best_V, "W": best_W}, best_err
    
    return None, best_err

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try ranks from 40 down to some lower bound
    # Known lower bound: For 2x4x5, the Strassen rank is at most 40
    # Try a few ranks below 40
    for r in range(39, 30, -1):
        if time.time() > deadline:
            break
        
        result, err = _try_rank_r(r, n, m, p, T, max_iterations=2000, seed=42)
        if result is not None:
            # Verify
            U_np = np.array(result["U"], dtype=np.float64)
            V_np = np.array(result["V"], dtype=np.float64)
            W_np = np.array(result["W"], dtype=np.float64)
            ok, max_err = _check_decomposition(U_np, V_np, W_np, T)
            if ok:
                return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
