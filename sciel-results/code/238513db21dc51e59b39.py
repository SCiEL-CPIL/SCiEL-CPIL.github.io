"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check all compatible triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    # Check all incompatible triples
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_c == i_a:
                    continue  # compatible, skip
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s) > 1e-9:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to nearest half-integer."""
    return 2 * round(x / 2)

def _try_rank(R, n, m, p, max_iters=500, tol=1e-6):
    """Try to find a rank-R decomposition. Returns (U, V, W) or None."""
    nm, mp, pn = n * m, m * p, p * n
    
    # Compatible triples
    compatible = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                compatible.append((a, b, c))
    
    # Incompatible triples (all others)
    all_triples = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    incompatible = [t for t in all_triples if t not in set(compatible)]
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(nm, R) * 0.5
    V = np.random.randn(mp, R) * 0.5
    W = np.random.randn(pn, R) * 0.5
    
    # Alternating least squares
    for it in range(max_iters):
        # Update U: for each compatible (a,b,c), need U[a,:] . (V[b,:] * W[c,:]) = 1
        # This is a linear system in U[a,:]
        for a in range(nm):
            # Collect constraints involving U[a,:]
            A = []
            y = []
            for (ca, cb, cc) in compatible:
                if ca == a:
                    row = V[cb] * W[cc]
                    A.append(row)
                    y.append(1.0)
            # Also need incompatible triples to be zero
            for (ca, cb, cc) in incompatible:
                if ca == a:
                    row = V[cb] * W[cc]
                    A.append(row)
                    y.append(0.0)
            if A:
                A = np.array(A)
                y = np.array(y)
                # Solve least squares
                if A.shape[0] >= R:
                    U[a], residuals, rank_, singular_values = np.linalg.lstsq(A, y, rcond=None)
        
        # Update V
        for b in range(mp):
            A = []
            y = []
            for (ca, cb, cc) in compatible:
                if cb == b:
                    row = U[ca] * W[cc]
                    A.append(row)
                    y.append(1.0)
            for (ca, cb, cc) in incompatible:
                if cb == b:
                    row = U[ca] * W[cc]
                    A.append(row)
                    y.append(0.0)
            if A:
                A = np.array(A)
                y = np.array(y)
                if A.shape[0] >= R:
                    V[b], residuals, rank_, singular_values = np.linalg.lstsq(A, y, rcond=None)
        
        # Update W
        for c in range(pn):
            A = []
            y = []
            for (ca, cb, cc) in compatible:
                if cc == c:
                    row = U[ca] * V[cb]
                    A.append(row)
                    y.append(1.0)
            for (ca, cb, cc) in incompatible:
                if cc == c:
                    row = U[ca] * V[cb]
                    A.append(row)
                    y.append(0.0)
            if A:
                A = np.array(A)
                y = np.array(y)
                if A.shape[0] >= R:
                    W[c], residuals, rank_, singular_values = np.linalg.lstsq(A, y, rcond=None)
        
        # Check convergence
        max_err = 0
        for (a, b, c) in compatible:
            val = np.dot(U[a], V[b] * W[c])
            max_err = max(max_err, abs(val - 1))
        for (a, b, c) in incompatible[:100]:  # Check subset for speed
            val = np.dot(U[a], V[b] * W[c])
            max_err = max(max_err, abs(val))
        
        if max_err < tol:
            break
    
    # Round to half-integers
    U_rounded = _round_to_half_integer(U)
    V_rounded = _round_to_half_integer(V)
    W_rounded = _round_to_half_integer(W)
    
    # Convert to lists
    U_list = U_rounded.tolist()
    V_list = V_rounded.tolist()
    W_list = W_rounded.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try without rounding (in case exact integers work)
    U_int = np.round(U).astype(int)
    V_int = np.round(V).astype(int)
    W_int = np.round(W).astype(int)
    if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    deadline = _START_TIME + _TIME_LIMIT
    
    # Try ranks from best_rank - 1 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Also, the rank must be at least the slice rank, which is at least max(n, m, p) = 5
    # But realistically, for 2x4*4x5, the minimum is likely around 20-30
    
    for R in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 30:  # Leave 30s for final verification
            break
        
        result = _try_rank(R, n, m, p, max_iters=200)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best
# EVOLVE-BLOCK-END
