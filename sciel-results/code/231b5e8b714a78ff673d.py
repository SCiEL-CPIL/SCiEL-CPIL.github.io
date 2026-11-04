"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
import sympy
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_exact_solve(n, m, p, rank, time_limit, start_time):
    """
    Try to find an exact rational solution using sympy.
    We have n*m*p equations: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    Unknowns: U (n*m * rank), V (m*p * rank), W (p*n * rank)
    Total unknowns: rank*(n*m + m*p + p*n)
    Total equations: n*m*p
    For rank >= 2, underdetermined, so we fix some variables to make it solvable.
    """
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    # We'll fix the first (n*m*p - rank*(n*m+m*p+p*n) + 1) variables... 
    # Actually, the system is underdetermined. Let's use a different approach:
    # Fix V and W to be simple (e.g., identity-like or random small integers),
    # then solve for U.
    # But that might not work. Let's try a structured approach.
    
    # Alternative: use the fact that we can add "free" rank-1 terms.
    # Start with a partial decomposition and add terms to satisfy remaining equations.
    
    # For now, let's try a simpler approach: 
    # Use numpy to solve a linear system after fixing some variables.
    # But the equations are bilinear/trilinear, not linear.
    
    # Let's try a different strategy: use a known construction.
    # For matrix multiplication, there are known sub-multiplicative algorithms.
    # But for small dimensions, let's just try to find any valid decomposition.
    
    # Approach: randomly initialize U, V, W with small rational numbers,
    # then use a local search to satisfy the equations.
    
    # Actually, let's try to use sympy for a small rank.
    # The number of unknowns is rank*60, equations are 72.
    # For rank=2: 120 unknowns, 72 equations. Underdetermined by 48.
    # We can fix 48 variables to simple values and solve for the rest.
    
    # Let's try rank=2 first.
    if rank < 2:
        return None
    
    # Create symbolic variables
    # To keep it manageable, let's fix most variables to 0 or 1.
    # Strategy: Use a "sparse" decomposition where each term has few non-zero entries.
    
    # Let's try a different approach: use the structure of matrix multiplication.
    # C = A*B, where A is n x m, B is m x p.
    # We can decompose this as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is already a decomposition of rank m (the inner dimension).
    # Wait, that's not right. The tensor rank is different.
    
    # Actually, the standard matrix multiplication can be done in rank m*p*n (schoolbook),
    # but there are faster algorithms. For 3x4x6, the minimal rank is unknown.
    
    # Let's just try to find any valid decomposition with rank < 72.
    # Use a random search with local optimization.
    
    return None  # Fall through to numerical approach

def _try_numerical_optimization(n, m, p, rank, time_limit, start_time):
    """
    Try to find a numerical solution using gradient-based optimization.
    Minimize sum over i,j,k of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    """
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    rng = np.random.RandomState(42)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Use a simple gradient descent with adaptive step size
    lr = 0.01
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current value
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        err = np.mean((result - target) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Compute gradients (approximate)
        # d/dU[i*m+j,c] = sum_{k} V[j*p+k,c]*W[k*n+i,c]*(result[i,j,k]-1)
        # This is expensive, so use finite differences or a simpler update
        
        # Simple stochastic gradient: pick random i,j,k and update
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        
        delta = result[i, j, k] - 1
        
        # Update U[i*m+j, :]
        for c in range(rank):
            U[i*m+j, c] -= lr * delta * V[j*p+k, c] * W[k*n+i, c]
            V[j*p+k, c] -= lr * delta * U[i*m+j, c] * W[k*n+i, c]
            W[k*n+i, c] -= lr * delta * U[i*m+j, c] * V[j*p+k, c]
        
        # Decay learning rate
        if it % 100 == 0:
            lr *= 0.99
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    
    return None

def _try_block_decomposition(n, m, p, time_limit, start_time):
    """
    Try a block-based decomposition that exploits the matrix structure.
    Split the multiplication into smaller blocks.
    """
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    # For 3x4x6, we can try to decompose as:
    # A (3x4) * B (4x6) = C (3x6)
    # Split A into [A1; A2] where A1 is 3x2, A2 is 3x2
    # Split B into [B1, B2] where B1 is 2x6, B2 is 2x6
    # Then C = A1*B1 + A2*B2
    # This gives rank = 2 * (3*2*6) = 72, same as schoolbook.
    
    # Try a different split: use Strassen-like ideas.
    # But Strassen is for 2x2. For 3x4x6, we need a custom construction.
    
    # Let's try to find a decomposition with rank < 72 by merging terms.
    # Two schoolbook terms (i,j,k) and (i',j',k') can be merged if:
    # There exists a rank-1 tensor that equals the sum of the two.
    # This requires U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] + U[i'*m+j',c']*V[j'*p+k',c']*W[k'*n+i',c']
    # to be expressible as a single rank-1 term.
    
    # This is complex. Let's just return None and use the fallback.
    
    return None

def _to_rational_list(arr, tol=1e-6):
    """Convert a numpy array to a list of lists with rational (Fraction) entries."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            if abs(val) < tol:
                row_list.append(0)
            else:
                # Try to find a simple fraction
                frac = Fraction(val).limit_denominator(100)
                row_list.append(frac)
        result.append(row_list)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # Known lower bound: rank >= max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # Try ranks from 24 to 71
    
    for target_rank in range(24, 72):
        if time.time() > deadline - 20:
            break
        
        # Try numerical optimization
        result = _try_numerical_optimization(n, m, p, target_rank, deadline - time.time(), start_time)
        if result is not None:
            U_found, V_found, W_found, err = result
            if _verify(U_found, V_found, W_found, n, m, p, tol=1e-8):
                # Convert to rational
                U_rational = _to_rational_list(U_found)
                V_rational = _to_rational_list(V_found)
                W_rational = _to_rational_list(W_found)
                
                # Verify the rational version
                if _verify(U_rational, V_rational, W_rational, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_result = {"U": U_rational, "V": V_rational, "W": W_rational}
                    break
    
    return best_result
# EVOLVE-BLOCK-END
