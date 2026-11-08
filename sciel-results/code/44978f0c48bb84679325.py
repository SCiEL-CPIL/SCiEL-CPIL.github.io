import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _try_rank_numpy(r, deadline, n, m, p, max_attempts=5000):
    """
    Try to find a rank-r decomposition by fixing U and V randomly and solving for W.
    Uses numpy for speed and Fraction for verification.
    """
    np.random.seed(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
            
        # Generate random U and V with small integer entries
        # Using -1, 0, 1 to keep numbers small
        U_np = np.random.randint(-1, 2, size=(n * m, r))
        V_np = np.random.randint(-1, 2, size=(m * p, r))
        
        # Ensure rows are not all zero
        if np.any(np.all(U_np == 0, axis=1)) or np.any(np.all(V_np == 0, axis=1)):
            continue
            
        W_np = np.zeros((p * n, r))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Construct A_{k,i} of shape (m, r)
                # A_{k,i, j, c} = U[i, j, c] * V[j, k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    # U index: i*m + j
                    # V index: j*p + k
                    A[j, :] = U_np[i * m + j, :] * V_np[j * p + k, :]
                
                # Solve A * w = [1, 1, 1]
                # Using lstsq to handle underdetermined systems
                try:
                    w_sol, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is close to a half-integer
                w_rounded = np.round(w_sol * 2) / 2.0
                
                # Verify if w_rounded satisfies the equation approximately
                # If not, this random U, V might not lead to a nice W
                # We can check the residual of the rounded solution
                residual = np.linalg.norm(A @ w_rounded - np.ones(m))
                if residual > 1e-6:
                    # The rounded solution is not exact. 
                    # We could try to find a better solution in the null space, 
                    # but for now, we reject this candidate.
                    valid = False
                    break
                
                W_np[k * n + i, :] = w_rounded
        
        if not valid:
            continue
            
        # Convert to lists of Fractions for verification
        U_list = [[int(x) for x in row] for row in U_np]
        V_list = [[int(x) for x in row] for row in V_np]
        W_list = [[_round_to_half(x) for x in row] for row in W_np]
        
        # Exact verification
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from lower bound up to schoolbook-1
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    for r in range(15, best_rank):
        if time.time() > deadline:
            break
            
        result = _try_rank_numpy(r, deadline, n, m, p)
        if result is not None:
            best_result = result
            best_rank = r
            break
            
    return best_result
# EVOLVE-BLOCK-END
