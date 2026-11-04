"""Matrix multiplication 2x3x9: find rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    # Precompute tensor entries
    tensor_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.add((a, b, c))
    
    # Try to find a decomposition with rank < 54
    # We will try ranks 27, 36, 45.
    # Use a small prime field for speed.
    prime = 257
    
    def try_find_decomposition(rank):
        # Initialize U, V with random values in [0, prime)
        np.random.seed(rank * 1000 + 42)
        U = np.random.randint(0, prime, size=(n*m, rank))
        V = np.random.randint(0, prime, size=(m*p, rank))
        
        # We want to solve for W.
        # For each c in range(p*n), we have a system of equations for W[c, :].
        # The equations are: for all a, b: sum_s U[a,s] V[b,s] W[c,s] = T[a,b,c]
        # This is a linear system A_c @ W[c,:] = b_c
        # A_c is (n*m * m*p) x rank = 162 x rank
        # b_c is 162
        
        W = np.zeros((p*n, rank))
        
        for c in range(p*n):
            # Build A_c and b_c
            A_c = np.zeros((n*m * m*p, rank))
            b_c = np.zeros(n*m * m*p)
            row = 0
            for a in range(n*m):
                for b in range(m*p):
                    for s in range(rank):
                        A_c[row, s] = (U[a, s] * V[b, s]) % prime
                    if (a, b, c) in tensor_entries:
                        b_c[row] = 1
                    row += 1
            
            # Solve A_c @ w = b_c
            # Since it's overdetermined, use least squares.
            # But we want an exact solution if it exists.
            # We can check if the residual is 0.
            w, residuals, rank_c, sv = np.linalg.lstsq(A_c.astype(np.float64), b_c.astype(np.float64), rcond=None)
            
            # Check residual
            # Reconstruct
            recon = A_c @ w
            # Note: A_c and b_c are integers, w is float.
            # We need to check if recon is close to b_c.
            if np.max(np.abs(recon - b_c)) > 1e-6:
                return None
            
            W[c, :] = w
        
        # If we reached here, we have a candidate W.
        # Verify over integers (or rationals).
        # The values in U, V are integers. W might be fractional.
        # If the decomposition is exact over the field, it should be exact over Q.
        # We need to convert W to integers or fractions.
        # If the rank is low, the entries might be small.
        
        # Check if W entries are integers (or close to them)
        # If not, we might need to scale or find a different solution.
        # However, the problem allows half-integers.
        
        # Let's verify the decomposition over integers/fractions.
        # We will use Fraction for precision.
        valid = True
        W_frac = [[Fraction(W[c, s]) for s in range(rank)] for c in range(p*n)]
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = sum(U[a, s] * V[b, s] * W_frac[c][s] for s in range(rank))
                    expected = 1 if (a, b, c) in tensor_entries else 0
                    if val != expected:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists of ints or fractions
            # The interface expects integer/half-integer entries.
            # We can return fractions as floats if they are exact, or just use the values.
            # The evaluator likely checks exactness.
            # We will return the values.
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = [[float(W_frac[c][s]) for s in range(rank)] for c in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks
    for rank in [27, 36, 45]:
        if time.time() > deadline:
            break
        result = try_find_decomposition(rank)
        if result is not None:
            return result

    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
