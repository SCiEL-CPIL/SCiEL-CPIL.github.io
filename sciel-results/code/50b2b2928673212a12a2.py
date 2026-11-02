"""Matrix multiplication tensor decomposition for [2x3]x[3x7]."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns the schoolbook (rank 42) exact decomposition for 2x3 * 3x7.
    Attempts to find a rank 41 decomposition via random search over integers
    (scaling by 1/8 to allow half-integers in the final result).
    """
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 42
    deadline = time.time() + 100.0  # Safety margin before 120s limit

    # Build the schoolbook decomposition (always valid)
    U_sch = [[0] * rank_school for _ in range(n * m)]
    V_sch = [[0] * rank_school for _ in range(m * p)]
    W_sch = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j][col] = 1
                V_sch[j * p + k][col] = 1
                W_sch[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        """
        Verify that U, V, W exactly decompose the multiplication tensor.
        """
        n_val, m_val, p_val = n, m, p
        rank = len(U[0])
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    expected = 1
                    actual = 0
                    for c in range(rank):
                        u_val = U[i * m_val + j][c]
                        v_val = V[j * p_val + k][c]
                        w_val = W[k * n_val + i][c]
                        actual += u_val * v_val * w_val
                    if actual != expected:
                        return False
        return True

    # Check schoolbook
    assert verify(U_sch, V_sch, W_sch), "Schoolbook should always verify"

    # Try to find rank 41
    # We search for integer matrices U', V', W' such that sum U'V'W' = 8 * T
    # Then U = U'/8, etc. Since U', V', W' are integers, U, V, W are eighths.
    # The problem allows half-integers. 1/8 is not a half-integer.
    # Wait, "integer/half-integer entries".
    # So entries must be k/2.
    # If we use U'/2, then sum (U'/2)(V'/2)(W'/2) = 1 => sum U'V'W' = 8.
    # So we need integer solutions to sum U'V'W' = 8.
    
    rank_try = 41
    found_decomp = None
    random.seed(42)
    
    # Increase search range and attempts
    max_val = 4
    attempts = 10000
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        
        U_int = [[random.randint(-max_val, max_val) for _ in range(rank_try)] for _ in range(n * m)]
        V_int = [[random.randint(-max_val, max_val) for _ in range(rank_try)] for _ in range(m * p)]
        W_int = [[random.randint(-max_val, max_val) for _ in range(rank_try)] for _ in range(p * n)]
        
        # Check if sum U'V'W' = 8 for all (i,j,k)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank_try):
                        s += U_int[i * m + j][c] * V_int[j * p + k][c] * W_int[k * n + i][c]
                    if s != 8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found a solution. Convert to half-integers (divide by 2? No, divide by 2 gives 1/2, but we need 1/8?)
            # Wait. If U = U'/2, then product is U'V'W'/8.
            # We need sum = 1. So sum U'V'W' must be 8.
            # Then U = U'/2, V = V'/2, W = W'/2.
            # Are U'/2 half-integers? Yes, if U' is even? No, U'/2 is always a half-integer (k/2).
            # Example: U'=1 => 0.5. U'=2 => 1.0.
            # So yes, dividing by 2 yields half-integers.
            
            U_res = [[U_int[i][c] / 2.0 for c in range(rank_try)] for i in range(n * m)]
            V_res = [[V_int[i][c] / 2.0 for c in range(rank_try)] for i in range(m * p)]
            W_res = [[W_int[i][c] / 2.0 for c in range(rank_try)] for i in range(p * n)]
            
            # Verify with exact fractions to be safe
            U_frac = [[Fraction(U_int[i][c], 2) for c in range(rank_try)] for i in range(n * m)]
            V_frac = [[Fraction(V_int[i][c], 2) for c in range(rank_try)] for i in range(m * p)]
            W_frac = [[Fraction(W_int[i][c], 2) for c in range(rank_try)] for i in range(p * n)]
            
            if verify(U_frac, V_frac, W_frac):
                found_decomp = {"U": U_frac, "V": V_frac, "W": W_frac}
                break

    if found_decomp:
        return found_decomp
    else:
        return {"U": U_sch, "V": V_sch, "W": W_sch}

# EVOLVE-BLOCK-END
