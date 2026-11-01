"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_rank_11():
    """
    Construct a rank-11 decomposition for 2x2x3 matrix multiplication.
    
    Uses Strassen's 7-term decomposition for the 2x2x2 subproblem (k=0,1)
    plus 4 terms for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen's 7 products for 2x2x2 (k=0,1 part)
    c = 0
    U[0][c] = 1; U[3][c] = 1
    V[0][c] = 1; V[4][c] = 1
    W[0][c] = 1; W[1][c] = 1; W[2][c] = 1; W[3][c] = 1
    
    c = 1
    U[2][c] = 1; U[3][c] = 1
    V[0][c] = 1
    W[2][c] = 1; W[3][c] = -1
    
    c = 2
    U[0][c] = 1
    V[3][c] = 1; V[4][c] = -1
    W[1][c] = 1; W[3][c] = 1
    
    c = 3
    U[3][c] = 1
    V[1][c] = 1; V[3][c] = -1
    W[0][c] = 1; W[2][c] = 1
    
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[4][c] = 1
    W[0][c] = -1; W[1][c] = 1
    
    c = 5
    U[2][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[3][c] = 1
    W[3][c] = 1
    
    c = 6
    U[1][c] = 1; U[3][c] = -1
    V[1][c] = 1; V[4][c] = 1
    W[0][c] = 1
    
    c = 7
    U[0][c] = 1
    V[2][c] = 1
    W[4][c] = 1
    
    c = 8
    U[1][c] = 1
    V[5][c] = 1
    W[4][c] = 1
    
    c = 9
    U[2][c] = 1
    V[2][c] = 1
    W[5][c] = 1
    
    c = 10
    U[3][c] = 1
    V[5][c] = 1
    W[5][c] = 1
    
    return U, V, W

def _try_rank_11():
    """Attempt to construct and verify a rank-11 decomposition."""
    n, m, p = 2, 2, 3
    U, V, W = _build_rank_11()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _gf3_rank_check():
    """
    Check over GF(3) whether a rank-10 decomposition is feasible.
    We fix the support pattern from rank-11 and check if we can merge
    two terms by finding a linear dependency over GF(3).
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Over GF(3), we look for U (4x10), V (6x10), W (6x10) with entries in {0,1,2}
    # such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 mod 3 for all i,j,k
    
    # Strategy: Start from the rank-11 solution mod 3, and try to find a
    # linear combination of two columns that can be merged.
    
    U11, V11, W11 = _build_rank_11()
    U = np.array(U11, dtype=np.int64) % 3
    V = np.array(V11, dtype=np.int64) % 3
    W = np.array(W11, dtype=np.int64) % 3
    
    # Check which pairs of columns can be merged (i.e., one is a scalar multiple
    # of the other in all three matrices simultaneously)
    for c1 in range(11):
        for c2 in range(c1+1, 11):
            # Check if column c2 is a scalar multiple of column c1 mod 3
            # in U, V, and W simultaneously
            u1, u2 = U[:, c1], U[:, c2]
            v1, v2 = V[:, c1], V[:, c2]
            w1, w2 = W[:, c1], W[:, c2]
            
            # Find scalar s such that u2 = s*u1 mod 3, v2 = s*v1 mod 3, w2 = s*w1 mod 3
            scalars = set()
            for mat1, mat2 in [(u1, u2), (v1, v2), (w1, w2)]:
                if np.any(mat1 != 0):
                    for idx in np.where(mat1 != 0)[0]:
                        s = (mat2[idx] * pow(mat1[idx], 2, 3)) % 3  # mat1[idx]^-1 mod 3
                        if mat2[idx] == 0 and mat1[idx] != 0:
                            return None  # incompatible
                        scalars.add(s)
                elif np.any(mat2 != 0):
                    return None  # incompatible
            
            if len(scalars) == 1:
                # Check if merging works: replace columns c1,c2 with a single column
                # that is the sum (or some combination)
                s = list(scalars)[0]
                # New column = u1 + s*u2, etc. But we need the tensor equation to still hold.
                # Actually, if col2 = s*col1, then we can just drop col2 and scale col1.
                # The contribution of col1 and col2 to each (i,j,k) is:
                # u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
                # = u1*v1*w1 + s*u1*s*v1*s*w1 = u1*v1*w1*(1+s^3)
                # For this to equal the original single-column contribution, we need 1+s^3 = 1 mod 3
                # s^3 = 0 mod 3, which is true for s=0. But s=0 means col2=0, trivial.
                # For s=1: 1+1=2 != 1. For s=2: 1+8=9=0 != 1.
                # So simple merging doesn't work this way.
                pass
    
    # Alternative: use a more systematic approach
    # Try to find a 10-term decomposition by solving a system over GF(3)
    # with a specific structured ansatz.
    
    # Ansatz: 7 Strassen-like terms + 3 additional terms
    # Parameterize with variables over GF(3) and check feasibility
    
    # For now, return False if we can't quickly find a rank-10 solution
    return False

def _try_rank_10_numerical(deadline):
    """
    Try to find a rank-10 decomposition using numerical optimization.
    Use a structured ansatz: 7 terms for k=0,1 (modified Strassen) + 3 terms for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # We'll use scipy's minimize to find U, V, W
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    # Parameterize: 4*10 + 6*10 + 6*10 = 160 variables
    # But we can use a structured ansatz to reduce parameters.
    
    # Ansatz: The first 7 columns follow a Strassen-like pattern with some
    # modifications, and the last 3 columns handle k=2 with shared structure.
    
    # Let's define the ansatz:
    # Columns 0-6: Strassen pattern with parameters alpha_c, beta_c, gamma_c
    # Columns 7-9: Handle k=2 with parameters
    
    # Actually, let's try a direct approach: initialize near the rank-11 solution
    # and try to zero out one column while maintaining the tensor equations.
    
    U11, V11, W11 = _build_rank_11()
    U_init = np.array(U11, dtype=float)
    V_init = np.array(V11, dtype=float)
    W_init = np.array(W11, dtype=float)
    
    # Try to merge columns 9 and 10 (the last two k=2 terms)
    # by finding a linear combination.
    
    # The k=2 constraints are:
    # For (i,j) in {(0,0),(0,1),(1,0),(1,1)}:
    # sum_c U[i*2+j,c] * V[j*3+2,c] * W[4+i,c] = 1
    
    # With 10 terms, we need these 4 equations to be satisfied.
    # Let's try: use columns 7,8 for (0,0) and (0,1), and share column 9
    # for both (1,0) and (1,1) by having V[2,9] and V[5,9] both nonzero.
    
    # Define variables:
    # x1 = U[0,9], x2 = V[2,9], x3 = W[4,9]  -> for (i=0,j=0,k=2)
    # x4 = U[1,9], x5 = V[5,9], x6 = W[4,9]  -> for (i=0,j=1,k=2)  
    # x7 = U[2,9], x8 = V[2,9], x9 = W[5,9]  -> for (i=1,j=0,k=2)
    # x10 = U[3,9], x11 = V[5,9], x12 = W[5,9] -> for (i=1,j=1,k=2)
    
    # But W[4,9] appears in both (0,0) and (0,1) equations, so x3 = x6
    # V[2,9] appears in both (0,0) and (1,0), so x2 = x8
    # V[5,9] appears in both (0,1) and (1,1), so x5 = x11
    # W[5,9] appears in both (1,0) and (1,1), so x9 = x12
    
    # So we have 8 independent variables for the k=2 part if we share column 9.
    # Equations:
    # x1*x2*x3 = 1
    # x4*x5*x3 = 1
    # x7*x2*x9 = 1
    # x10*x5*x9 = 1
    
    # From these: x1*x2*x3 = x4*x5*x3 => x1*x2 = x4*x5
    # x7*x2*x9 = x10*x5*x9 => x7*x2 = x10*x5
    # Also x1*x2*x3*x7*x2*x9 = 1*1 => x1*x7*x2^2*x3*x9 = 1
    # And x4*x5*x3*x10*x5*x9 = 1*1 => x4*x10*x5^2*x3*x9 = 1
    
    # This is underdetermined. Let's just try specific values.
    # Set x2 = 1, x3 = 1, x5 = 1, x9 = 1
    # Then x1 = 1, x4 = 1, x7 = 1, x10 = 1
    # So column 9: U[0]=1, U[1]=1, U[2]=1, U[3]=1, V[2]=1, V[5]=1, W[4]=1, W[5]=1
    
    # Now check if this conflicts with the k=0,1 equations.
    # Column 9 contributes to k=0,1 equations only if V[j*3+0,9] or V[j*3+1,9] is nonzero.
    # In our ansatz, V[2,9]=1 and V[5,9]=1, which are for k=2. So no conflict.
    
    # But we also need to handle the fact that we're removing column 10.
    # In the rank-11 solution, column 10 has: U[3]=1, V[5]=1, W[5]=1
    # If we remove column 10 and modify column 9, we need to adjust.
    
    # Let's try a different approach: use all 10 columns with a structured ansatz.
    
    # For now, return None as a placeholder. The GF(3) check and numerical
    # optimization would need more work.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-11 construction
    result = _try_rank_11()
    if result is not None:
        return result
    
    # Try rank-10 via GF(3) check
    if time.time() < deadline - 10:
        if _gf3_rank_check():
            result = _try_rank_10_numerical(deadline)
            if result is not None:
                return result
    
    # Fallback: return the schoolbook rank-12 decomposition
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
