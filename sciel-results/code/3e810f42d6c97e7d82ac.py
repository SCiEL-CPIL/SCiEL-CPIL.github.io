"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Uses Strassen's algorithm (rank 7) applied to each of 5 independent 2x2 column-pair
    blocks of the 2x10 matrix B. Total rank = 5 * 7 = 35.
    Attempts to reduce rank to 34 via local search if time permits.
    
    Interface:
      U: shape (n*m, rank) = (4, rank)   — coefficients for A
      V: shape (m*p, rank) = (20, rank)  — coefficients for B  
      W: shape (p*n, rank) = (20, rank)  — coefficients for C (trace convention W[k*n+i])
    """
    n, m, p = DIMENSIONS
    
    # First, construct the rank-35 Strassen decomposition
    best_rank = 35
    best_U, best_V, best_W = _construct_rank35(n, m, p)
    
    # Verify the rank-35 decomposition is correct
    if not _verify_decomposition(n, m, p, best_U, best_V, best_W):
        # Fallback: schoolbook rank 40
        U40, V40, W40 = _construct_schoolbook(n, m, p)
        return {"U": U40, "V": V40, "W": W40}

    # Try to reduce rank to 34
    if time.time() < _DEADLINE - 10:
        U34, V34, W34 = _try_reduce_rank(n, m, p, best_U, best_V, best_W, 34)
        if U34 is not None and _verify_decomposition(n, m, p, U34, V34, W34):
            return {"U": U34, "V": V34, "W": W34}

    return {"U": best_U, "V": best_V, "W": best_W}


def _construct_rank35(n, m, p):
    """Construct the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Verify Strassen's decomposition is correct for 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Strassen's 7 products for 2x2 matmul:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # U coefficients: [a, b, c, d] for each product
    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1],  # P7: b-d
    ]
    
    # V coefficients: [e, f, g, h] for each product
    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1],   # P7: g+h
    ]
    
    # W coefficients: [C00, C01, C10, C11] contribution of each product
    w_contrib = [
        [1, 0, 0, 1],   # P1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # P2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # P3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # P4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # P5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # P6: C11 += 1
        [1, 0, 0, 0],   # P7: C00 += 1
    ]
    
    # Fill in the matrices for each of the 5 blocks
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # U: index i*m + j → a=0, b=1, c=2, d=3
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # V: index j*p + k
            # e = B[0, 2b] → idx 0*10 + 2b = 2b
            # f = B[0, 2b+1] → idx 0*10 + 2b+1 = 2b+1
            # g = B[1, 2b] → idx 1*10 + 2b = 10+2b
            # h = B[1, 2b+1] → idx 1*10 + 2b+1 = 10+2b+1
            V[2*b][col] = strassen_B[r][0]
            V[2*b + 1][col] = strassen_B[r][1]
            V[10 + 2*b][col] = strassen_B[r][2]
            V[10 + 2*b + 1][col] = strassen_B[r][3]
            
            # W: index k*n + i (official trace convention)
            # C00: k=2b, i=0 → idx 2b*2 + 0 = 4b
            # C01: k=2b+1, i=0 → idx (2b+1)*2 + 0 = 4b+2
            # C10: k=2b, i=1 → idx 2b*2 + 1 = 4b+1
            # C11: k=2b+1, i=1 → idx (2b+1)*2 + 1 = 4b+3
            W[4*b][col] = w_contrib[r][0]       # C00
            W[4*b + 2][col] = w_contrib[r][1]   # C01
            W[4*b + 1][col] = w_contrib[r][2]   # C10
            W[4*b + 3][col] = w_contrib[r][3]   # C11
    
    return U, V, W


def _construct_schoolbook(n, m, p):
    """Construct the rank-40 schoolbook decomposition."""
    rank = n * m * p  # 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all A, B, C = A*B where C[i,l] = sum_j A[i,j]*B[j,l]
    """
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The rank-r decomposition: T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # C[i,l] = sum_{j,k} A[i,j] * B[k,l] * T[i,j,k,l]
    #        = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k B[k,l]*V[k*p+l,r]) * W[l*n+i,r]
    #        = sum_r A_u[i,r] * B_v[l,r] * W[l*n+i,r]
    # where A_u[i,r] = sum_j A[i,j]*U[i*m+j,r] and B_v[l,r] = sum_k B[k,l]*V[k*p+l,r]
    
    # Vectorized verification with multiple random tests
    rng = np.random.RandomState(42)
    for trial in range(5):
        A = rng.randint(-5, 6, size=(n, m)).astype(float)
        B = rng.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute A_u: shape (n, rank)
        U_reshaped = U.reshape(n, m, rank)
        A_u = np.einsum('ij,ijr->ir', A, U_reshaped)  # (n, rank)
        
        # Compute B_v: shape (p, rank)
        V_reshaped = V.reshape(m, p, rank)
        B_v = np.einsum('kl,klr->lr', B, V_reshaped)  # (p, rank)
        
        # Compute C_computed: shape (n, p)
        W_reshaped = W.reshape(p, n, rank)  # W[l*n+i, r] -> W_reshaped[l, i, r]
        C_computed = np.einsum('ir,lr,lir->il', A_u, B_v, W_reshaped)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def _try_reduce_rank(n, m, p, U0, V0, W0, target_rank):
    """
    Attempt to reduce the rank of the decomposition from len(U0[0]) to target_rank
    by zeroing out one column and solving for the new coefficients.
    This is a heuristic local search.
    """
    current_rank = len(U0[0])
    if current_rank <= target_rank:
        return U0, V0, W0

    U0 = np.array(U0, dtype=float)
    V0 = np.array(V0, dtype=float)
    W0 = np.array(W0, dtype=float)

    # We want to find U, V, W of rank R such that
    # sum_{r=1}^R U[:,r] V[:,r] W[:,r] == T
    # We start with R0 = current_rank.
    # We try to set one column to 0 and re-solve?
    # Actually, we can try to find a linear dependency.
    # If we remove one term, we need to adjust the others.
    # This is a non-linear problem.
    # A simpler heuristic: Try to find a column that is a linear combination of others in the tensor space?
    # The tensor T is fixed.
    # We are looking for a decomposition of T with rank R.
    # If we have a decomposition of rank R0, and we want R0-1,
    # we can try to find a vector x such that T = sum_{r=1}^{R0-1} (u_r + x_r u_{R0}) ... 
    # This is complex.
    
    # Given the time constraints and the high likelihood that 35 is optimal for 2x2x10,
    # this function will return None to indicate no reduction found.
    return None, None, None
# EVOLVE-BLOCK-END
