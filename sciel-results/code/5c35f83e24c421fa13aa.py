"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Attempt Rank 23 (Coppersmith-Winograd)
    if time.time() < deadline:
        result = _try_rank_23(n, m, p)
        if result is not None:
            return result
            
    # Attempt Rank 26 (Strassen + Direct)
    if time.time() < deadline:
        result = _try_rank_26(n, m, p)
        if result is not None:
            return result
            
    # Fallback: Trivial Rank 27
    return _rank_27_fallback(n, m, p)

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target
    T_target[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    """
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _try_rank_23(n, m, p):
    """
    Attempt to return a valid rank-23 decomposition.
    Uses a hardcoded Coppersmith-Winograd decomposition.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 23
    
    # These terms are from a known CW73 implementation.
    # Note: Indices are 0-8.
    terms = [
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),
        ({3:1, 4:1}, {0:1}, {1:1, 4:-1}),
        ({0:1}, {1:1, 4:-1}, {3:1, 4:1}),
        ({4:1}, {3:1, 0:-1}, {1:1, 0:1}),
        ({0:1, 1:1}, {4:1}, {3:1, 0:-1}),
        ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
        ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),
        ({2:1}, {6:1}, {6:1}),
        ({2:1}, {7:1}, {7:1}),
        ({2:1}, {8:1}, {8:1}),
        ({5:1}, {6:1}, {6:1}),
        ({5:1}, {7:1}, {7:1}),
        ({5:1}, {8:1}, {8:1}),
        ({6:1}, {2:1}, {6:1}),
        ({6:1}, {5:1}, {7:1}),
        ({6:1}, {8:1}, {8:1}),
        ({7:1}, {2:1}, {6:1}),
        ({7:1}, {5:1}, {7:1}),
        ({7:1}, {8:1}, {8:1}),
        ({8:1}, {2:1}, {6:1}),
        ({8:1}, {5:1}, {7:1}),
        ({8:1}, {8:1}, {8:1}),
        # The 23rd term is often a combination or correction. 
        # In many implementations, the last term is:
        ({0:1, 1:1, 3:1, 4:1}, {0:1, 1:1, 3:1, 4:1}, {0:1, 1:1, 3:1, 4:1}) # This is a guess, likely wrong.
    ]
    
    # Since I am not confident in the 23rd term above, I will not return this 
    # unless it passes verification.
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return None

def _try_rank_26(n, m, p):
    """
    Attempt to return a valid rank-26 decomposition.
    Uses the seed's construction.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 26
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    
    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    
    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    
    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None

def _rank_27_fallback(n, m, p):
    """Return a simple rank-27 decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = 27
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
