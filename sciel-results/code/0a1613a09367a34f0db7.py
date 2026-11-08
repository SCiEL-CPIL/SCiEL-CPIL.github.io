"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 23 (Coppersmith-Winograd) first
    if time.time() < deadline:
        result = _try_rank_23(n, m, p)
        if result is not None:
            return result
    
    # Try rank 24
    if time.time() < deadline:
        result = _try_rank_24(n, m, p)
        if result is not None:
            return result
    
    # Try rank 25
    if time.time() < deadline:
        result = _try_rank_25(n, m, p)
        if result is not None:
            return result
            
    # Fallback: return a known rank-26 decomposition
    return _rank_26_seed()

def _try_rank_23(n, m, p):
    """
    Attempt to return a valid rank-23 decomposition.
    Uses a hardcoded Coppersmith-Winograd decomposition.
    Verifies the result before returning.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 23
    
    # Hardcoded terms for Rank 23 (Coppersmith-Winograd)
    # Each term is a tuple (U_dict, V_dict, W_dict)
    terms = [
        ({0:1, 1:1, 3:1, 4:1}, {0:1, 1:1, 3:1, 4:1}, {0:1, 1:1, 3:1, 4:1}),
        ({0:1, 4:-1}, {0:-1, 4:1}, {0:1, 4:1}),
        ({1:1, 4:1}, {0:1, 3:1}, {3:1, 4:1}),
        ({0:1, 1:1}, {3:1, 4:1}, {0:1, 1:1}),
        ({3:1, 4:1}, {1:1, 4:1}, {4:1}),
        ({2:1, 5:1}, {6:1, 7:1}, {0:1, 1:1}),
        ({2:1, 5:1}, {6:1, 7:1}, {2:1, 3:1}),
        ({6:1, 7:1}, {2:1, 5:1}, {0:1, 1:1}),
        ({6:1, 7:1}, {2:1, 5:1}, {2:1, 3:1}),
        ({8:1}, {8:1}, {4:1}),
        ({2:1}, {8:1}, {2:1}),
        ({5:1}, {8:1}, {3:1}),
        ({8:1}, {2:1}, {0:1}),
        ({8:1}, {5:1}, {1:1}),
        ({2:1}, {6:1}, {0:1}),
        ({2:1}, {7:1}, {1:1}),
        ({5:1}, {6:1}, {2:1}),
        ({5:1}, {7:1}, {3:1}),
        ({6:1}, {2:1}, {0:1}),
        ({6:1}, {5:1}, {1:1}),
        ({7:1}, {2:1}, {2:1}),
        ({7:1}, {5:1}, {3:1}),
        ({8:1}, {8:1}, {4:1})
    ]
    
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
            
    # Verification
    # Target Tensor T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=i*p+j
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = i * p + j
                T_target[a, b, c] = 1
                
    # Reconstructed Tensor
    T_recon = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    if np.allclose(T_recon, T_target):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return None

def _try_rank_24(n, m, p):
    """Attempt rank-24 decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = 24
    return None

def _try_rank_25(n, m, p):
    """Attempt rank-25 decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = 25
    return None

def _rank_26_seed():
    """Replicate the seed's rank-26 construction."""
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
