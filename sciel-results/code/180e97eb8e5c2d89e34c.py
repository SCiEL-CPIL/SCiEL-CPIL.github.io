"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 23 (Coppersmith-Winograd) first
    result = _rank_23_cw(n, m, p)
    if result is not None:
        return result
    
    # Fallback: return a known rank-26 decomposition
    return _rank_26_seed()

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = i * p + j
                T_target[a, b, c] = 1
    
    # Reconstruct
    T_recon = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _rank_23_cw(n, m, p):
    """
    Return a valid rank-23 decomposition using Coppersmith-Winograd construction.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 23
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # The 23 terms for 3x3 matrix multiplication (Coppersmith-Winograd 1973).
    # Each term: (U_entries, V_entries, W_entries)
    # U index = i*3+k, V index = k*3+j, W index = i*3+j
    
    terms = [
        # 1: (A00+A11)(B00+B11)
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),
        # 2: (A10+A11)B00
        ({3:1, 4:1}, {0:1}, {3:1, 4:-1}),
        # 3: A00(B01-B11)
        ({0:1}, {1:1, 4:-1}, {1:1, 4:1}),
        # 4: A11(B10-B00)
        ({4:1}, {3:1, 0:-1}, {0:1, 3:1}),
        # 5: (A00+A01)B11
        ({0:1, 1:1}, {4:1}, {0:-1, 3:1}),
        # 6: (A10-A00)(B00+B01)
        ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
        # 7: (A01-A11)(B10+B11)
        ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),
        # 8: (A02+A12)(B20+B21)  -- shared term for top-right and bottom-left
        ({2:1, 5:1}, {6:1, 7:1}, {0:1, 3:1}),
        # 9: A02*B21 - A12*B20  (correction term)
        ({2:1, 5:-1}, {7:1, 6:-1}, {1:1, 4:1}),
        # 10: A02*B22
        ({2:1}, {8:1}, {2:1}),
        # 11: A12*B22
        ({5:1}, {8:1}, {5:1}),
        # 12: A20*B00
        ({6:1}, {0:1}, {6:1}),
        # 13: A21*B10
        ({7:1}, {3:1}, {6:1}),
        # 14: A22*B20
        ({8:1}, {6:1}, {6:1}),
        # 15: A20*B01
        ({6:1}, {1:1}, {7:1}),
        # 16: A21*B11
        ({7:1}, {4:1}, {7:1}),
        # 17: A22*B21
        ({8:1}, {7:1}, {7:1}),
        # 18: A00*B02
        ({0:1}, {2:1}, {2:1}),
        # 19: A01*B02
        ({1:1}, {2:1}, {2:1}),
        # 20: A10*B12
        ({3:1}, {5:1}, {5:1}),
        # 21: A11*B12
        ({4:1}, {5:1}, {5:1}),
        # 22: A20*B02
        ({6:1}, {2:1}, {8:1}),
        # 23: A21*B12
        ({7:1}, {5:1}, {8:1}),
    ]
    
    # This is 23 terms but I'm not confident they're correct. Let me verify.
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
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
