"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                expected = 1.0 if i == k else 0.0
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _known_rank26():
    """Return a known valid rank-26 decomposition."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen for 2x2 block
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
    # Cross terms
    cross = [(2,6,0),(2,7,3),(5,6,1),(5,7,4),(0,2,6),(1,5,6),(3,2,7),(4,5,7),
             (2,8,6),(5,8,7),(6,0,2),(7,3,2),(6,1,5),(7,4,5),(8,6,2),(8,7,5),
             (6,2,8),(7,5,8),(8,8,8)]
    for (u,v,w) in cross:
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank25_structured():
    """
    Try to find a rank-25 decomposition by exploiting block structure.
    We treat the 3x3 matrix as a 2x2 block matrix:
    A = [[A00, A01], [A10, A11]] where A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1
    Similarly for B and C.
    
    The product C = A*B has blocks:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    We use Strassen for the 2x2 part (7 multiplications) and need to handle
    the interactions with the 1x1 and 1x2/2x1 blocks efficiently.
    
    The naive approach uses 7 (Strassen) + 12 (cross terms) = 19, but we need
    to account for the fact that some cross terms can be combined.
    
    Actually, let's think about this more carefully. The total number of scalar
    multiplications needed is 27 (for the full 3x3). Strassen saves 1 multiplication
    for the 2x2 block, giving us 7 + 12 = 19... wait, that's not right either.
    
    Let me reconsider. The 3x3 multiplication has 27 terms. If we group them into
    blocks:
    - 4 terms in C00 from A00*B00 (2x2 * 2x2)
    - 2 terms in C00 from A01*B10 (2x1 * 1x2)
    - 2 terms in C01 from A00*B01 (2x2 * 2x1)
    - 1 term in C01 from A01*B11 (2x1 * 1x1)
    - 2 terms in C10 from A10*B00 (1x2 * 2x2)
    - 1 term in C10 from A11*B10 (1x1 * 1x2)
    - 1 term in C11 from A10*B01 (1x2 * 2x1)
    - 1 term in C11 from A11*B11 (1x1 * 1x1)
    
    Total: 4+2+2+1+2+1+1+1 = 14... that doesn't add up to 27.
    
    I think I'm confusing myself. Let me just try a direct search for rank 25
    using the structure as a starting point.
    """
    n, m, p = 3, 3, 3
    target_rank = 25
    
    # Start from the known rank-26 decomposition and try to merge two terms
    base = _known_rank26()
    U = [row[:] for row in base['U']]
    V = [row[:] for row in base['V']]
    W = [row[:] for row in base['W']]
    
    # Try to see if we can eliminate one column by making it a linear combination
    # of others, or by finding a dependency.
    
    # For now, just return None to indicate no improvement found
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    best_result = _known_rank26()
    best_rank = 26
    
    # Try structured rank-25 search
    r25 = _try_rank25_structured()
    if r25 is not None:
        if _validate(r25['U'], r25['V'], r25['W'], n, m, p):
            best_result = r25
            best_rank = 25
    
    # Try random search for rank 25 if we didn't find a structured one
    if best_rank == 26:
        if time.time() < deadline:
            max_iters = 3000
            for attempt in range(max_iters):
                if time.time() > deadline:
                    break
                target_rank = 25
                U = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(n*m)]
                V = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(m*p)]
                W = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(p*n)]
                
                if _validate(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = 25
                    break
    
    return best_result

# EVOLVE-BLOCK-END
