import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def _build_rank_44():
    """Build the standard rank-44 decomposition using 1+2 split with Strassen."""
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    for block in range(4):
        # Strassen steps for 2x2 block
        # 1. (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1 * p + block * 2][col] = 1
        V[2 * p + block * 2 + 1][col] = 1
        W[(block * 2) * n + 0][col] = 1
        W[(block * 2 + 1) * n + 1][col] = 1
        col += 1
        
        # 2. (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1 * p + block * 2][col] = 1
        W[(block * 2) * n + 1][col] = 1
        W[(block * 2 + 1) * n + 1][col] = -1
        col += 1
        
        # 3. a00(b01-b11)
        U[1][col] = 1
        V[1 * p + block * 2 + 1][col] = 1
        V[2 * p + block * 2 + 1][col] = -1
        W[(block * 2 + 1) * n + 0][col] = 1
        W[(block * 2 + 1) * n + 1][col] = 1
        col += 1
        
        # 4. a11(b10-b00)
        U[5][col] = 1
        V[2 * p + block * 2][col] = 1
        V[1 * p + block * 2][col] = -1
        W[(block * 2) * n + 0][col] = 1
        W[(block * 2) * n + 1][col] = 1
        col += 1
        
        # 5. (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2 * p + block * 2 + 1][col] = 1
        W[(block * 2) * n + 0][col] = -1
        W[(block * 2 + 1) * n + 0][col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1 * p + block * 2][col] = 1
        V[1 * p + block * 2 + 1][col] = 1
        W[(block * 2 + 1) * n + 1][col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2 * p + block * 2][col] = 1
        V[2 * p + block * 2 + 1][col] = 1
        W[(block * 2) * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Start with the known rank-44 solution
    best_result = _build_rank_44()
    best_rank = 44
    
    n, m, p = 2, 3, 8
    
    # Attempt to improve via search: try to find rank 43, 42, ...
    # Strategy: Start from rank-44 solution and try to merge/eliminate terms
    # Or: try different block splits
    
    # Try alternative: 2+1 split (same rank but different structure)
    # This won't improve rank but might enable further optimization
    
    # For now, return the known best
    if time.time() > deadline:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    n, m, p = 2, 3, 8
    U, V, W = result["U"], result["V"], result["W"]
    print(f"Rank: {len(U[0])}")
    print(f"Verified: {_verify_decomposition(U, V, W, n, m, p)}")
