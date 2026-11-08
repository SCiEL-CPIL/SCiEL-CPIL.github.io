# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B using NumPy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(Ua[a, :rank] * Va[b, :rank], Wa[c, :rank])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                i_prime = c % n
                k_prime = c // n
                
                if i == i_prime and j == j_prime and k == k_prime:
                    continue
                
                val = np.dot(Ua[a, :rank] * Va[b, :rank], Wa[c, :rank])
                if abs(val) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # Direct terms for row 2 (A02, A12) with B2
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _search_rank10_block(deadline):
    """
    Search for a rank-10 decomposition of the 2x3 x 3x2 tensor.
    If found, return (U, V, W) for the block.
    U: 6x10, V: 6x10, W: 4x10
    """
    import random
    random.seed(42)
    
    # Define the target tensor T for 2x3 x 3x2
    # Indices: i in 0..1, j in 0..2, k in 0..1
    # T[i, j, k, i', j', k'] = 1 if i==i' and j==j' and k==k'
    # We need U (6x10), V (6x10), W (4x10) such that
    # sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    # where a = i*3+j, b = j*2+k, c = k*2+i
    
    # We will use a local search / random restart approach.
    # Initialize with the known rank-11 solution and try to zero out one column.
    
    # Rank-11 solution for the block:
    U11 = np.zeros((6, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((4, 11))
    
    # Strassen terms for j=0,1
    # M1
    U11[0, 0] += 1; U11[4, 0] += 1
    V11[0, 0] += 1; V11[3, 0] += 1
    W11[0, 0] += 1; W11[3, 0] += 1
    # M2
    U11[3, 1] += 1; U11[4, 1] += 1
    V11[0, 1] += 1
    W11[1, 1] += 1; W11[3, 1] -= 1
    # M3
    U11[0, 2] += 1
    V11[1, 2] += 1; V11[3, 2] -= 1
    W11[2, 2] += 1; W11[3, 2] += 1
    # M4
    U11[4, 3] += 1
    V11[2, 3] += 1; V11[0, 3] -= 1
    W11[0, 3] += 1; W11[1, 3] += 1
    # M5
    U11[0, 4] += 1; U11[1, 4] += 1
    V11[3, 4] += 1
    W11[0, 4] -= 1; W11[2, 4] += 1
    # M6
    U11[3, 5] += 1; U11[0, 5] -= 1
    V11[0, 5] += 1; V11[1, 5] += 1
    W11[3, 5] += 1
    # M7
    U11[1, 6] += 1; U11[4, 6] -= 1
    V11[2, 6] += 1; V11[3, 6] += 1
    W11[0, 6] += 1
    
    # Direct terms for j=2
    U11[2, 7] += 1; V11[4, 7] += 1; W11[0, 7] += 1
    U11[2, 8] += 1; V11[5, 8] += 1; W11[2, 8] += 1
    U11[5, 9] += 1; V11[4, 9] += 1; W11[1, 9] += 1
    U11[5, 10] += 1; V11[5, 10] += 1; W11[3, 10] += 1
    
    # Verify rank-11
    # (Assume it's correct)
    
    # Try to find a rank-10 solution by random search
    for _ in range(500):
        if time.time() > deadline - 20:
            break
        
        U = np.zeros((6, 10))
        V = np.zeros((6, 10))
        W = np.zeros((4, 10))
        
        # Random initialization with small integers
        for r in range(10):
            # Randomly select non-zero positions
            for a in range(6):
                if random.random() < 0.5:
                    U[a, r] = random.choice([-1, 1])
            for b in range(6):
                if random.random() < 0.5:
                    V[b, r] = random.choice([-1, 1])
            for c in range(4):
                if random.random() < 0.5:
                    W[c, r] = random.choice([-1, 1])
        
        # Verify
        valid = True
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    a = i * 3 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: continue
        
        for a in range(6):
            for b in range(6):
                for c in range(4):
                    i = a // 3
                    j = a % 3
                    j_p = b // 2
                    k = b % 2
                    i_p = c % 2
                    k_p = c // 2
                    if i == i_p and j == j_p and k == k_p:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: continue
        
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank-55
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                # Check if we can find a better one
                if time.time() < deadline - 30:
                    block_sol = _search_rank10_block(deadline)
                    if block_sol:
                        U_b, V_b, W_b = block_sol
                        # Construct rank-50 solution
                        rank = 50
                        U = [[0] * rank for _ in range(n * m)]
                        V = [[0] * rank for _ in range(m * p)]
                        W = [[0] * rank for _ in range(p * n)]
                        
                        for t in range(5):
                            k0 = 2 * t
                            k1 = 2 * t + 1
                            for r in range(10):
                                col = t * 10 + r
                                for a in range(6):
                                    U[a][col] += U_b[a][r]
                                for b in range(6):
                                    # Map b (j*2+k) to global b (j*p+k)
                                    j = b // 2
                                    k = b % 2
                                    global_b = j * p + (k0 + k)
                                    V[global_b][col] += V_b[b][r]
                                for c in range(4):
                                    # Map c (k*2+i) to global c (k*n+i)
                                    k = c // 2
                                    i = c % 2
                                    global_c = (k0 + k) * n + i
                                    W[global_c][col] += W_b[c][r]
                        
                        if _verify(U, V, W, n, m, p, rank):
                            return {"U": U, "V": V, "W": W}
                
                return res55
    except Exception:
        pass
    
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
