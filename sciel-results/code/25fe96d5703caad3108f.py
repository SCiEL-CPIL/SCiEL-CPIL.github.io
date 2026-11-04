"""Rewritten program for 3x5x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s wall-clock
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We need to find rank-r decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # The tensor has n*m*p = 105 nonzero entries.
    # Lower bound on rank: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But actual lower bound is higher.
    
    # Strategy: Use a combination of Strassen and schoolbook blocks
    # to get a good rank, then try to reduce via linear algebra.
    
    # First, build a baseline decomposition using block tiling.
    
    def build_baseline():
        """Build a baseline decomposition using Strassen + schoolbook blocks."""
        rank = 99  # Start with known good rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Helper: add Strassen 2x2x2 block
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0 * m + j0][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[j0 * p + k0][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[k0 * n + i0][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            # M2
            U[(i0 + 1) * m + j0][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + i0 + 1][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] -= 1
            col += 1
            
            # M3
            U[i0 * m + j0][col] += 1
            V[j0 * p + k0 + 1][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] -= 1
            W[(k0 + 1) * n + i0][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            # M4
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[(j0 + 1) * p + k0][col] += 1
            V[j0 * p + k0][col] -= 1
            W[k0 * n + i0][col] += 1
            W[k0 * n + i0 + 1][col] += 1
            col += 1
            
            # M5
            U[i0 * m + j0][col] += 1
            U[i0 * m + j0 + 1][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[(k0 + 1) * n + i0][col] += 1
            W[k0 * n + i0][col] -= 1
            col += 1
            
            # M6
            U[(i0 + 1) * m + j0][col] += 1
            U[i0 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k0 + 1][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            # M7
            U[i0 * m + j0 + 1][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] -= 1
            V[(j0 + 1) * p + k0][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[k0 * n + i0][col] += 1
            col += 1
        
        # Helper: add schoolbook terms
        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        # Part 1: j in {0, 1}
        for k0 in [0, 2, 4]:
            add_strassen(0, 0, k0)
        add_schoolbook([0, 1], [0, 1], [6])
        add_schoolbook([2], [0, 1], range(7))
        
        # Part 2: j in {2, 3, 4}
        for k0 in [0, 2, 4]:
            add_strassen(0, 2, k0)
        add_schoolbook([0, 1], [2, 3], [6])
        add_schoolbook([0, 1], [4], range(7))
        add_schoolbook([2], [2, 3, 4], range(7))
        
        return U, V, W, col
    
    U, V, W, actual_rank = build_baseline()
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Expected value
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                    
                    # Computed value
                    computed = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    
                    if abs(computed - expected) > 1e-9:
                        return False, (a, b, c, expected, computed)
        return True, None
    
    is_valid, err = verify(U, V, W, actual_rank)
    if not is_valid:
        # Fallback to pure schoolbook if something went wrong
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
        actual_rank = col
        return {"U": U, "V": V, "W": W}
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    # The current decomposition has rank 99.
    # We verify it and return it.
    return {"U": U, "V": V, "W": W}
