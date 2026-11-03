"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def schoolbook():
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

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U,V,W give a valid tensor decomposition."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s - 1) > tol:
                            return False
                    else:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # The tensor for 2x3 * 3x7 has bilinear complexity at most 42 (schoolbook).
    # We try to find a lower rank decomposition.
    
    # Key structural insight: For fixed j, the terms A[i,j]*B[j,k] for i in {0,1}, k in {0..6}
    # form a 2x7 outer product structure. The tensor rank of a 2x7 matrix multiplication
    # (viewed as a bilinear map) is 14 (since it's just element-wise product of a 2-vector and 7-vector).
    # But we can potentially share across different j values.
    
    # Let's try a more systematic approach using the structure of the problem.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For n=2, m=3, p=7, the schoolbook rank is 42.
    # Can we do better? Let's try to find a rank-36 or lower decomposition.
    
    # Approach: Use a parametric family of decompositions.
    # For each j, define the contribution as a sum over some auxiliary variables.
    
    # Let's try a direct search for small rank improvements.
    # We'll use a structured random search that respects the tensor structure.
    
    def structured_search(rank, max_iters=500, seed=42):
        """Structured random search for rank-r decomposition."""
        random.seed(seed)
        entries = [-1, -0.5, 0, 0, 0, 0, 0, 0.5, 1]
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Assign each term (i,j,k) to a column, allowing sharing
            # For each (i,j,k), we need U[i*m+j][col] * V[j*p+k][col] * W[k*n+i][col] to contribute
            # Let's try assigning each term to a random column and setting values accordingly.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = random.randint(0, rank - 1)
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        
                        # To make the product contribute 1, we can set:
                        # U[a][col] = 1, V[b][col] = 1, W[c][col] = 1
                        # But this may conflict with other terms using the same column.
                        
                        # Instead, let's just try random values and verify.
                        U[a][col] = random.choice(entries)
                        V[b][col] = random.choice(entries)
                        W[c][col] = random.choice(entries)
            
            # Add some random entries to help with cancellation
            for a in range(n * m):
                for r in range(rank):
                    if random.random() < 0.05:
                        U[a][r] = random.choice(entries)
            for b in range(m * p):
                for r in range(rank):
                    if random.random() < 0.05:
                        V[b][r] = random.choice(entries)
            for c in range(p * n):
                for r in range(rank):
                    if random.random() < 0.05:
                        W[c][r] = random.choice(entries)
            
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try to find a better rank using structured search
    for r in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 10.0:
            break
        result = structured_search(r, max_iters=100, seed=r * 7919 + 13)
        if result is not None:
            best = result
            best_rank = r
            break

    # Final verification
    if verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
