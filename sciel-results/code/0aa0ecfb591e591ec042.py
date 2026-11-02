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

    # Key structural insight: The tensor T[(i,j),(j,k),(k,i)] = 1
    # can be decomposed by exploiting the fact that for each j, the 14
    # entries form a "grid" that can be covered with fewer rank-1 terms
    # if we allow the rank-1 terms to have support on multiple j values.
    #
    # Approach: Use a parametric family where each rank term is defined
    # by a pattern of nonzeros in U, V, W. We search over such patterns.

    def parametric_search(target_rank, max_attempts=1000, seed=42):
        """Search using parametric patterns for rank terms."""
        random.seed(seed)
        
        # The 42 target (i,j,k) triples
        all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10.0:
                return None
            
            # Assign each triple to a rank column
            assignments = [[] for _ in range(target_rank)]
            
            # Shuffle and assign
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            
            for idx, triple in enumerate(shuffled):
                # Cycle through columns
                col = idx % target_rank
                assignments[col].append(triple)
            
            # Build U, V, W
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for col, triples in enumerate(assignments):
                if not triples:
                    continue
                
                # For this column, set U, V, W to cover all triples
                # Strategy: set U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 for all
                # triples in this column. This may create spurious nonzeros,
                # but we'll verify.
                for (i, j, k) in triples:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
            
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            # If direct assignment doesn't work, try with signs
            for sign_pattern in range(min(4, 1 << min(len(assignments), 3))):
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                for col, triples in enumerate(assignments):
                    if not triples:
                        continue
                    sign = 1 if (sign_pattern >> col) & 1 else -1
                    for (i, j, k) in triples:
                        U[i * m + j][col] = sign
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                
                if verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Try to find a decomposition with rank < 42
    # The theoretical lower bound is max(6, 21, 14) = 21
    for r in range(min(best_rank - 1, 35), 20, -1):
        if time.time() > deadline - 15.0:
            break
        result = parametric_search(r, max_attempts=500, seed=r * 7919)
        if result is not None:
            best = result
            best_rank = r
            break

    return best
# EVOLVE-BLOCK-END
