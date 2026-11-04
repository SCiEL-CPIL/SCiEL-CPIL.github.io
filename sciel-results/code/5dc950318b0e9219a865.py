"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    The decomposition should satisfy:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    and 0 for all other (a, b, c) combinations.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    
    # Check all (i, j, k) triples: should sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    
    # Check that entries where the tensor is 0 are indeed 0
    # The tensor T[a,b,c] = 1 iff a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # where i=i', j=j', k=k' (i.e., the middle index matches)
    # So T[a,b,c] = 1 iff (a mod m) == (b // p) and (b mod p) == (c // n) and (c mod n) == (a // m)
    # Actually, let's be more careful.
    # a = i*m + j where i in [0,n), j in [0,m)
    # b = j'*p + k where j' in [0,m), k in [0,p)
    # c = k'*n + i' where k' in [0,p), i' in [0,n)
    # T[a,b,c] = 1 iff i=i', j=j', k=k'
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # So T[a,b,c] = 1 iff a//m == c%n and a%m == b//p and b%p == c//n
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                k_prime = c // n
                i_prime = c % n
                
                should_be_one = (i == i_prime) and (j == j_prime) and (k == k_prime)
                
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                
                if should_be_one:
                    if s != 1:
                        return False
                else:
                    if s != 0:
                        return False
    
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j]
                v_val = V_arr[j*p+k]
                w_val = W_arr[k*n+i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check zero entries
    # T[a,b,c] = 1 iff a//m == c%n, a%m == b//p, b%p == c//n
    # For all other (a,b,c), the sum should be 0.
    # This is expensive to check all, but we can check a sample.
    # Actually, let's check all since nm*mp*pn = 15*30*18 = 8100 which is manageable.
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                k_prime = c // n
                i_prime = c % n
                
                should_be_one = (i == i_prime) and (j == j_prime) and (k == k_prime)
                
                u_val = U_arr[a]
                v_val = V_arr[b]
                w_val = W_arr[c]
                s = np.dot(u_val, v_val * w_val)
                
                if should_be_one:
                    if abs(s - 1) > 1e-9:
                        return False
                else:
                    if abs(s) > 1e-9:
                        return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Deadline guard: 150s limit, use 130s for safety
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Trivial decomposition with rank 90
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Strategy: Try to find a lower rank decomposition using a block-partitioning approach.
    # The 3x5x6 multiplication can be seen as:
    #   A (3x5) * B (5x6) = C (3x6)
    #   
    # We can partition A into blocks and B into blocks.
    # One approach: use the fact that we can combine some of the rank-1 terms.
    #
    # Let's try a different approach: use a known low-rank decomposition for a sub-problem.
    # For example, if we can find a rank-r decomposition for a 2x2x2 sub-problem (Strassen, rank 7),
    # we might be able to use it to reduce the overall rank.
    #
    # Actually, let's try a more direct approach: use a greedy algorithm to try to merge terms.
    # Start with the trivial decomposition and try to find linear dependencies among the terms.
    #
    # Another approach: use random search with small integer entries.
    # For rank r, we have nm*r + mp*r + pn*r = (15+30+18)*r = 63r unknowns.
    # We have 8100 equations (but many are trivially satisfied).
    # The number of non-trivial equations is 90 (for the 1-entries) plus the zero constraints.
    #
    # Let's try a systematic search for rank r = 80, 70, 60, etc.
    # We'll use a randomized approach: generate random U, V, W with small integer entries
    # and check if they work. This is unlikely to work, but let's try.
    #
    # Actually, a better approach: use the structure of the problem.
    # The tensor is T[i,j,k] = 1 for all i,j,k, and 0 otherwise (in the appropriate indexing).
    # We can try to find a decomposition by solving a system of linear equations.
    #
    # For rank r, we can fix U and V to be random (or structured) and solve for W.
    # Then check if the resulting W gives the correct tensor.
    #
    # Let's try this approach:
    # 1. For a given rank r, generate random U and V with small integer entries.
    # 2. For each (a, b, c) where T[a,b,c] = 1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1.
    #    This is a linear system in W[c,r] for fixed c.
    # 3. For each c, we have nm*mp equations in r unknowns (W[c,0], ..., W[c,r-1]).
    #    This is overdetermined unless r is large enough.
    #
    # Actually, for a fixed c, the equations are:
    #   For all (a, b) such that T[a,b,c] = 1: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    #   For all (a, b) such that T[a,b,c] = 0: sum_r U[a,r]*V[b,r]*W[c,r] = 0
    #
    # The number of (a,b) pairs with T[a,b,c] = 1 is:
    #   c = k*n + i, so i = c % n, k = c // n
    #   T[a,b,c] = 1 iff a//m == i, a%m == b//p, b%p == k
    #   So a = i*m + j, b = j*p + k for j in [0,m)
    #   There are m such pairs.
    #
    # The number of (a,b) pairs with T[a,b,c] = 0 is nm*mp - m.
    #
    # So for each c, we have m equations equal to 1 and (nm*mp - m) equations equal to 0.
    # Total: nm*mp = 15*30 = 450 equations in r unknowns.
    #
    # For this to be solvable, we need r >= 450 (in general), which is worse than the trivial solution.
    # So this approach doesn't work directly.
    #
    # Let's try a different approach: use the fact that the tensor has a specific structure.
    # We can try to find a decomposition by exploiting the symmetry.
    #
    # Actually, let's just return the trivial decomposition for now.
    # The search for a lower rank decomposition is non-trivial and may not succeed within the time limit.
    
    return best_result

# EVOLVE-BLOCK-END
