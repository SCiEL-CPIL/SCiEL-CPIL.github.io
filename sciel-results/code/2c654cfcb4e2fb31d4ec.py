import numpy as np
import time
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p matrix multiplication."""
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

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = Fraction(0)
                for col in range(rank):
                    actual += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
                
                if actual != expected:
                    return False
    return True

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] for matrix multiplication."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _try_column_elimination(U, V, W, n, m, p, deadline, max_eliminations=6):
    """
    Try to eliminate columns from the schoolbook decomposition.
    For each column, check if it can be expressed as a linear combination of others.
    """
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try to eliminate columns one by one
    for _ in range(max_eliminations):
        if time.time() > deadline - 20:
            break
        
        eliminated = False
        for c in range(len(U[0])):
            if time.time() > deadline - 20:
                break
            
            # Check if column c can be eliminated
            # We need to find coefficients alpha such that for all (a,b,c_idx):
            # U[a,c]*V[b,c]*W[c_idx,c] = sum_{c' != c} alpha_{c'} * U[a,c']*V[b,c']*W[c_idx,c']
            
            # This is a linear system. Let's try a simpler approach:
            # Try removing column c and see if the tensor is still valid
            
            U_test = [row[:] for row in U]
            V_test = [row[:] for row in V]
            W_test = [row[:] for row in W]
            
            # Remove column c
            for a in range(dim_a):
                U_test[a] = U_test[a][:c] + U_test[a][c+1:]
            for b in range(dim_b):
                V_test[b] = V_test[b][:c] + V_test[b][c+1:]
            for c_idx in range(dim_c):
                W_test[c_idx] = W_test[c_idx][:c] + W_test[c_idx][c+1:]
            
            # Check if still valid
            if _validate(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                best_rank = len(U[0])
                eliminated = True
                break
        
        if not eliminated:
            break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def _try_structured_search(n, m, p, deadline, target_rank=30):
    """
    Try to find a structured low-rank decomposition.
    For 3x3x4, we can try to use shared computations across rows.
    """
    if n != 3 or m != 3 or p != 4:
        return None
    
    dim_a = n * m  # 9
    dim_b = m * p  # 12
    dim_c = p * n  # 12
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Try to construct a decomposition with rank < 36
    # Strategy: Use the structure that C[i,k] = sum_j A[i,j]*B[j,k]
    # We can try to share computations across different (i,k) pairs
    
    # For now, try a simple approach: use schoolbook and try to merge terms
    sb = _schoolbook(n, m, p)
    U = [[Fraction(x) for x in row] for row in sb["U"]]
    V = [[Fraction(x) for x in row] for row in sb["V"]]
    W = [[Fraction(x) for x in row] for row in sb["W"]]
    
    # Try random perturbations to see if we can reduce rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = len(U[0])
    
    for attempt in range(20):
        if time.time() > deadline - 15:
            break
        
        # Try to find a linear dependency among columns
        # This is a heuristic approach
        
        # For each pair of columns, check if one can be expressed as a combination of others
        for c1 in range(len(U[0])):
            if time.time() > deadline - 15:
                break
            for c2 in range(c1 + 1, len(U[0])):
                # Check if column c2 can be eliminated given column c1
                U_test = [row[:] for row in U]
                V_test = [row[:] for row in V]
                W_test = [row[:] for row in W]
                
                # Try to express column c2 as a multiple of column c1
                # This is a simplification, but let's try it
                
                # Check if the tensor contributions are proportional
                can_eliminate = True
                for a in range(dim_a):
                    for b in range(dim_b):
                        for c_idx in range(dim_c):
                            contrib_c1 = U[a][c1] * V[b][c1] * W[c_idx][c1]
                            contrib_c2 = U[a][c2] * V[b][c2] * W[c_idx][c2]
                            
                            if contrib_c1 != 0 and contrib_c2 != 0:
                                ratio = contrib_c2 / contrib_c1
                                # Check if this ratio is consistent
                                # This is a simplified check
                                pass
                            elif contrib_c1 != 0 or contrib_c2 != 0:
                                can_eliminate = False
                                break
                        if not can_eliminate:
                            break
                    if not can_eliminate:
                        break
                    if not can_eliminate:
                        break
                
                if can_eliminate:
                    # Try to eliminate column c2
                    for a in range(dim_a):
                        U_test[a] = U_test[a][:c2] + U_test[a][c2+1:]
                    for b in range(dim_b):
                        V_test[b] = V_test[b][:c2] + V_test[b][c2+1:]
                    for c_idx in range(dim_c):
                        W_test[c_idx] = W_test[c_idx][:c2] + W_test[c_idx][c2+1:]
                    
                    if _validate(U_test, V_test, W_test, n, m, p):
                        U, V, W = U_test, V_test, W_test
                        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                        best_rank = len(U[0])
                        if best_rank <= target_rank:
                            return {"U": best_U, "V": best_V, "W": best_W}
    
    if best_rank < 36:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try column elimination
    result = _try_column_elimination(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
            best_rank = rank
    
    # Try structured search if we still have time
    if best_rank > 30 and time.time() < deadline - 30:
        result = _try_structured_search(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
