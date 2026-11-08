"""Matrix multiplication 4,4,9 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Build schoolbook decomposition using Fraction for exact arithmetic
    schoolbook_rank = n * m * p  # 144
    U = [[Fraction(0)] * schoolbook_rank for _ in range(n * m)]
    V = [[Fraction(0)] * schoolbook_rank for _ in range(m * p)]
    W = [[Fraction(0)] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # Apply BCLR elimination steps to reduce rank
    U2, V2, W2, eliminated = _bclr_eliminate(U, V, W, n, m, p, max_steps=1000)
    if eliminated:
        if _verify_decomposition(U2, V2, W2, n, m, p):
            new_rank = len(U2[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U2, V2, W2
                U, V, W = U2, V2, W2
    
    # Try random perturbations and re-eliminate
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        U_perm, V_perm, W_perm = _permute_columns(U, V, W, n, m, p, trial)
        U3, V3, W3, elim2 = _bclr_eliminate(U_perm, V_perm, W_perm, n, m, p, max_steps=500)
        if elim2:
            if _verify_decomposition(U3, V3, W3, n, m, p):
                new_rank = len(U3[0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U3, V3, W3
                    U, V, W = U3, V3, W3
    
    # Convert to output format
    def to_output(mat):
        out = []
        for row in mat:
            out_row = []
            for val in row:
                if val.denominator == 1:
                    out_row.append(int(val))
                else:
                    doubled = val * 2
                    if doubled.denominator == 1:
                        out_row.append(int(doubled) / 2.0)
                    else:
                        out_row.append(float(val))
            out.append(out_row)
        return out
    
    return {"U": to_output(best_U), "V": to_output(best_V), "W": to_output(best_W)}


def _bclr_eliminate(U, V, W, n, m, p, max_steps=1000):
    """Apply BCLR elimination to reduce rank. Returns (U, V, W, eliminated_any)."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    eliminated_any = False
    
    for step in range(max_steps):
        if rank <= 1:
            break
        
        # Find a column c where we can eliminate an entry
        found = False
        for c in range(rank):
            # Look for two distinct i values where W[k*n + i, c] != 0 for same k
            i_vals = {}
            for k in range(p):
                for i in range(n):
                    if W[k * n + i][c] != 0:
                        if k not in i_vals:
                            i_vals[k] = []
                        i_vals[k].append(i)
            
            # Find a k with at least 2 i values
            target_k = None
            target_i1 = None
            target_i2 = None
            for k, i_list in i_vals.items():
                if len(i_list) >= 2:
                    target_k = k
                    target_i1 = i_list[0]
                    target_i2 = i_list[1]
                    break
            
            if target_k is None:
                continue
            
            # Find j0 such that U[i1*m + j0, c] != 0
            j0 = None
            for j in range(m):
                if U[target_i1 * m + j][c] != 0:
                    j0 = j
                    break
            
            if j0 is None:
                continue
            
            # Find k0 such that V[j0*p + k0, c] != 0
            k0 = None
            for k in range(p):
                if V[j0 * p + k][c] != 0:
                    k0 = k
                    break
            
            if k0 is None:
                continue
            
            # Perform elimination
            alpha = W[target_k * n + target_i2][c] / W[target_k * n + target_i1][c]
            
            # Update U: for all i', j'
            for i_prime in range(n):
                for j_prime in range(m):
                    U[i_prime * m + j_prime][c] -= alpha * U[target_i1 * m + j_prime][c] * (1 if i_prime == target_i1 else 0)
            
            # Update V: for all j', k'
            for j_prime in range(m):
                for k_prime in range(p):
                    V[j_prime * p + k_prime][c] -= alpha * V[j0 * p + k_prime][c] * (1 if j_prime == j0 else 0)
            
            # Update W: for all k', i'
            for k_prime in range(p):
                for i_prime in range(n):
                    W[k_prime * n + i_prime][c] -= alpha * W[k0 * n + i_prime][c] * (1 if k_prime == target_k else 0)
            
            found = True
            eliminated_any = True
            break
        
        if not found:
            break
    
    return U, V, W, eliminated_any


def _permute_columns(U, V, W, n, m, p, seed):
    """Randomly permute columns of U, V, W."""
    rng = random.Random(seed)
    rank = len(U[0])
    perm = list(range(rank))
    rng.shuffle(perm)
    
    U2 = [[U[i][perm[c]] for c in range(rank)] for i in range(len(U))]
    V2 = [[V[i][perm[c]] for c in range(rank)] for i in range(len(V))]
    W2 = [[W[i][perm[c]] for c in range(rank)] for i in range(len(W))]
    
    return U2, V2, W2


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if val != Fraction(1):
                    return False
    return True
# EVOLVE-BLOCK-END
