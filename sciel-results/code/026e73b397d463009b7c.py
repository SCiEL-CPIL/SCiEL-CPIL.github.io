"""Matrix multiplication 2x5x5 rank-minimal decomposition search."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Return list of (i*m+j, j*p+k, k*n+i) for all nonzero entries."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entries.append((i * m + j, j * p + k, k * n + i))
    return entries

def _verify(U, V, W, entries):
    """Check if U, V, W (lists of lists of Fraction) decompose the tensor."""
    rank = len(U[0]) if U else 0
    for a, b, c in entries:
        s = Fraction(0)
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if s != 1:
            return False
    # Check non-entries (all other index triples should sum to 0)
    n, m, p = 2, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) in entries:
                    continue
                s = Fraction(0)
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 0:
                    return False
    return True

def _verify_fast(U, V, W, entries_set, nm, mp, pn):
    """Faster verification using the entries set."""
    rank = len(U[0]) if U else 0
    # First check entries
    for (a, b, c) in entries_set:
        s = Fraction(0)
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if s != 1:
            return False
    # Check non-entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) in entries_set:
                    continue
                s = Fraction(0)
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 0:
                    return False
    return True

def _random_decomposition(rank, nm, mp, pn, entries):
    """Generate a random decomposition with small integer entries."""
    U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
    V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
    W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(pn)]
    # Convert to Fraction for verification
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    return Uf, Vf, Wf

def _try_local_search(U, V, W, entries, nm, mp, pn, max_iter=10000):
    """Try to fix a decomposition by local changes."""
    rank = len(U[0])
    entries_set = set(entries)
    
    # Count errors
    def count_errors(Uf, Vf, Wf):
        err = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    expected = 1 if (a, b, c) in entries_set else 0
                    if s != expected:
                        err += 1
        return err
    
    # This is too slow for full enumeration. Use a simpler approach.
    # Just return current if it's already valid.
    if _verify(U, V, W, entries):
        return U, V, W, True
    return U, V, W, False

def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    entries = _build_tensor(n, m, p)
    entries_set = set(entries)
    
    deadline = time.time() + 110  # 110s safety margin before 120s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank via random search
    # For 2x5x5, rank is at least 25 (since it contains 5x5x1 sub-tensor with rank 25? No, 1x5x5 has rank 25? Actually min rank of 1xn x nxn is n^2? No, it's n. Wait.
    
    # Actually, the matrix multiplication tensor T_{n,m,p} has rank at least max(n*m, m*p, p*n) and at most n*m*p.
    # For 2x5x5: at least 25, at most 50.
    
    # Let's try ranks from 25 up to 49
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        # Try random searches for this rank
        max_attempts = 50 if rank <= 35 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            Uf, Vf, Wf = _random_decomposition(rank, nm, mp, pn, entries)
            
            # Quick check: verify
            if _verify(Uf, Vf, Wf, entries):
                best_rank = rank
                best_U, best_V, best_W = Uf, Vf, Wf
                break
        else:
            continue
        break
    
    # Convert best to list of ints (or Fractions)
    U_out = [[str(x.numerator) + '/' + str(x.denominator) if x.denominator != 1 else str(x.numerator) for x in row] for row in best_U]
    V_out = [[str(x.numerator) + '/' + str(x.denominator) if x.denominator != 1 else str(x.numerator) for x in row] for row in best_V]
    W_out = [[str(x.numerator) + '/' + str(x.denominator) if x.denominator != 1 else str(x.numerator) for x in row] for row in best_W]
    
    # Actually, the interface expects integer/half-integer entries. Let's just return the numeric values.
    U_final = [[float(x) for x in row] for row in best_U]
    V_final = [[float(x) for x in row] for row in best_V]
    W_final = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
