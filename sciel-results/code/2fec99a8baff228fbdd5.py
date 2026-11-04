"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses structured algebraic search with exact rational verification.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _verify_exact(n, m, p, U, V, W):
    """
    Verify exact decomposition using Fraction arithmetic.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    uc = u_row[c] if isinstance(u_row[c], Fraction) else Fraction(u_row[c])
                    vc = v_row[c] if isinstance(v_row[c], Fraction) else Fraction(v_row[c])
                    wc = w_row[c] if isinstance(w_row[c], Fraction) else Fraction(w_row[c])
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _verify_float(n, m, p, U, V, W, tol=1e-10):
    """Fast float verification."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    return True


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with Fraction entries.
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U[i*m+j, c] for i,j in {0,1}, m=2
    # V[j*p+k, c] for j,k in {0,1}, p=2
    # W[k*n+i, c] for k,i in {0,1}, n=2
    
    U = [[Fraction(0)] * 7 for _ in range(4)]
    V = [[Fraction(0)] * 7 for _ in range(4)]
    W = [[Fraction(0)] * 7 for _ in range(4)]
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    # M1: (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[0*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1; W[0*2+1, 0] = 1; W[1*2+0, 0] = 1
    
    # M2: (A10+A11)B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+0, 1] = 1; W[0*2+1, 1] = 1
    
    # M3: A00(B10-B11)
    U[0*2+0, 2] = 1
    V[1*2+0, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+0, 2] = 1; W[1*2+0, 2] = 1
    
    # M4: A11(B01-B10)
    U[1*2+1, 3] = 1
    V[0*2+1, 3] = 1; V[1*2+0, 3] = -1
    W[0*2+1, 3] = 1; W[1*2+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[1*2+0, 4] = 1; W[1*2+1, 4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[1*2+0, 5] = 1
    W[0*2+0, 5] = 1; W[0*2+1, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[0*2+1, 6] = 1; V[1*2+1, 6] = 1
    W[1*2+0, 6] = 1; W[1*2+1, 6] = 1
    
    return U, V, W


def _block_decomposition_5x5(deadline):
    """
    Construct a decomposition for 5x5x5 using block structure.
    5 = 2 + 3, so we can use Strassen for the 2x2 blocks and
    standard multiplication for other blocks.
    
    This gives a rank of 7*4 + 3*3*4 = 28 + 36 = 64 (rough estimate)
    Actually, let's be more careful.
    
    A is 5x5, split as:
    A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    
    Similarly B = [B11 B12; B21 B22]
    
    C = A*B = [A11*B11 + A12*B21,  A11*B12 + A12*B22;
               A21*B11 + A22*B21,  A21*B12 + A22*B22]
    
    Each block product can be computed with rank equal to the product of dimensions
    using schoolbook, or less using Strassen for square blocks.
    
    Let's use Strassen for the 2x2*2x2 product (rank 7 instead of 8).
    
    Block products needed:
    1. A11*B11: 2x2 * 2x2 -> rank 7 (Strassen)
    2. A12*B21: 2x3 * 3x2 -> rank 12 (schoolbook)
    3. A11*B12: 2x2 * 2x3 -> rank 12 (schoolbook)
    4. A12*B22: 2x3 * 3x3 -> rank 18 (schoolbook)
    5. A21*B11: 3x2 * 2x2 -> rank 12 (schoolbook)
    6. A22*B21: 3x3 * 3x2 -> rank 18 (schoolbook)
    7. A21*B12: 3x2 * 2x3 -> rank 18 (schoolbook)
    8. A22*B22: 3x3 * 3x3 -> rank 27 (schoolbook)
    
    Total rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    That's barely better than 125. Let's try to use Strassen for the 3x3 blocks too.
    
    Actually, for 3x3, the best known is 23 (Laderman), not 27.
    But implementing Laderman is complex. Let's just use schoolbook for now.
    
    Alternatively, let's try a different partition: 5 = 1 + 4 or other splits.
    
    Actually, let's just return the schoolbook for now and rely on the search.
    """
    return None


def _structured_search(n, m, p, max_rank, deadline):
    """
    Search for structured decompositions using algebraic patterns.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = None
    best_rank = max_rank + 1
    
    # Try ranks from known lower bounds upward
    for target_rank in range(49, min(max_rank + 1, 126)):
        if time.time() > deadline:
            break
        
        # Try a few random structured initializations
        found = False
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Initialize with a mix of structured and random
            U = np.zeros((nm, target_rank), dtype=float)
            V = np.zeros((mp, target_rank), dtype=float)
            W = np.zeros((pn, target_rank), dtype=float)
            
            # Start from schoolbook and try to merge columns
            for c in range(target_rank):
                i = (c * 7 + attempt * 3) % n
                j = (c * 13 + attempt * 5) % m
                k = (c * 17 + attempt * 7) % p
                
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                
                if attempt > 0 and c < target_rank - 1:
                    if (c + attempt) % 3 == 0:
                        U[i * m + j, c] += 0.5
                        U[(i+1) % n * m + j, c] -= 0.5
                    if (c + attempt) % 5 == 0:
                        V[j * p + k, c] += 0.5
                        V[j * p + (k+1) % p, c] -= 0.5
            
            if _verify_float(n, m, p, U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                U_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(100) for x in row] for row in U.tolist()]
                V_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(100) for x in row] for row in V.tolist()]
                W_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 else Fraction(x).limit_denominator(100) for x in row] for row in W.tolist()]
                
                if _verify_exact(n, m, p, U_frac, V_frac, W_frac):
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result, best_rank


def _try_strassen_like(n, m, p, deadline):
    """
    Try a Strassen-like decomposition for 5x5x5.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for target_rank in range(49, 70):
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.zeros((nm, target_rank), dtype=float)
            V = np.zeros((mp, target_rank), dtype=float)
            W = np.zeros((pn, target_rank), dtype=float)
            
            assignments = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        c = idx % target_rank
                        assignments[(i, j, k)] = c
            
            col_triples = {}
            for (i, j, k), c in assignments.items():
                if c not in col_triples:
                    col_triples[c] = []
                col_triples[c].append((i, j, k))
            
            for c, triples in col_triples.items():
                weight = 1.0 / len(triples)
                for (i, j, k) in triples:
                    U[i * m + j, c] += weight
                    V[j * p + k, c] += weight
                    W[k * n + i, c] += weight
            
            pass
    
    return None, 126


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    result, rank = _structured_search(n, m, p, best_rank - 1, deadline)
    if result is not None and rank < best_rank:
        best = result
        best_rank = rank
    
    if time.time() < deadline:
        result2, rank2 = _try_strassen_like(n, m, p, deadline)
        if result2 is not None and rank2 < best_rank:
            best = result2
            best_rank = rank2
    
    return best
# EVOLVE-BLOCK-END
