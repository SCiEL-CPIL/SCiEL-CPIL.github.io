"""Matrix multiplication 3x3x8 rank-minimal decomposition."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _elapsed(t0):
    return time.time() - t0

def _time_left(t0, limit):
    return limit - _elapsed(t0)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Check nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of zero entries
    # The zero entries are those (a,b,c) where there's no valid (i,j,k)
    # a = i*m+j, b = j*p+k, c = k*n+i
    # For a given (a,b), the valid c is determined: if a% m == b // p and b% p == c // n and c % n == a // m
    # Let's check all (a,b) pairs and verify the only nonzero c is the correct one
    for a in range(n*m):
        for b in range(m*p):
            # Determine if there's a valid (i,j,k)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            if j == j2:
                # Valid entry at c = k*n + i
                c_valid = k*n + i
                val = np.sum(U[a, :] * V[b, :] * W[c_valid, :])
                if abs(val - 1.0) > tol:
                    return False
                # Check other c values are zero
                for c in range(p*n):
                    if c != c_valid:
                        val2 = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val2) > tol:
                            return False
            else:
                # All c should be zero
                for c in range(p*n):
                    val2 = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val2) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2(n, m, p):
    """
    Decompose 3x3 * 3x8 using Strassen on 2x2 blocks.
    
    Strategy: Split A (3x3) into [[A00, A01], [A10, A11]] where A00 is 2x2,
    A01 is 2x1, A10 is 1x2, A11 is 1x1.
    Split B (3x8) into [[B0], [B1]] where B0 is 2x8, B1 is 1x8.
    
    C = AB = [A00*B0 + A01*B1, A10*B0 + A11*B1]
    
    For A00*B0 (2x2 * 2x8): Split B0 into four 2x2 blocks, use Strassen on each.
    That's 4 * 7 = 28 terms.
    
    For A01*B1 (2x1 * 1x8): 16 scalar products.
    For A10*B0 (1x2 * 2x8): 16 scalar products.
    For A11*B1 (1x1 * 1x8): 8 scalar products.
    
    Total: 28 + 16 + 16 + 8 = 68.
    
    But we can do better by using a smarter decomposition of the 2x2*2x8 part.
    Actually, let me use a direct construction.
    """
    n, m, p = 3, 3, 8
    
    # We'll build the decomposition term by term.
    # Each term is (u, v, w) where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    terms = []
    
    # A indices: A[i,j] for i in 0,1,2; j in 0,1,2
    # B indices: B[j,k] for j in 0,1,2; k in 0,1,...,7
    # C indices: C[i,k] for i in 0,1,2; k in 0,1,...,7
    
    # Strassen for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-b)(e+f)
    # M7 = (d-a)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x8, we apply Strassen to each pair of columns.
    # B0 has rows 0,1 and columns 0-7.
    # Split into 4 blocks of 2 columns each.
    
    # Block b (b=0,1,2,3) corresponds to columns 2b, 2b+1 of B0.
    # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B0_block = [[B[0,2b], B[0,2b+1]], [B[1,2b], B[1,2b+1]]]
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for this 2x2 * 2x2 block
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0]) ... wait, Strassen uses both columns.
        
        # Actually, Strassen for 2x2*2x2 uses:
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        
        # M1 = (a+d)(e+h): u has A[0,0] and A[1,1], v has B[0,k0] and B[1,k1]
        # M2 = (c+d)e: u has A[1,0] and A[1,1], v has B[0,k0]
        # M3 = a(f-h): u has A[0,0], v has B[0,k1] and B[1,k1]
        # M4 = d(g-e): u has A[1,1], v has B[1,k0] and B[0,k0]
        # M5 = (a+b)h: u has A[0,0] and A[0,1], v has B[1,k1]
        # M6 = (c-b)(e+f): u has A[1,0] and A[0,1], v has B[0,k0] and B[0,k1]
        # M7 = (d-a)(g+h): u has A[1,1] and A[0,0], v has B[1,k0] and B[1,k1]
        
        # Output contributions:
        # C[0,k0] += M1 + M4 - M5 + M7
        # C[0,k1] += M3 + M5
        # C[1,k0] += M2 + M4
        # C[1,k1] += M1 - M2 + M3 + M6
        
        # M1: (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # u: A[0,0]=1, A[1,1]=1
        # v: B[0,k0]=1, B[1,k1]=1
        # w: C[0,k0]=1, C[1,k1]=1
        u1 = np.zeros(n*m); u1[0*3+0] = 1; u1[1*3+1] = 1
        v1 = np.zeros(m*p); v1[0*8+k0] = 1; v1[1*8+k1] = 1
        w1 = np.zeros(p*n); w1[k0*3+0] = 1; w1[k1*3+1] = 1
        terms.append((u1, v1, w1))
        
        # M2: (A[1,0]+A[1,1]) * B[0,k0]
        u2 = np.zeros(n*m); u2[1*3+0] = 1; u2[1*3+1] = 1
        v2 = np.zeros(m*p); v2[0*8+k0] = 1
        w2 = np.zeros(p*n); w2[k0*3+1] = 1
        terms.append((u2, v2, w2))
        
        # M3: A[0,0] * (B[0,k1]-B[1,k1])
        u3 = np.zeros(n*m); u3[0*3+0] = 1
        v3 = np.zeros(m*p); v3[0*8+k1] = 1; v3[1*8+k1] = -1
        w3 = np.zeros(p*n); w3[k1*3+0] = 1
        terms.append((u3, v3, w3))
        
        # M4: A[1,1] * (B[1,k0]-B[0,k0])
        u4 = np.zeros(n*m); u4[1*3+1] = 1
        v4 = np.zeros(m*p); v4[1*8+k0] = 1; v4[0*8+k0] = -1
        w4 = np.zeros(p*n); w4[k0*3+0] = 1; w4[k0*3+1] = 1
        terms.append((u4, v4, w4))
        
        # M5: (A[0,0]+A[0,1]) * B[1,k1]
        u5 = np.zeros(n*m); u5[0*3+0] = 1; u5[0*3+1] = 1
        v5 = np.zeros(m*p); v5[1*8+k1] = 1
        w5 = np.zeros(p*n); w5[k1*3+0] = 1; w5[k0*3+0] = -1
        terms.append((u5, v5, w5))
        
        # M6: (A[1,0]-A[0,1]) * (B[0,k0]+B[0,k1])
        u6 = np.zeros(n*m); u6[1*3+0] = 1; u6[0*3+1] = -1
        v6 = np.zeros(m*p); v6[0*8+k0] = 1; v6[0*8+k1] = 1
        w6 = np.zeros(p*n); w6[k1*3+1] = 1
        terms.append((u6, v6, w6))
        
        # M7: (A[1,1]-A[0,0]) * (B[1,k0]+B[1,k1])
        u7 = np.zeros(n*m); u7[1*3+1] = 1; u7[0*3+0] = -1
        v7 = np.zeros(m*p); v7[1*8+k0] = 1; v7[1*8+k1] = 1
        w7 = np.zeros(p*n); w7[k0*3+0] = 1; w7[k1*3+1] = 1
        terms.append((u7, v7, w7))
    
    # Now handle the remaining terms:
    # A[2,j] * B[j,k] for j=0,1,2 and all k=0..7: 24 terms
    # A[i,2] * B[2,k] for i=0,1 and all k=0..7: 16 terms
    # But wait, we need to be careful not to double-count.
    
    # The Strassen part computed:
    # C[0,k] for k=0..7 from A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] for k=0..7 from A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Remaining terms needed:
    # C[0,k] += A[0,2]*B[2,k] for all k
    # C[1,k] += A[1,2]*B[2,k] for all k
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] for all k
    
    # So we need:
    # A[0,2]*B[2,k] for k=0..7: 8 terms
    # A[1,2]*B[2,k] for k=0..7: 8 terms
    # A[2,0]*B[0,k] for k=0..7: 8 terms
    # A[2,1]*B[1,k] for k=0..7: 8 terms
    # A[2,2]*B[2,k] for k=0..7: 8 terms
    # Total: 40 terms
    
    # Total rank: 28 + 40 = 68
    
    for i, j, k in itertools.product([0, 1, 2], [0, 1, 2], range(8)):
        # Skip terms already handled by Strassen
        if i in [0, 1] and j in [0, 1]:
            continue
        u = np.zeros(n*m); u[i*3+j] = 1
        v = np.zeros(m*p); v[j*8+k] = 1
        w = np.zeros(p*n); w[k*3+i] = 1
        terms.append((u, v, w))
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    return U, V, W

def _try_improvements(U, V, W, n, m, p, t0, limit):
    """
    Try to improve the decomposition by finding linear dependencies or
    using algebraic tricks to reduce rank.
    
    One approach: check if any column of the combined matrix is redundant.
    Another: use the fact that entries can be half-integers to find
    a lower-rank representation.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to find a lower rank by checking if the decomposition can be
    # represented with fewer terms. This is equivalent to finding a lower-rank
    # tensor decomposition, which is generally hard.
    
    # One heuristic: try to combine terms that have overlapping support.
    # If two terms have the same support pattern (up to scaling), they can be merged.
    
    # For now, just return the current best.
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    t0 = time.time()
    limit = 140  # Leave 10s margin before the 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Start with Strassen-based construction (rank 68)
    U, V, W = _strassen_2x2x2(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Try to improve
    if _time_left(t0, limit) > 30:
        U, V, W, rank = _try_improvements(U, V, W, n, m, p, t0, limit)
    
    # Final check
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
