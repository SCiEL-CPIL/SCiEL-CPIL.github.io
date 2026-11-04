"""Matrix multiplication 3x3x5 rank-minimal exact tensor decomposition search."""
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations

def _build_tensor(n, m, p):
    """Return dict T[(a,b,c)] = 1 for the multiplication tensor.
    a = i*m + j, b = j*p + k, c = k*n + i.
    """
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify(U, V, W, T, n, m, p):
    """Check that sum_c U[a,c]*V[b,c]*W[c] == T[(a,b,c)] for all a,b,c."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = Fraction(0)
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                expected = T.get((a, b, c), 0)
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _to_output(U, V, W):
    """Convert Fraction matrices to int/half-int lists for output."""
    def conv(M):
        out = []
        for row in M:
            r = []
            for x in row:
                if x.denominator == 1:
                    r.append(int(x.numerator))
                elif x.denominator == 2:
                    r.append(Fraction(x.numerator, 2))
                else:
                    r.append(x)
            out.append(r)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}

def _try_merge_two_terms(n, m, p, term1, term2, U, V, W):
    """
    Try to merge two schoolbook terms into one by finding a rank-1 tensor
    that equals the sum of the two original rank-1 tensors, OR by adjusting
    coefficients. Returns modified (U,V,W) with one fewer column, or None.
    """
    i1, j1, k1 = term1
    i2, j2, k2 = term2
    
    # The two terms contribute to positions:
    # Term1: a1=i1*m+j1, b1=j1*p+k1, c1=k1*n+i1
    # Term2: a2=i2*m+j2, b2=j2*p+k2, c2=k2*n+i2
    
    a1, b1, c1 = i1*m+j1, j1*p+k1, k1*n+i1
    a2, b2, c2 = i2*m+j2, j2*p+k2, k2*n+i2
    
    # For a simple merge, we need a single rank-1 tensor u*v*w that
    # reproduces both (a1,b1,c1) and (a2,b2,c2) contributions.
    # This requires u[a1]*v[b1]*w[c1] = 1 and u[a2]*v[b2]*w[c2] = 1,
    # and all other positions = 0.
    
    # This is only possible if the two tensors share enough structure.
    # Let's try: set u, v, w to have 1s at the required positions.
    
    # Check if we can find u, v, w in {0, 1, -1, 1/2, -1/2} that work.
    # The support of u*v*w must be contained in {(a1,b1,c1), (a2,b2,c2)}.
    
    # u must be nonzero only at positions that, when combined with v and w,
    # give only the two target positions.
    
    # Try: u has 1 at a1 and a2 (if a1 != a2), or 1 at a1 (if a1 == a2)
    # Similarly for v and w.
    
    # Case analysis based on index equalities:
    if a1 == a2 and b1 == b2 and c1 == c2:
        # Same position - shouldn't happen for distinct terms
        return None
    
    # Try to construct u, v, w
    # u[a]*v[b]*w[c] should be 1 at (a1,b1,c1) and (a2,b2,c2), 0 elsewhere.
    
    # Let's try small values: u in {0, 1, -1, 2, -2, 1/2, -1/2}
    # For efficiency, only try if the index patterns allow it.
    
    # A rank-1 tensor u*v*w has support that is a "grid" - the set of
    # positions (a,b,c) where u[a]!=0, v[b]!=0, w[c]!=0.
    # For the support to be exactly {(a1,b1,c1), (a2,b2,c2)}, we need:
    # - If a1 != a2: u must be nonzero at both a1 and a2
    # - If b1 != b2: v must be nonzero at both b1 and b2  
    # - If c1 != c2: w must be nonzero at both c1 and c2
    # - No other combinations of (a,b,c) from the supports should give nonzero product.
    
    # This is very restrictive. Let's check if it's feasible.
    u_support = {a1} if a1 == a2 else {a1, a2}
    v_support = {b1} if b1 == b2 else {b1, b2}
    w_support = {c1} if c1 == c2 else {c1, c2}
    
    # Check all combinations
    for a in u_support:
        for b in v_support:
            for c in w_support:
                if (a, b, c) != (a1, b1, c1) and (a, b, c) != (a2, b2, c2):
                    return None  # Would create extra nonzero entries
    
    # Now try to find actual values
    # We need u[a1]*v[b1]*w[c1] = 1 and u[a2]*v[b2]*w[c2] = 1
    # With the constraint that no other (a,b,c) in the support grid gives nonzero.
    
    # Try small rational values
    candidates = [Fraction(1), Fraction(-1), Fraction(2), Fraction(-2), 
                  Fraction(1,2), Fraction(-1,2)]
    
    for u_val1 in candidates:
        for u_val2 in candidates:
            for v_val1 in candidates:
                for v_val2 in candidates:
                    # Compute required w values
                    # u[a1]*v[b1]*w[c1] = 1 => w[c1] = 1/(u[a1]*v[b1])
                    # u[a2]*v[b2]*w[c2] = 1 => w[c2] = 1/(u[a2]*v[b2])
                    
                    try:
                        w_c1 = Fraction(1) / (u_val1 * v_val1)
                        w_c2 = Fraction(1) / (u_val2 * v_val2)
                    except ZeroDivisionError:
                        continue
                    
                    if w_c1 not in candidates and w_c2 not in candidates:
                        continue
                    
                    # Build u, v, w vectors
                    U_new = [row[:] for row in U]
                    V_new = [row[:] for row in V]
                    W_new = [row[:] for row in W]
                    
                    # Find the column indices for term1 and term2
                    col1 = None
                    col2 = None
                    for col in range(len(U[0])):
                        if U[a1][col] == 1 and V[b1][col] == 1 and W[c1][col] == 1:
                            if col1 is None:
                                col1 = col
                        if U[a2][col] == 1 and V[b2][col] == 1 and W[c2][col] == 1:
                            if col2 is None:
                                col2 = col
                    
                    if col1 is None or col2 is None or col1 == col2:
                        continue
                    
                    # Remove col2, modify col1
                    # Set col1 to the new merged term
                    for a in range(n*m):
                        U_new[a][col1] = Fraction(0)
                    for b in range(m*p):
                        V_new[b][col1] = Fraction(0)
                    for c in range(p*n):
                        W_new[c][col1] = Fraction(0)
                    
                    # Set values for merged term
                    for a in u_support:
                        if a == a1 and a == a2:
                            U_new[a][col1] = u_val1  # same position
                        elif a == a1:
                            U_new[a][col1] = u_val1
                        elif a == a2:
                            U_new[a][col1] = u_val2
                    
                    for b in v_support:
                        if b == b1 and b == b2:
                            V_new[b][col1] = v_val1
                        elif b == b1:
                            V_new[b][col1] = v_val1
                        elif b == b2:
                            V_new[b][col1] = v_val2
                    
                    for c in w_support:
                        if c == c1 and c == c2:
                            W_new[c][col1] = w_c1
                        elif c == c1:
                            W_new[c][col1] = w_c1
                        elif c == c2:
                            W_new[c][col1] = w_c2
                    
                    # Zero out col2
                    for a in range(n*m):
                        U_new[a][col2] = Fraction(0)
                    for b in range(m*p):
                        V_new[b][col2] = Fraction(0)
                    for c in range(p*n):
                        W_new[c][col2] = Fraction(0)
                    
                    return U_new, V_new, W_new, col2
    
    return None

def _try_strassen_like(n, m, p):
    """
    Try to use a Strassen-like decomposition for the 3x3x3 core
    and handle the extra 2 columns of the 5-column matrix.
    """
    # For 3x3x3, rank 23 is known (Strassen).
    # For 3x3x5, we can split the 5 columns into 3+2.
    # The 3x3x3 part can use rank 23.
    # The 3x3x2 part needs rank 6 (schoolbook) or less.
    # But the indices overlap, so we need to be careful.
    
    # Actually, let's try a different approach:
    # Use the known 3x3x3 rank-23 decomposition for the first 3 output columns,
    # and add 6 more terms for the last 2 output columns.
    # Total: 23 + 6 = 29? No, that's not right either.
    
    # Let me think more carefully. The matrix multiplication C = A*B where
    # A is 3x3, B is 3x5, C is 3x5.
    # C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1,2,3,4.
    
    # We can write this as:
    # C[:,0:3] = A * B[:,0:3]  (3x3 * 3x3 = 3x3)
    # C[:,3:5] = A * B[:,3:5]  (3x3 * 3x2 = 3x2)
    
    # For the 3x3x3 part, use Strassen rank 23.
    # For the 3x3x2 part, use schoolbook rank 6.
    # Total rank: 23 + 6 = 29.
    
    # But wait, the U and V matrices need to be consistent.
    # U has shape (9, rank), V has shape (15, rank), W has shape (15, rank).
    
    # For the 3x3x3 part (k=0,1,2):
    # U[a, col] for a = i*3+j, col in Strassen terms
    # V[b, col] for b = j*5+k, col in Strassen terms  
    # W[c, col] for c = k*3+i, col in Strassen terms
    
    # For the 3x3x2 part (k=3,4):
    # Additional terms for k=3,4
    
    # This is complex. Let me try a simpler approach:
    # Just try to find any rank-44 decomposition by checking
    # if any single schoolbook term is redundant.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    T = _build_tensor(n, m, p)
    rank0 = n * m * p

    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = rank0

    # Strategy 1: Try to merge pairs of terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    # Only try pairs that share at least one index dimension
    # to make merging more likely
    candidate_pairs = []
    for idx1, t1 in enumerate(terms):
        for idx2 in range(idx1 + 1, len(terms)):
            t2 = terms[idx2]
            # Check if they share an index
            if t1[0] == t2[0] or t1[1] == t2[1] or t1[2] == t2[2]:
                candidate_pairs.append((t1, t2))
    
    random.seed(42)
    random.shuffle(candidate_pairs)
    
    for t1, t2 in candidate_pairs:
        if time.time() > deadline:
            break
        result = _try_merge_two_terms(n, m, p, t1, t2, best_U, best_V, best_W)
        if result is not None:
            U_new, V_new, W_new, col_to_remove = result
            # Remove the zero column
            U_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_new]
            V_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_new]
            W_final = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_new]
            
            if _verify(U_final, V_final, W_final, T, n, m, p):
                best_U, best_V, best_W = U_final, V_final, W_final
                best_rank = best_rank - 1
                print(f"Found rank {best_rank} decomposition!")
                # Continue searching for more reductions
                U, V, W = U_final, V_final, W_final
                # Rebuild terms for the reduced decomposition is complex;
                # break after first success for simplicity
                break
    
    return _to_output(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
