"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

DEADLINE_SECONDS = 115  # Safety margin within 120s evaluation limit

def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) for the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # The tensor T[i, j, k, i', j', k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # We need: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    for i in range(n):
        for j in range(m):
            for i_prime in range(n):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            a = i * m + j
                            b = j_prime * p + k_prime
                            c = k_prime * n + i_prime
                            expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                            actual = np.dot(U[a], V[b] * W[c])
                            if abs(actual - expected) > tol:
                                return False
    return True

def _try_poly_decomposition(n, m, p, max_time):
    """
    Attempt to find a decomposition using the polynomial ring Z[x]/(x^p) structure.
    This exploits the fact that multiplication by x is a cyclic shift (or truncated shift)
    and can potentially reduce rank compared to schoolbook.
    
    For 2x2 x 2xp, we can view B as two polynomials b0(x), b1(x) of degree < p.
    A is a 2x2 matrix.
    C = A * B gives two polynomials c0(x), c1(x).
    
    c0(x) = a00*b0(x) + a01*b1(x)
    c1(x) = a10*b0(x) + a11*b1(x)
    
    This is a bilinear map. We can try to decompose the polynomial multiplication part.
    Multiplication of two polynomials of degree < p in Z[x]/(x^p) has a known rank.
    However, here we are just multiplying a scalar (from A) by a polynomial (from B).
    So each term a_ij * b_j(x) is just scalar multiplication.
    
    Wait, the tensor decomposition is for the whole multiplication.
    T[i, j, k, i', j', k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    Let's try a structured search based on the polynomial idea.
    We look for U, V, W with entries in {0, 1, -1} or small integers/halves.
    """
    start_time = time.time()
    
    # A simpler structured approach:
    # Since n=2, m=2, the input to U is 4 vectors, to V is 26 vectors.
    # The output is 26 vectors.
    
    # Let's try a random search with a bias towards structured matrices.
    # We will generate random matrices with entries from {-1, 0, 1} and try to solve for the third.
    # This is hard.
    
    # Alternative: Use the known lower bound and try to construct a specific rank.
    # For 2x2 x 2xp, the rank is at least 3p - 1? Or something similar.
    # Schoolbook is 4p = 52.
    # If we can get 51, that's an improvement.
    
    # Let's try to find a decomposition of rank 51.
    # We can try to "merge" two schoolbook terms if they don't interfere.
    
    # The interference condition for merging terms (i1, j1, k1) and (i2, j2, k2) is:
    # They must not share any index in a way that creates cross-terms.
    # Specifically, if we merge two columns r1 and r2 into one column r, we need:
    # U[i*m+j, r] = U1[i*m+j, r1] + U2[i*m+j, r2]
    # V[j*p+k, r] = V1[j*p+k, r1] + V2[j*p+k, r2]
    # W[k*n+i, r] = W1[k*n+i, r1] + W2[k*n+i, r2]
    # This works if the supports of the terms are disjoint or orthogonal.
    # In schoolbook, each term has support only at (i, j, k).
    # If we merge (i1, j1, k1) and (i2, j2, k2), the new term will have non-zero values at
    # (i1, j1, k1) AND (i2, j2, k2).
    # When we sum over r, we get:
    # sum_r U[a,r] V[b,r] W[c,r]
    # If we merge two terms, the product U*V*W will be non-zero at 4 combinations of indices:
    # (i1, j1, k1), (i1, j2, k2), (i2, j1, k1), (i2, j2, k2).
    # We need the cross terms (i1, j2, k2) and (i2, j1, k1) to be zero.
    # This happens if i1 != i2 AND j1 != j2 AND k1 != k2? No.
    # The cross term for (i1, j2, k2) is U[i1*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i1, r].
    # In the merged column, U[i1*m+j2, r] = U1[i1*m+j2, r1] + U2[i1*m+j2, r2].
    # Since U1 is non-zero only at i1, j1, U1[i1*m+j2, r1] is 0 if j2 != j1.
    # Similarly U2 is non-zero only at i2, j2.
    # So if i1 != i2 and j1 != j2, the cross terms might vanish?
    # Let's check:
    # U[i1*m+j2, r] = 0 + 0 = 0 (if i1!=i2 and j2!=j1) -> Wait.
    # U1[i1*m+j2, r1] is non-zero only if i1==i1 and j2==j1. So if j2!=j1, it is 0.
    # U2[i1*m+j2, r2] is non-zero only if i1==i2 and j2==j2. So if i1!=i2, it is 0.
    # So yes, if i1!=i2 and j1!=j2, the U component at (i1, j2) is 0.
    # Similarly, we need to check V and W.
    # V[j2*p+k2, r] = V1[j2*p+k2, r1] + V2[j2*p+k2, r2].
    # V1 is non-zero at j1, k1. So if j2!=j1 or k2!=k1, it is 0.
    # V2 is non-zero at j2, k2. So V2[j2*p+k2, r2] = 1.
    # So V[j2*p+k2, r] = 1.
    # W[k2*n+i1, r] = W1[k2*n+i1, r1] + W2[k2*n+i1, r2].
    # W1 is non-zero at k1, i1. So if k2!=k1, it is 0.
    # W2 is non-zero at k2, i2. So if i1!=i2, it is 0.
    # So W[k2*n+i1, r] = 0.
    # Thus the product U*V*W at (i1, j2, k2) is 0 * 1 * 0 = 0.
    
    # So the condition to merge two schoolbook terms (i1, j1, k1) and (i2, j2, k2) is:
    # i1 != i2 AND j1 != j2 AND k1 != k2.
    # If this holds, we can merge them into a single rank term.
    
    # Let's find the maximum matching of such triples.
    # We have 4p triples. We want to pair them up such that i1!=i2, j1!=j2, k1!=k2.
    # This is a matching problem in a graph where edges connect compatible triples.
    
    # Let's try to find a matching of size 1 to reduce rank by 1.
    # If we can find 2 disjoint compatible pairs, we reduce rank by 2.
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Graph of compatibility
    compatible = {}
    for idx1, t1 in enumerate(triples):
        compatible[idx1] = []
        for idx2, t2 in enumerate(triples):
            if idx1 < idx2:
                if t1[0] != t2[0] and t1[1] != t2[1] and t1[2] != t2[2]:
                    compatible[idx1].append(idx2)
    
    # Find maximum matching
    # Since n=2, m=2, p=13, the number of triples is 52.
    # We can use a simple greedy or Hopcroft-Karp.
    # Let's use a simple DFS-based maximum bipartite matching if we split the graph?
    # No, it's a general graph. But the graph is sparse.
    # Let's just try to find one pair first.
    
    # Actually, we can merge more than 2 if they are all mutually compatible.
    # But let's start with pairs.
    
    # Find one pair
    pair_found = None
    for idx1 in range(len(triples)):
        for idx2 in compatible[idx1]:
            pair_found = (idx1, idx2)
            break
        if pair_found:
            break
    
    if pair_found:
        # Construct the new decomposition
        # Start with schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        rank = U.shape[1]
        
        # Merge pair_found[0] and pair_found[1]
        col1 = pair_found[0]
        col2 = pair_found[1]
        
        # Add col2 to col1
        U[:, col1] += U[:, col2]
        V[:, col1] += V[:, col2]
        W[:, col1] += W[:, col2]
        
        # Remove col2
        U = np.delete(U, col2, axis=1)
        V = np.delete(V, col2, axis=1)
        W = np.delete(W, col2, axis=1)
        
        new_rank = U.shape[1]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W, new_rank
    
    # Try to find more pairs
    # This is a heuristic, so we might not find the optimal merging.
    # But even one merge is an improvement.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to improve
    result = _try_poly_decomposition(n, m, p, DEADLINE_SECONDS)
    if result:
        U, V, W, rank = result
        if rank < best_rank:
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    
    # Convert to lists as required by the interface
    rank = best_U.shape[1]
    U_list = [[float(best_U[i, j]) for j in range(rank)] for i in range(len(best_U))]
    V_list = [[float(best_V[i, j]) for j in range(rank)] for i in range(len(best_V))]
    W_list = [[float(best_W[i, j]) for j in range(rank)] for i in range(len(best_W))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
