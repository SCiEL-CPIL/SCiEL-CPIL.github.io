"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_eliminate_terms(U, V, W, n, m, p, max_attempts=500):
    """
    Try to eliminate terms by checking if removing a term and adjusting others
    maintains the decomposition. Uses the fact that n=2 gives structure.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    attempts = 0
    while attempts < max_attempts and _time_left() > 20:
        attempts += 1
        cur_rank = U.shape[1]
        if cur_rank <= best_rank:
            break
        
        # Try removing each term and see if we can compensate
        found = False
        for r_to_remove in range(cur_rank):
            if _time_left() < 20:
                break
            
            # Temporarily remove term r_to_remove
            U_test = np.delete(U, r_to_remove, axis=1)
            V_test = np.delete(V, r_to_remove, axis=1)
            W_test = np.delete(W, r_to_remove, axis=1)
            
            # Check if the remaining terms still form a valid decomposition
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-8):
                U, V, W = U_test, V_test, W_test
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if not found:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_different_partitions(n, m, p):
    """
    Try different block partitions to find lower-rank decompositions.
    """
    best_rank = 54
    best_U, best_V, best_W = None, None, None
    
    # Try different ways to partition j indices into groups of size 2 and 1
    # and k indices into groups of size 2
    j_indices = list(range(m))
    k_indices = list(range(p))
    
    # Generate all ways to partition j into pairs and singles
    def get_j_partitions(indices):
        """Generate all partitions of indices into pairs and singles."""
        if not indices:
            yield []
            return
        if len(indices) == 1:
            yield [(indices[0],)]
            return
        # Try making the first element a single
        for rest in get_j_partitions(indices[1:]):
            yield [(indices[0],)] + rest
        # Try pairing the first element with another
        for i in range(1, len(indices)):
            pair = (indices[0], indices[i])
            remaining = [indices[0]] if False else indices[1:i] + indices[i+1:]
            # Fix: remaining should exclude indices[0] and indices[i]
            remaining = [x for x in indices[1:] if x != indices[i]]
            for rest in get_j_partitions(remaining):
                yield [pair] + rest
    
    def get_k_partitions(indices):
        """Generate all partitions of indices into pairs and singles."""
        if not indices:
            yield []
            return
        if len(indices) == 1:
            yield [(indices[0],)]
            return
        for i in range(1, len(indices)):
            pair = (indices[0], indices[i])
            remaining = [x for x in indices[1:] if x != indices[i]]
            for rest in get_k_partitions(remaining):
                yield [pair] + rest
    
    # For efficiency, only try a few promising partitions
    # The current one: j=[(0,1),(2,3),(4,)], k=[(0,1),(2,3),(4,5)]
    # Try variations
    
    j_partitions = list(get_j_partitions(j_indices))
    k_partitions = list(get_k_partitions(k_indices))
    
    # Limit search
    max_j_parts = min(20, len(j_partitions))
    max_k_parts = min(20, len(k_partitions))
    
    for j_part in j_partitions[:max_j_parts]:
        for k_part in k_partitions[:max_k_parts]:
            if _time_left() < 30:
                break
            
            # Count rank for this partition
            rank = 0
            for jp in j_part:
                for kp in k_part:
                    if len(jp) == 2 and len(kp) == 2:
                        rank += 7  # Strassen
                    elif len(jp) == 1 and len(kp) == 1:
                        rank += 2  # Two separate products (one per row of A)
                    elif len(jp) == 2 and len(kp) == 1:
                        rank += 2  # Each j in pair needs its own product with single k
                    elif len(jp) == 1 and len(kp) == 2:
                        rank += 2  # Single j, two k's - can we do better?
                        # Actually for single j, we need A[i,j]*B[j,k0] and A[i,j]*B[j,k1]
                        # These share the A factor, so we might combine
                        # But in tensor decomposition, each is a separate term unless we can merge
                        rank += 2
                    
            if rank >= best_rank:
                continue
            
            # Build decomposition for this partition
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            
            valid = True
            for jp in j_part:
                for kp in k_part:
                    if len(jp) == 2 and len(kp) == 2:
                        j0, j1 = jp
                        k0, k1 = kp
                        # Standard Strassen 2x2x2
                        # P1
                        U[0*m + j0, col] += 1
                        U[1*m + j1, col] += 1
                        V[j0*p + k0, col] += 1
                        V[j1*p + k1, col] += 1
                        W[k0*n + 0, col] += 1
                        W[k1*n + 1, col] += 1
                        col += 1
                        # P2
                        U[1*m + j0, col] += 1
                        U[1*m + j1, col] += 1
                        V[j0*p + k0, col] += 1
                        W[k0*n + 1, col] += 1
                        W[k1*n + 1, col] -= 1
                        col += 1
                        # P3
                        U[0*m + j0, col] += 1
                        V[j0*p + k1, col] += 1
                        V[j1*p + k1, col] -= 1
                        W[k1*n + 0, col] += 1
                        W[k1*n + 1, col] += 1
                        col += 1
                        # P4
                        U[1*m + j1, col] += 1
                        V[j1*p + k0, col] += 1
                        V[j0*p + k0, col] -= 1
                        W[k0*n + 0, col] += 1
                        W[k0*n + 1, col] += 1
                        col += 1
                        # P5
                        U[0*m + j0, col] += 1
                        U[0*m + j1, col] += 1
                        V[j1*p + k1, col] += 1
                        W[k0*n + 0, col] -= 1
                        W[k1*n + 0, col] += 1
                        col += 1
                        # P6
                        U[1*m + j0, col] += 1
                        U[0*m + j0, col] -= 1
                        V[j0*p + k0, col] += 1
                        V[j0*p + k1, col] += 1
                        W[k1*n + 1, col] += 1
                        col += 1
                        # P7
                        U[0*m + j1, col] += 1
                        U[1*m + j1, col] -= 1
                        V[j1*p + k0, col] += 1
                        V[j1*p + k1, col] += 1
                        W[k0*n + 0, col] += 1
                        col += 1
                    elif len(jp) == 1 and len(kp) == 1:
                        j = jp[0]
                        k = kp[0]
                        for i in range(n):
                            U[i*m + j, col] += 1
                            V[j*p + k, col] += 1
                            W[k*n + i, col] += 1
                            col += 1
                    elif len(jp) == 2 and len(kp) == 1:
                        j0, j1 = jp
                        k = kp[0]
                        for j in [j0, j1]:
                            for i in range(n):
                                U[i*m + j, col] += 1
                                V[j*p + k, col] += 1
                                W[k*n + i, col] += 1
                                col += 1
                    elif len(jp) == 1 and len(kp) == 2:
                        j = jp[0]
                        k0, k1 = kp
                        for k in [k0, k1]:
                            for i in range(n):
                                U[i*m + j, col] += 1
                                V[j*p + k, col] += 1
                                W[k*n + i, col] += 1
                                col += 1
            
            if col != rank:
                continue
            
            if _verify_decomposition(U, V, W, n, m, p):
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    else:
        # Try to eliminate terms
        if _time_left() > 40:
            U2, V2, W2, rank2 = _try_eliminate_terms(U, V, W, n, m, p, max_attempts=200)
            if rank2 < best_rank:
                best_U, best_V, best_W = U2, V2, W2
                best_rank = rank2
        
        # Try different partitions
        if _time_left() > 30:
            U3, V3, W3, rank3 = _try_different_partitions(n, m, p)
            if U3 is not None and rank3 < best_rank:
                best_U, best_V, best_W = U3, V3, W3
                best_rank = rank3
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
