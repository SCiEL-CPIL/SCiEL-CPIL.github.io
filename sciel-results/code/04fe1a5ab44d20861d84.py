"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a decomposition with rank < nm*mp*pn
    # Start with schoolbook and try to reduce
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # This is expensive, so do it efficiently
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Only check non-zero and zero positions
        # For efficiency, check a subset first
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Should be 1
                    val = 0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if val != 1:
                        return False
        # Check some zero positions
        checked = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if T[a, b, c] == 0:
                        val = 0
                        for r in range(rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if val != 0:
                            return False
                        checked += 1
                        if checked > 1000:  # Check a subset for speed
                            break
                if checked > 1000:
                    break
            if checked > 1000:
                break
        return True
    
    def verify_full(U, V, W):
        """Full verification - expensive but complete."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=int)
        for r in range(rank):
            # Outer product of U[:,r], V[:,r], W[:,r]
            for a in range(nm):
                if U[a, r] != 0:
                    for b in range(mp):
                        if V[b, r] != 0:
                            for c in range(pn):
                                if W[c, r] != 0:
                                    T_recon[a, b, c] += U[a, r] * V[b, r] * W[c, r]
        return np.array_equal(T_recon, T)
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=int)
    V_school = np.zeros((mp, rank_school), dtype=int)
    W_school = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank by merging terms
    # For each pair of terms, try to merge them into one
    # This is a heuristic approach
    
    # Alternative: try specific structured decompositions
    # For now, return schoolbook as baseline
    # TODO: implement actual search
    
    # Convert to list of lists for output
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
