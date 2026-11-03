"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Try rank 16 first, fall back to rank 17
    for target_rank in [16, 17]:
        result = _try_rank(target_rank, deadline, start_time)
        if result is not None:
            return result

    # Fallback: return rank 17 solution
    return _rank17_solution()

def _rank17_solution():
    n, m, p = DIMENSIONS
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen part for 2x2 block (indices 0-6)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for remaining terms (indices 7-16)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, rank, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    
    # Target: T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    target_terms = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_terms.add((u_idx, v_idx, w_idx))
    
    # Check all target terms sum to 1
    for (u_idx, v_idx, w_idx) in target_terms:
        s = 0
        for r in range(rank):
            s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
        if s != 1:
            return False
    
    # Check a sample of non-target terms sum to 0
    import random
    random.seed(42)
    for _ in range(50):
        u_idx = random.randint(0, dim_u - 1)
        v_idx = random.randint(0, dim_v - 1)
        w_idx = random.randint(0, dim_w - 1)
        if (u_idx, v_idx, w_idx) in target_terms:
            continue
        s = 0
        for r in range(rank):
            s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
        if s != 0:
            return False
    
    return True

def _try_rank(rank, deadline, start_time):
    """Try to find a rank-rank decomposition. Returns dict or None."""
    n, m, p = DIMENSIONS
    
    if rank >= 17:
        sol = _rank17_solution()
        return sol
    
    # For rank 16, we need to find if there's a valid decomposition
    # This is a complex combinatorial search. Let's try a structured approach.
    # 
    # The 10 schoolbook terms in the rank-17 solution are:
    # 7: A02*B20 -> C00
    # 8: A02*B21 -> C01
    # 9: A12*B20 -> C10
    # 10: A12*B21 -> C11
    # 11: A00*B02 -> C02
    # 12: A01*B12 -> C02
    # 13: A10*B02 -> C12
    # 14: A11*B12 -> C12
    # 15: A02*B22 -> C02
    # 16: A12*B22 -> C12
    #
    # To get rank 16, we need to combine two of these into one rank-1 term
    # or find a Strassen-like identity that eliminates one more term.
    #
    # One approach: try to use a linear combination of two schoolbook terms
    # that can be expressed as a single rank-1 term.
    #
    # For example, can we combine A02*B20 -> C00 and A12*B20 -> C10?
    # These share B20. If we could write (A02 + c*A12)*B20 contributing to C00 and C10,
    # but that would require the W factor to have entries in both C00 and C10 positions,
    # which is fine, but the U factor would need to be A02 + c*A12, which is not a single
    # matrix entry - it's a linear combination, which is allowed in tensor decomposition.
    #
    # Actually, in tensor decomposition, U, V, W can have arbitrary linear combinations.
    # So let's try: U = [A02, A12] (as a vector in the A-space), V = [B20, 0], 
    # but that doesn't work because V would need to be a single element.
    #
    # The correct approach: a rank-1 term is U[u,r]*V[v,r]*W[w,r] for all u,v,w.
    # For it to contribute to A02*B20->C00 AND A12*B20->C10, we'd need:
    #   U[2,r]*V[6,r]*W[0,r] = 1  (for A02*B20->C00)
    #   U[5,r]*V[6,r]*W[1,r] = 1  (for A12*B20->C10)
    #   And zero elsewhere.
    # This requires U[2,r], U[5,r] both nonzero, V[6,r] nonzero, W[0,r], W[1,r] nonzero.
    # That's valid! Let's try this.
    #
    # Similarly for other pairs that share a B element.
    
    # Let's try to construct rank 16 by merging pairs of schoolbook terms that share a B column
    # or share an A row.
    
    # Pairs sharing B20: (A02*B20->C00, A12*B20->C10)
    # Pairs sharing B21: (A02*B21->C01, A12*B21->C11)
    # Pairs sharing B22: (A02*B22->C02, A12*B22->C12)
    # Pairs sharing B02: (A00*B02->C02, A10*B02->C12)
    # Pairs sharing B12: (A01*B12->C02, A11*B12->C12)
    
    # Let's try merging the B20 pair and see if the rest still works.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen part (indices 0-6) - same as before
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Now for the 10 schoolbook terms, try to merge into 9 terms
    # Strategy: merge two pairs and keep 6 individual, total = 2 + 6 = 8... no, 7+8=15? 
    # Wait, 7 Strassen + 9 merged schoolbook = 16. Let's try merging one pair.
    
    # Merge A02*B20->C00 and A12*B20->C10 into one term (index 7)
    # U[2,7]=a, U[5,7]=b, V[6,7]=c, W[0,7]=d, W[1,7]=e
    # Need: a*c*d = 1, b*c*e = 1, and no other contributions
    # Let a=1, c=1, d=1, b=1, e=1
    U[2, 7] = 1; U[5, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1; W[1, 7] = 1
    
    # But wait - this term also contributes to other (u,v,w) positions!
    # U[2,7]*V[6,7]*W[1,7] contributes to (2,6,1) which is A02*B20->C10? No.
    # (u=2, v=6, w=1): u=2 means i=0,j=2; v=6 means j=2,k=0; w=1 means k=0,n=2... 
    # w=1: k*n+i = k*2+i = 1. If k=0, i=1. But u=2 means i=0. Contradiction, so this is a zero term in target.
    # So U[2,7]*V[6,7]*W[1,7] = 1*1*1 = 1 at position (2,6,1) which is NOT a target term. This is bad!
    
    # The merged term creates spurious non-zero entries. We need the W vector to be zero 
    # at all positions except the two we want. But W[0,7]=1 and W[1,7]=1 means it's nonzero 
    # at w=0 and w=1. For u=2, v=6: w=0 gives (2,6,0) which IS target (A02*B20->C00). 
    # w=1 gives (2,6,1) which is NOT target. So this doesn't work directly.
    
    # We need a different approach. Let's try to use the freedom in the Strassen part
    # to absorb some schoolbook terms.
    
    # Actually, let me try a completely different rank-16 construction.
    # I'll use a more sophisticated Strassen-like approach.
    
    # For now, if we can't easily find rank 16, return None to use rank 17
    return None

# EVOLVE-BLOCK-END
