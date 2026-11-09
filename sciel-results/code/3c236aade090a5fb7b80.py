"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _build_naive(n, m, p):
    """Build a naive rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def _build_rank28():
    """Build a rank-28 Strassen-based decomposition for 2x4x4 matmul.
    
    Uses 4 independent 2x2x2 Strassen blocks (for j-blocks {0,1},{2,3} and k-blocks {0,1},{2,3}).
    Each 2x2x2 Strassen block uses 7 multiplications, giving 4*7=28 total.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen 7 products for 2x2 matmul:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1=(a+d)(e+h), P2=(c+d)(e), P3=a(f-h), P4=d(g-e), P5=(a+b)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
    # C[0,0]=P1-P4+P5-P7, C[0,1]=P3+P5, C[1,0]=P2+P4, C[1,1]=P1-P2+P3+P6
    # 
    # In our indexing for a 2x2 block:
    # A local: a=A[0,0]=idx0, b=A[0,1]=idx1, c=A[1,0]=idx2, d=A[1,1]=idx3
    # B local: e=B[0,0]=idx0, f=B[0,1]=idx1, g=B[1,0]=idx2, h=B[1,1]=idx3
    # C local: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
    #
    # U terms (coefficients on A entries):
    # P1: a+d -> {(0,1),(3,1)}
    # P2: c+d -> {(2,1),(3,1)}
    # P3: a   -> {(0,1)}
    # P4: d   -> {(3,1)}
    # P5: a+b -> {(0,1),(1,1)}
    # P6: c-a -> {(2,1),(0,-1)}
    # P7: b-d -> {(1,1),(3,-1)}
    #
    # V terms (coefficients on B entries):
    # P1: e+h -> {(0,1),(3,1)}
    # P2: e   -> {(0,1)}
    # P3: f-h -> {(1,1),(3,-1)}
    # P4: g-e -> {(2,1),(0,-1)}
    # P5: h   -> {(3,1)}
    # P6: e+f -> {(0,1),(1,1)}
    # P7: g+h -> {(2,1),(3,1)}
    #
    # W terms (coefficients on C entries, with signs):
    # C[0,0]=P1-P4+P5-P7: P1:+1, P4:-1, P5:+1, P7:-1
    # C[0,1]=P3+P5:       P3:+1, P5:+1
    # C[1,0]=P2+P4:       P2:+1, P4:+1
    # C[1,1]=P1-P2+P3+P6: P1:+1, P2:-1, P3:+1, P6:+1
    
    S = [
        # (U_terms, V_terms, W_terms) for each Strassen product
        # P1
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),
        # P2
        ({(2,1),(3,1)}, {(0,1)}, {(2,1),(3,-1)}),
        # P3
        ({(0,1)}, {(1,1),(3,-1)}, {(1,1),(3,1)}),
        # P4
        ({(3,1)}, {(2,1),(0,-1)}, {(0,-1),(2,1)}),
        # P5
        ({(0,1),(1,1)}, {(3,1)}, {(0,1),(1,1)}),
        # P6
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),
        # P7
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,-1)}),
    ]
    
    # 4 block combinations of (j_block, k_block) each of size 2
    j_offsets = [0, 2]  # j in {0,1} or {2,3}
    k_offsets = [0, 2]  # k in {0,1} or {2,3}
    
    col = 0
    for j_off in j_offsets:
        for k_off in k_offsets:
            for u_terms, v_terms, w_terms in S:
                for (li, coeff) in u_terms:
                    i, j = li // 2, li % 2
                    gi, gj = i, j + j_off
                    U[gi * m + gj][col] += coeff
                
                for (li, coeff) in v_terms:
                    j, k = li // 2, li % 2
                    gj, gk = j + j_off, k + k_off
                    V[gj * p + gk][col] += coeff
                
                for (li, coeff) in w_terms:
                    k, i = li // 2, li % 2
                    gk, gi = k + k_off, i
                    W[gk * n + gi][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(U, V, W, n, m, p, max_rank, time_limit):
    """Try to find a lower rank decomposition by checking if columns are linearly dependent."""
    deadline = time.time() + time_limit
    rank = len(U[0])
    
    # Try to find if any subset of columns can be removed or if there's a lower rank representation
    # Use the fact that the tensor is fixed; check if we can express some columns as combinations of others
    
    # First, let's check the current rank validity
    current_rank = rank
    
    # Try removing one column at a time and see if the decomposition still works
    # This is a greedy approach
    for attempt in range(current_rank):
        if time.time() > deadline:
            break
        
        # Try removing column `attempt`
        # Check if the remaining columns still span the same tensor
        # We need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
        # If we remove column r0, we need the other columns to still sum to 1
        
        # This is complex; instead, let's try a different approach
        # Check if any two columns are proportional (which would indicate redundancy)
        pass
    
    return None

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the rank-28 Strassen-based decomposition
    result = _build_rank28()
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: naive decomposition
        result = _build_naive(n, m, p)
    
    # Try to find a lower rank decomposition via structured search
    elapsed = time.time() - start
    if elapsed < 120:
        # The rank-28 solution is our best known valid decomposition
        # A proper optimization would require solving a complex tensor decomposition problem
        pass
    
    return result
# EVOLVE-BLOCK-END
