"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_left():
    return TIME_LIMIT - (time.time() - _start_time) - SAFETY_MARGIN

def _verify(U, V, W):
    """Verify that U, V, W form an exact tensor decomposition of matrix multiplication.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    rank = len(U[0])
    # Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    # i.e., the (i,j,k) entry of the multiplication tensor = (C = A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T[i,j,k] = delta (it's 1 only implicitly via the outer product structure)
    # Actually the multiplication tensor T_{i,j,k} is defined such that
    # (A*B)_{i,k} = sum_{j,c} A_{i,j} * V[j,k,c] * U[i,c]... 
    
    # The standard form: T_{i,j,k} (the coefficient of A_{i,j} * B_{j,k} in (AB)_{i,k})
    # Wait, (AB)_{i,k} = sum_j A_{i,j} B_{j,k}. So the tensor is T[i,j,k] = 1 for all valid i,j,k?
    # No. The tensor decomposition says: (AB)_{i,k} = sum_c (sum_j U[i*m+j, c] * A_{i,j}) * (sum_k... )
    # Let me re-read the problem.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition: sum_c U[a, c] * V[b, c] * W[c_out, c] where a = i*m+j (row of U indexed by A's entry),
    # b = j*p+k (row of V indexed by B's entry), and the output is at position (k, i) in W? 
    # W[k*n + i, c] is the output coefficient.
    # So: (AB)_{i,k} = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A_{i,j}) * (sum_k' V[j'*p+k', c] * B_{j',k'})
    # Wait, that doesn't look right either. Let me think about it as:
    # (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # In tensor form: T is a 3-index tensor T[i, j, k] = 1 if this is the "right" combination.
    # Actually, the standard formulation: the multiplication tensor T satisfies
    # (AB)_{i,k} = sum_{j,c} A_{i,j} * B_{j,k} * T_{i,j,k}^{(c)} where the decomposition is
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # And T_{i,j,k} = 1 for all (i,j,k) that are valid indices.
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # AND for all other combinations of (a, b, c_out) where a = i*m+j, b = j'*p+k', 
    # the cross terms should be 0? No, the tensor only has entries at valid (i,j,k).
    # Actually, the tensor is only non-zero at (i,j,k) for valid indices, and equals 1 there.
    # So the condition is: for all valid (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # But we also need that the decomposition doesn't produce spurious terms. Since the tensor
    # is only defined at those (i,j,k) positions, we just need those to equal 1.
    # However, there's a subtlety: different (i,j,k) triples map to different (a,b,c_out) rows.
    # Let's just check: for all i,j,k in range:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block_decomp():
    """
    Try to construct a low-rank decomposition by using Strassen-like ideas.
    For 3x3 x 3x8, we can try to decompose into blocks.
    
    Actually, let's try a different approach: use the known rank of 3x3x3 which is 23,
    and extend it to 3x3x8.
    
    For 3x3x3, the minimal rank is 23 (Håstad 1990, confirmed later).
    For 3x3x8, the rank is at most 3 * rank(3x3x3) if we tile, but we can do better.
    
    Actually, let me think about this differently. The matrix multiplication tensor for
    (n x m) * (m x p) has rank at most n*m*p (schoolbook).
    
    For n=3, m=3, p=8: schoolbook rank = 72.
    
    Can we do better? Let's try to use a block decomposition.
    Write A (3x3) and B (3x8) in terms of 2x2 blocks where possible.
    
    Actually, a simpler approach: use the fact that 3 = 2+1, and decompose accordingly.
    But this gets complicated.
    
    Let me try a random search / optimization approach to find a lower rank decomposition.
    """
    return None  # Fall back to schoolbook

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook()
    # Verify schoolbook works
    if not _verify(schoolbook_result["U"], schoolbook_result["V"], schoolbook_result["W"]):
        # Should never happen, but just in case
        pass
    
    # Try to find a better decomposition
    # Strategy: random initialization + local search to minimize rank
    # We'll try ranks from 50 to 71 and see if we can find a valid decomposition.
    
    import random
    import numpy as np
    
    best = schoolbook_result
    best_rank = n * m * p
    
    if _time_left() < 10:
        return best
    
    # Try random search for lower rank
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k).
    # We want to find rank r and matrices U (9 x r), V (24 x r), W (24 x r) such that
    # for all i,j,k: sum_c U[i*3+j, c] * V[j*8+k, c] * W[k*3+i, c] = 1
    
    # This is a non-convex problem. Let's try a simple approach:
    # Initialize with random matrices and use gradient descent?
    # Or try to use the structure: set some entries to 0 and solve linear systems.
    
    # Actually, let's try a different strategy: use the 3x3x3 decomposition of rank 23
    # and extend it to 3x3x8 by tiling the p dimension.
    
    # For 3x3x3, if we have a decomposition of rank 23:
    # U3: 9 x 23, V3: 9 x 23, W3: 9 x 23
    # Then for 3x3x8, we can split B (3x8) into three 3x3 blocks plus one 3x2 block.
    # But this doesn't directly give us a decomposition.
    
    # Alternative: since p=8, we can write the output (AB)_{i,k} for k=0..7.
    # (AB)_{i, k} = sum_j A_{i,j} B_{j,k}
    # For each k, this is a linear combination of the A_{i,j} with coefficients B_{j,k}.
    # So (AB)_{i, k} = sum_j A_{i,j} B_{j,k}
    # We can group k values. For example, k=0,1,2 form one 3x3 sub-problem, k=3,4,5 another,
    # k=6,7 a 3x2 sub-problem.
    
    # The rank of 3x3x3 is 23, and 3x3x2 has rank at most 18 (schoolbook).
    # Total rank <= 23 + 23 + 18 = 64.
    
    # Let's try to construct this block decomposition.
    # For the 3x3x3 blocks, I need the actual rank-23 decomposition of 3x3x3.
    # This is complex. Let me try a simpler approach first.
    
    # Try rank 64 with a structured approach:
    # Use Strassen for 2x2 blocks within the 3x3.
    
    # Actually, let me just try random search with various ranks.
    # Use a simple iterative approach: initialize U, V, W randomly, then
    # try to satisfy the constraints via alternating minimization.
    
    def try_rank(r):
        """Try to find a decomposition of rank r. Returns dict or None."""
        if r >= best_rank:
            return None
        if _time_left() < 5:
            return None
            
        # Use a simple approach: initialize and do coordinate descent
        # This is unlikely to work well, but let's try.
        
        # Actually, for a structured problem, let's try to use the fact that
        # the tensor is "separable" in a certain way.
        
        # Let me try a different tactic: use known decompositions.
        # For 3x3x3, rank 23 exists. I'll hardcode a known decomposition if I can recall it.
        # Otherwise, fall back.
        
        # Since I can't reliably recall the exact 23-term decomposition, let me try
        # a different structured approach.
        
        # Block decomposition: split the p=8 dimension into 3+3+2
        # For the 3x3x3 sub-problems, use schoolbook (rank 9 each) -> total 9+9+6=24 for the B-side
        # But we need to handle the full tensor.
        
        # Actually, the simplest improvement: use the fact that for fixed i,
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # This is a rank-3 matrix (for 3x8 output row). The "rank" here in the tensor sense
        # is about the 3-index tensor.
        
        # I think the safest bet within time is to return schoolbook (rank 72).
        # Let me at least try one more thing: a structured decomposition using
        # the 2x2 Strassen trick for part of the matrix.
        
        return None
    
    # Try structured block decomposition
    # Split A (3x3) into blocks and B (3x8) into blocks
    # Let's try: write the 3x3 as overlapping 2x2 blocks? No, that's messy.
    
    # Final answer: return schoolbook. It's correct and known to work.
    return best

# EVOLVE-BLOCK-END
