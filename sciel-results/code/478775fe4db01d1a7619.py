import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _build_strassen_114(n, m, p):
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S222 = [
        ([1,0,0,1],[1,0,0,1],[1,0,0,1]),
        ([0,0,1,1],[1,0,0,0],[0,0,1,0]),
        ([1,0,0,0],[0,1,0,-1],[0,1,0,0]),
        ([0,0,0,1],[0,-1,1,0],[1,0,1,0]),
        ([1,1,0,0],[0,0,0,1],[-1,1,0,0]),
        ([-1,0,1,0],[1,1,0,0],[0,0,0,1]),
        ([0,1,0,-1],[0,0,1,1],[1,0,0,0]),
    ]
    
    col = 0
    for (u,v,w) in S222:
        for t in range(4):
            if u[t]: U[t][col] += u[t]
            if v[t]: V[t][col] += v[t]
            if w[t]: W[t][col] += w[t]
        col += 1
    for (u,v,w) in S222:
        for t in range(4):
            if u[t]: U[t][col] += u[t]
            if v[t]: V[4+t][col] += v[t]
            if w[t]: W[t][col] += w[t]
        col += 1
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+8][col] = 1
            W[8*4+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i*m+j, j*p+k, k*n+i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def _build_combined_108(n, m, p):
    """
    Try a combined decomposition: split 4x9 into 4x4 + 4x4 + 4x1.
    Use a shared structure for the two 4x4 blocks to save rank.
    
    Key idea: The two 4x4*4x4 subproblems share the same A matrix.
    Instead of using 7+7=14 independent Strassen terms, we can try to
    use a combined decomposition where some terms serve both blocks.
    
    For the 4x1 block, we need rank 16 (schoolbook) or potentially less
    if we can share with the 4x4 blocks.
    
    Total: aim for rank 108 = 7 + 7 + 14 (saving 2 from the 4x1 block)
    or rank 105 = 7 + 7 + 11, etc.
    """
    # Start with the 114-rank structure and try to compress the 4x1 part.
    # The 4x1 part has 16 schoolbook terms:
    # U[i*4+j, r] = 1, V[j*4+8, r] = 1, W[8*4+i, r] = 1
    # for i,j in range(4), r = 28 + i*4 + j
    
    # Can we reduce this? The 4x4*4x1 multiplication has tensor rank 16
    # (it's a matrix-vector product, and the rank is m*p = 4*1 = 4... wait no).
    # Actually, for A(4x4) * b(4x1) = c(4x1), the tensor is T[i,j,k] where
    # k is always 0 (since p=1). So T[i,j,0] = 1 for all i,j.
    # The rank of this tensor is 4 (not 16) because T[i,j,0] = sum_r u_r[i,j] * v_r[j,0] * w_r[0,i]
    # We need: for all i,j: sum_r u_r[i,j] * v_r[j,0] * w_r[0,i] = 1
    # And for all (a,b,c) not of form (i,j,0): sum_r u_r[a] * v_r[b] * w_r[c] = 0
    
    # If we restrict to p=1, then V has shape (4*1, R) = (4, R) and W has shape (1*4, R) = (4, R).
    # The constraint is: T[i,j,0] = sum_r U[i*4+j, r] * V[j, r] * W[i, r] = 1 for all i,j.
    # And T[a,b,c] = 0 for c != 0, but c is always 0 when p=1, so no extra constraints.
    # Wait, when p=1, the W index is k*n+i = 0*4+i = i, so W has 4 entries.
    # V index is j*p+k = j*1+0 = j, so V has 4 entries.
    # U index is i*m+j = i*4+j, so U has 16 entries.
    
    # So we need: sum_r U[i*4+j,r] * V[j,r] * W[i,r] = 1 for all i,j in 0..3.
    # This is 16 equations. With R terms, we have 16R + 4R + 4R = 24R unknowns.
    # For R=4: 96 unknowns, 16 equations. Very underdetermined.
    # We can easily find a solution with R=4!
    
    # Let's construct it:
    # Choose W[i,r] = delta_{i,r} (identity), V[j,r] = delta_{j,r}, U[i*4+j,r] = 1 for all.
    # Then sum_r U[i*4+j,r] * V[j,r] * W[i,r] = sum_r 1 * delta_{j,r} * delta_{i,r} = delta_{i,j}.
    # That gives 1 only when i=j, not for all i,j.
    
    # Try: W[i,r] = 1 for all, V[j,r] = 1 for all, U[i*4+j,r] = 1/4.
    # Then sum_r (1/4) * 1 * 1 = R/4. Set R=4: gives 1. 
    # But we need integer/half-integer entries. 1/4 is not allowed.
    
    # Try R=4: W[i,r] = 1, V[j,r] = 1, U[i*4+j,r] = 1/4. Not integer.
    # Try R=8: W[i,r] = 1, V[j,r] = 1, U[i*4+j,r] = 1/8. Still not integer.
    
    # Alternative: Use the structure more carefully.
    # We need sum_r U[i*4+j,r] * V[j,r] * W[i,r] = 1 for all i,j.
    # Let's set V[j,r] = delta_{j,r} for r=0..3, and W[i,r] = delta_{i,r} for r=0..3.
    # Then sum_r U[i*4+j,r] * delta_{j,r} * delta_{i,r} = U[i*4+j,i] * delta_{i,j} (if i=j) 
    # This doesn't work for i != j.
    
    # Let's use a different approach: set V[j,r] = 1 for all j,r, W[i,r] = 1 for all i,r.
    # Then sum_r U[i*4+j,r] = 1 for all i,j.
    # We can set U[i*4+j,0] = 1 and U[i*4+j,r] = 0 for r>0. Rank 1!
    # But wait, that means V[j,0] = 1 for all j, W[i,0] = 1 for all i.
    # Check: sum_r U[i*4+j,r] * V[j,r] * W[i,r] = U[i*4+j,0] * V[j,0] * W[i,0] = 1*1*1 = 1. 
    # So the 4x4*4x1 block has rank 1!
    
    # But wait, we also need the cross-terms to be zero. In the full 4x4x9 problem,
    # the U, V, W vectors are shared across all blocks. The 4x1 block's U entries
    # are U[i*4+j, r] for r corresponding to the 4x1 block. These are separate from
    # the U entries used by the 4x4 blocks.
    
    # So in the combined decomposition, the 4x1 block can use rank 1 instead of 16!
    # Total rank: 7 + 7 + 1 = 15? That seems too good.
    
    # Wait, I think I'm making an error. Let me re-examine.
    # In the full problem, U has shape (16, R), V has shape (36, R), W has shape (36, R).
    # The 4x1 block uses:
    #   U indices: i*4+j for i,j in 0..3 (all 16 entries)
    #   V indices: j*4+8 for j in 0..3 (entries 8,9,10,11)
    #   W indices: 8*4+i for i in 0..3 (entries 32,33,34,35)
    #
    # For the rank-1 term: U[i*4+j, r] = 1 for all i,j, V[j*4+8, r] = 1 for all j, W[8*4+i, r] = 1 for all i.
    # Contribution to T[i,j,k]: U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r]
    # For k=8: = 1 * 1 * 1 = 1. Correct!
    # For k != 8: = 1 * V[j*4+k,r] * W[k*4+i,r]
    #   V[j*4+k,r] is 0 for k != 8 (since we only set V[8..11, r] = 1).
    #   So the contribution is 0 for k != 8. Correct!
    #
    # But wait, this rank-1 term also contributes to other (i,j,k) entries through
    # the U[i*4+j,r] = 1 entries. Specifically:
    #   T[i,j,k] += U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r]
    #   For k=8: += 1 * 1 * 1 = 1 (desired)
    #   For k != 8: += 1 * 0 * 0 = 0 (since V[j*4+k,r] = 0 for k != 8)
    #
    # So it works! The 4x1 block can be done in rank 1.
    #
    # But hold on - does this interfere with the 4x4 blocks? The U entries U[i*4+j,r]
    # are shared. If the 4x4 blocks also use these U entries, there could be interference.
    #
    # In the Strassen decomposition, the 4x4 blocks use U entries U[0..15, 0..6] and U[0..15, 7..13].
    # The 4x1 block would use U[0..15, 14] (a new column). Since it's a new column,
    # there's no interference.
    #
    # So total rank = 7 + 7 + 1 = 15?!
    
    # Wait, that can't be right. The schoolbook rank is 144, and Strassen gives 114.
    # Rank 15 would be a massive improvement. Let me double-check.
    
    # The issue: in the full tensor, T[i,j,k] for k=8 (the 9th column) should be 1
    # for all i,j. The rank-1 term gives exactly that.
    # For k=0..7, the rank-1 term gives 0 (since V[j*4+k,14] = 0 for k != 8).
    # So it doesn't interfere with the 4x4 blocks.
    
    # This means the total rank is 7 + 7 + 1 = 15!
    
    # Let me verify this more carefully by constructing it and checking.
    
    R = 15
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(36)]
    W = [[0] * R for _ in range(36)]
    
    # First 4x4 block (cols 0-3): Strassen terms 0-6
    S222 = [
        ([1,0,0,1],[1,0,0,1],[1,0,0,1]),
        ([0,0,1,1],[1,0,0,0],[0,0,1,0]),
        ([1,0,0,0],[0,1,0,-1],[0,1,0,0]),
        ([0,0,0,1],[0,-1,1,0],[1,0,1,0]),
        ([1,1,0,0],[0,0,0,1],[-1,1,0,0]),
        ([-1,0,1,0],[1,1,0,0],[0,0,0,1]),
        ([0,1,0,-1],[0,0,1,1],[1,0,0,0]),
    ]
    
    col = 0
    for (u,v,w) in S222:
        for t in range(4):
            if u[t]: U[t][col] += u[t]
            if v[t]: V[t][col] += v[t]
            if w[t]: W[t][col] += w[t]
        col += 1
    
    # Second 4x4 block (cols 4-7): Strassen terms 7-13
    for (u,v,w) in S222:
        for t in range(4):
            if u[t]: U[t][col] += u[t]
            if v[t]: V[4+t][col] += v[t]
            if w[t]: W[t][col] += w[t]
        col += 1
    
    # Third 4x1 block (col 8): rank-1 term at col 14
    for i in range(4):
        for j in range(4):
            U[i*4+j][14] = 1
    for j in range(4):
        V[j*4+8][14] = 1
    for i in range(4):
        W[8*4+i][14] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try rank-15 combined decomposition
    res = _build_combined_108(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
    
    # Fallback: Strassen-based rank-114
    res = _build_strassen_114(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
    
    # Fallback: schoolbook rank 144
    return _build_schoolbook(n, m, p)
