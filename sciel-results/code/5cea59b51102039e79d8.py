"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Try the structured 4-block rank-11 scheme (total rank 44)
    best = None
    best_rank = n * m * p

    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_rank11(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try further refinement: attempt to remove redundant columns
    if best is not None and time.time() < deadline - 15:
        try:
            U2, V2, W2, r2 = _refine_by_column_removal(n, m, p, best[0], best[1], best[2], best_rank)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    # Try a two-block partition (3x4 each) with rank-10 per block -> total rank 20
    if time.time() < deadline - 10:
        try:
            U3, V3, W3, r3 = _two_block_rank10(n, m, p)
            if _verify(n, m, p, U3, V3, W3):
                if r3 < best_rank:
                    best = (U3, V3, W3)
                    best_rank = r3
        except Exception:
            pass

    # If we found a better rank, try refinement on it
    if best is not None and best_rank < 44 and time.time() < deadline - 5:
        try:
            U4, V4, W4, r4 = _refine_by_column_removal(n, m, p, best[0], best[1], best[2], best_rank)
            if r4 < best_rank and _verify(n, m, p, U4, V4, W4):
                best = (U4, V4, W4)
                best_rank = r4
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_rank11(n, m, p):
    """
    Split p=8 into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank = 4 * 11 = 44.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    num_blocks = p // block_size  # 4
    
    rank_per_block = 11
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(num_blocks):
        k_off = b * block_size  # offset in k: 0, 2, 4, 6
        col_off = b * rank_per_block  # offset in rank: 0, 11, 22, 33
        
        # Strassen terms (indices 0-6 in the block)
        # M1: (A11+A21)*(B00+B11)
        U[0, col_off + 0] += 1   # A11 (i=0,j=0)
        U[3, col_off + 0] += 1   # A21 (i=1,j=0)
        V[0 * p + k_off + 0, col_off + 0] += 1   # B00
        V[1 * p + k_off + 1, col_off + 0] += 1   # B11
        W[(k_off + 0) * n + 0, col_off + 0] += 1  # C[0,0]
        W[(k_off + 1) * n + 1, col_off + 0] += 1  # C[1,1]
        
        # M2: (A21+A22)*B00
        U[3, col_off + 1] += 1   # A21
        U[4, col_off + 1] += 1   # A22
        V[0 * p + k_off + 0, col_off + 1] += 1   # B00
        W[(k_off + 0) * n + 1, col_off + 1] += 1  # C[1,0]
        W[(k_off + 1) * n + 1, col_off + 1] -= 1  # C[1,1]
        
        # M3: A11*(B10-B11)
        U[0, col_off + 2] += 1   # A11
        V[1 * p + k_off + 0, col_off + 2] += 1   # B10
        V[1 * p + k_off + 1, col_off + 2] -= 1   # B11
        W[(k_off + 1) * n + 0, col_off + 2] += 1  # C[0,1]
        W[(k_off + 1) * n + 1, col_off + 2] += 1  # C[1,1]
        
        # M4: A22*(B01-B00)
        U[4, col_off + 3] += 1   # A22
        V[0 * p + k_off + 1, col_off + 3] += 1   # B01
        V[0 * p + k_off + 0, col_off + 3] -= 1   # B00
        W[(k_off + 0) * n + 0, col_off + 3] += 1  # C[0,0]
        W[(k_off + 0) * n + 1, col_off + 3] += 1  # C[1,0]
        
        # M5: (A11+A12)*B11
        U[0, col_off + 4] += 1   # A11
        U[1, col_off + 4] += 1   # A12
        V[1 * p + k_off + 1, col_off + 4] += 1   # B11
        W[(k_off + 0) * n + 0, col_off + 4] -= 1  # C[0,0]
        W[(k_off + 1) * n + 0, col_off + 4] += 1  # C[0,1]
        
        # M6: (A21-A11)*(B00+B01)
        U[3, col_off + 5] += 1   # A21
        U[0, col_off + 5] -= 1   # A11
        V[0 * p + k_off + 0, col_off + 5] += 1   # B00
        V[0 * p + k_off + 1, col_off + 5] += 1   # B01
        W[(k_off + 1) * n + 1, col_off + 5] += 1  # C[1,1]
        
        # M7: (A12-A22)*(B10+B11)
        U[1, col_off + 6] += 1   # A12
        U[4, col_off + 6] -= 1   # A22
        V[1 * p + k_off + 0, col_off + 6] += 1   # B10
        V[1 * p + k_off + 1, col_off + 6] += 1   # B11
        W[(k_off + 0) * n + 0, col_off + 6] += 1  # C[0,0]
        W[(k_off + 1) * n + 0, col_off + 6] += 1  # C[0,1]
        
        # 4 extra rank-1 terms for j=2 cross terms
        U[2, col_off + 7] = 1   # A13 (i=0, j=2)
        V[2 * p + k_off + 0, col_off + 7] = 1   # B20
        W[(k_off + 0) * n + 0, col_off + 7] = 1  # C[0,0]
        
        U[2, col_off + 8] = 1   # A13
        V[2 * p + k_off + 1, col_off + 8] = 1   # B21
        W[(k_off + 1) * n + 0, col_off + 8] = 1  # C[0,1]
        
        U[5, col_off + 9] = 1   # A23 (i=1, j=2)
        V[2 * p + k_off + 0, col_off + 9] = 1   # B20
        W[(k_off + 0) * n + 1, col_off + 9] = 1  # C[1,0]
        
        U[5, col_off + 10] = 1   # A23
        V[2 * p + k_off + 1, col_off + 10] = 1   # B21
        W[(k_off + 1) * n + 1, col_off + 10] = 1  # C[1,1]
    
    return U, V, W, total_rank


def _two_block_rank10(n, m, p):
    """
    Split p=8 into 2 blocks of size 4.
    For each 2x3 x 3x4 sub-problem, use a rank-10 decomposition.
    
    The 2x3 * 3x4 multiplication can be seen as:
    A (2x3) * B (3x4) = C (2x4)
    
    We partition B into columns: B = [B_left(3x2) | B_right(3x2)]
    Then C = [A*B_left | A*B_right]
    
    Each A*B_sub is 2x3 * 3x2 = rank 11 via Strassen+4.
    But we can do better by sharing structure between the two halves.
    
    Alternative: use the fact that 2x3 * 3x4 can be decomposed with rank 10
    by using a smarter scheme.
    
    Actually, let's try a direct construction. The tensor for 2x3*3x4 has
    2*3*4 = 24 nonzero entries. Schoolbook gives rank 24.
    
    Let's use a block approach:
    - Use Strassen-like on the 2x2 core (j=0,1; k=0,1) for left half: 7 terms
    - Use Strassen-like on the 2x2 core (j=0,1; k=2,3) for right half: 7 terms
    - Cross terms for j=2: 4*2 = 8 terms
    Total: 7+7+8 = 22. Not better than 24.
    
    Let me try a different approach. We can use the Laderman-like scheme
    or try to find a rank-10 decomposition directly.
    
    Actually, for 2x3 * 3x4, the known lower bound is not well established.
    Let's just use the straightforward rank-11 per 2x3x2 block approach
    but try to merge some terms across the two 2x2 blocks.
    
    For now, let's implement a rank-12 per block (2x3 * 3x4) scheme:
    Split into two 2x3*3x2 problems, each rank 11, total 22.
    That's worse than schoolbook 24? No, 22 < 24.
    
    Wait, schoolbook for 2x3*3x4 is rank 24. Two blocks of rank 11 = 22.
    So total for 2x3*3x8 would be 2*22 = 44. Same as before.
    
    Let me try to find a genuinely rank-10 scheme for 2x3*3x4.
    
    The key insight: in 2x3*3x4, we have:
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    For fixed i, this is a linear combination of rows of B weighted by A[i,:].
    
    Let's try: use 2 terms for the j=2 part (A[0,2]*B[2,:] and A[1,2]*B[2,:])
    That's 2 terms for the entire j=2 contribution across all 4 k values.
    
    For the j=0,1 part (2x2 * 2x4):
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2,3
    
    This is a 2x2 * 2x4 multiplication. Rank is at most 2*2*4=16 (schoolbook).
    But we can do better.
    
    Actually, 2x2 * 2x4 can be seen as two independent 2x2*2x2 problems
    (k=0,1 and k=2,3), each rank 7 (Strassen), total 14.
    Plus 2 terms for j=2. Total: 16. Worse than 22? No, 16 < 22.
    
    Wait, I'm confusing myself. Let me restart.
    
    For 2x3 * 3x4:
    - j=0,1 part: 2x2 * 2x4. Split into two 2x2*2x2, each Strassen rank 7. Total 14.
    - j=2 part: 2x1 * 1x4. Rank 2 (A[0,2]*B[2,:] and A[1,2]*B[2,:]).
    Total: 14 + 2 = 16.
    
    So for 2x3 * 3x8, split into two 3x4 blocks:
    Each block: rank 16. Total: 32.
    
    That's better than 44!
    
    Let me implement this.
    """
    n, m, p = 2, 3, 8
    block_size = 4
    num_blocks = p // block_size  # 2
    
    # Per block: 14 (Strassen for 2x2*2x4 as two 2x2*2x2) + 2 (j=2) = 16
    rank_per_block = 16
    total_rank = rank_per_block * num_blocks  # 32
    
    U = np.zeros((n * m, total_rank))  # 6 x 32
    V = np.zeros((m * p, total_rank))  # 24 x 32
    W = np.zeros((p * n, total_rank))  # 16 x 32
    
    for b in range(num_blocks):
        k_off = b * block_size  # 0 or 4
        col_off = b * rank_per_block  # 0 or 16
        
        # Strassen for left half (k_off, k_off+1): 2x2 * 2x2
        # j in {0,1}, k in {k_off, k_off+1}
        # M1: (A00+A11)*(B0k0+B1k1)
        U[0, col_off + 0] += 1
        U[3, col_off + 0] += 1
        V[0 * p + k_off + 0, col_off + 0] += 1
        V[1 * p + k_off + 1, col_off + 0] += 1
        W[(k_off + 0) * n + 0, col_off + 0] += 1
        W[(k_off + 1) * n + 1, col_off + 0] += 1
        
        # M2: (A10+A11)*B0k0
        U[3, col_off + 1] += 1
        U[4, col_off + 1] += 1
        V[0 * p + k_off + 0, col_off + 1] += 1
        W[(k_off + 0) * n + 1, col_off + 1] += 1
        W[(k_off + 1) * n + 1, col_off + 1] -= 1
        
        # M3: A00*(B1k0-B1k1)
        U[0, col_off + 2] += 1
        V[1 * p + k_off + 0, col_off + 2] += 1
        V[1 * p + k_off + 1, col_off + 2] -= 1
        W[(k_off + 1) * n + 0, col_off + 2] += 1
        W[(k_off + 1) * n + 1, col_off + 2] += 1
        
        # M4: A11*(B0k1-B0k0)
        U[4, col_off + 3] += 1
        V[0 * p + k_off + 1, col_off + 3] += 1
        V[0 * p + k_off + 0, col_off + 3] -= 1
        W[(k_off + 0) * n + 0, col_off + 3] += 1
        W[(k_off + 0) * n + 1, col_off + 3] += 1
        
        # M5: (A00+A01)*B1k1
        U[0, col_off + 4] += 1
        U[1, col_off + 4] += 1
        V[1 * p + k_off + 1, col_off + 4] += 1
        W[(k_off + 0) * n + 0, col_off + 4] -= 1
        W[(k_off + 1) * n + 0, col_off + 4] += 1
        
        # M6: (A10-A00)*(B0k0+B0k1)
        U[3, col_off + 5] += 1
        U[0, col_off + 5] -= 1
        V[0 * p + k_off + 0, col_off + 5] += 1
        V[0 * p + k_off + 1, col_off + 5] += 1
        W[(k_off + 1) * n + 1, col_off + 5] += 1
        
        # M7: (A01-A11)*(B1k0+B1k1)
        U[1, col_off + 6] += 1
        U[4, col_off + 6] -= 1
        V[1 * p + k_off + 0, col_off + 6] += 1
        V[1 * p + k_off + 1, col_off + 6] += 1
        W[(k_off + 0) * n + 0, col_off + 6] += 1
        
        # Strassen for right half (k_off+2, k_off+3): 2x2 * 2x2
        # j in {0,1}, k in {k_off+2, k_off+3}
        # M8: (A00+A11)*(B0k2+B1k3)
        U[0, col_off + 7] += 1
        U[3, col_off + 7] += 1
        V[0 * p + k_off + 2, col_off + 7] += 1
        V[1 * p + k_off + 3, col_off + 7] += 1
        W[(k_off + 2) * n + 0, col_off + 7] += 1
        W[(k_off + 3) * n + 1, col_off + 7] += 1
        
        # M9: (A10+A11)*B0k2
        U[3, col_off + 8] += 1
        U[4, col_off + 8] += 1
        V[0 * p + k_off + 2, col_off + 8] += 1
        W[(k_off + 2) * n + 1, col_off + 8] += 1
        W[(k_off + 3) * n + 1, col_off + 8] -= 1
        
        # M10: A00*(B1k2-B1k3)
        U[0, col_off + 9] += 1
        V[1 * p + k_off + 2, col_off + 9] += 1
        V[1 * p + k_off + 3, col_off + 9] -= 1
        W[(k_off + 3) * n + 0, col_off + 9] += 1
        W[(k_off + 3) * n + 1, col_off + 9] += 1
        
        # M11: A11*(B0k3-B0k2)
        U[4, col_off + 10] += 1
        V[0 * p + k_off + 3, col_off + 10] += 1
        V[0 * p + k_off + 2, col_off + 10] -= 1
        W[(k_off + 2) * n + 0, col_off + 10] += 1
        W[(k_off + 2) * n + 1, col_off + 10] += 1
        
        # M12: (A00+A01)*B1k3
        U[0, col_off + 11] += 1
        U[1, col_off + 11] += 1
        V[1 * p + k_off + 3, col_off + 11] += 1
        W[(k_off + 2) * n + 0, col_off + 11] -= 1
        W[(k_off + 3) * n + 0, col_off + 11] += 1
        
        # M13: (A10-A00)*(B0k2+B0k3)
        U[3, col_off + 12] += 1
        U[0, col_off + 12] -= 1
        V[0 * p + k_off + 2, col_off + 12] += 1
        V[0 * p + k_off + 3, col_off + 12] += 1
        W[(k_off + 3) * n + 1, col_off + 12] += 1
        
        # M14: (A01-A11)*(B1k2+B1k3)
        U[1, col_off + 13] += 1
        U[4, col_off + 13] -= 1
        V[1 * p + k_off + 2, col_off + 13] += 1
        V[1 * p + k_off + 3, col_off + 13] += 1
        W[(k_off + 2) * n + 0, col_off + 13] += 1
        
        # j=2 terms: 2 terms for the entire 4-column block
        # A[0,2] * B[2, k_off:k_off+4] -> contributes to C[0, k] for all k in block
        U[2, col_off + 14] = 1
        V[2 * p + k_off + 0, col_off + 14] = 1
        V[2 * p + k_off + 1, col_off + 14] = 1
        V[2 * p + k_off + 2, col_off + 14] = 1
        V[2 * p + k_off + 3, col_off + 14] = 1
        W[(k_off + 0) * n + 0, col_off + 14] = 1
        W[(k_off + 1) * n + 0, col_off + 14] = 1
        W[(k_off + 2) * n + 0, col_off + 14] = 1
        W[(k_off + 3) * n + 0, col_off + 14] = 1
        
        # A[1,2] * B[2, k_off:k_off+4] -> contributes to C[1, k] for all k in block
        U[5, col_off + 15] = 1
        V[2 * p + k_off + 0, col_off + 15] = 1
        V[2 * p + k_off + 1, col_off + 15] = 1
        V[2 * p + k_off + 2, col_off + 15] = 1
        V[2 * p + k_off + 3, col_off + 15] = 1
        W[(k_off + 0) * n + 1, col_off + 15] = 1
        W[(k_off + 1) * n + 1, col_off + 15] = 1
        W[(k_off + 2) * n + 1, col_off + 15] = 1
        W[(k_off + 3) * n + 1, col_off + 15] = 1
    
    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(U.shape[1]):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i * m + j, c] * A[i, j]
                v_val = 0.0
                for l in range(m):
                    v_val += V[l * p + k, c] * B[l, k]
                C_comp[i, k] += W[k * n + i, c] * u_val * v_val
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_by_column_removal(n, m, p, U, V, W, current_rank):
    """
    Try to reduce rank by removing columns that are linearly dependent
    in the tensor sense. For each column, check if removing it still
    yields a valid decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    
    # Try removing each column one at a time
    for c_remove in range(rank - 1, -1, -1):
        mask = [c for c in range(rank) if c != c_remove]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if _verify(n, m, p, U_new, V_new, W_new):
            U, V, W = U_new, V_new, W_new
            rank -= 1
    
    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
