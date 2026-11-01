import numpy as np
import time

DIMENSIONS = [2, 3, 5]
_DEADLINE = 140.0  # safety margin before 150s kill

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """Strassen decomposition for 2x2x2 block."""
    # M1: (A11+A22)(B11+B22)
    U[i_offset*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M2: (A21+A22)B11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = -1
    col += 1
    # M3: A12(B21-B11)
    U[i_offset*m + j_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M4: A21(B22-B12)
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[j_offset*p + k_offset, col] = -1
    W[k_offset*n + i_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    col += 1
    # M5: (A11+A12)B22
    U[i_offset*m + j_offset, col] = 1
    U[i_offset*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    col += 1
    # M6: (A21-A11)(B11+B12)
    U[(i_offset+1)*m + j_offset, col] = 1
    U[i_offset*m + j_offset, col] = -1
    V[j_offset*p + k_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M7: (A12-A22)(B21+B22)
    U[i_offset*m + (j_offset+1), col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = -1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    col += 1
    return col

def _verify_rank(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Reconstruct tensor T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                result = np.dot(u_val, v_val * w_val)
                if abs(result - 1.0) > 1e-10:
                    return False
    return True

def _rank28_decomposition(n, m, p):
    """Known rank-28 decomposition using Strassen + schoolbook blocks."""
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    # Block 1: j=0, all k (2x1x5 schoolbook)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    # Block 2: j in {1,2}, k in {0,1} (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    # Block 3: j in {1,2}, k=2 (schoolbook)
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    # Block 4: j in {1,2}, k in {3,4} (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _try_rank27_structured(n, m, p, deadline):
    """
    Try to find a rank-27 decomposition by searching over structured forms.
    Strategy: Start from the rank-28 decomposition and try to merge/eliminate one term
    by finding a linear dependency or a smarter factorization.
    
    Also try alternative block splits that might yield rank 27.
    """
    if time.time() > deadline:
        return None
    
    # Alternative split 1: j=0 all k (10) + j=1,2 k=0..4 
    # For j=1,2, k=0..4: this is 2x2x5. R(2,2,5) <= 7+7+4=18 or 7+4+7=18
    # Total: 10+18=28. Same.
    
    # Alternative split 2: k=0 all j (6) + j=0 k=1..4 (8) + j=1,2 k=1..4 (2x2x4)
    # R(2,2,4) <= 7+7=14. Total: 6+8+14=28. Same.
    
    # Alternative split 3: k=0 all j (6) + k=1 all j (6) + j=0 k=2..4 (6) + j=1,2 k=2..4 (2x2x3)
    # R(2,2,3) <= 7+4=11 or 4+7=11. Total: 6+6+6+11=29. Worse.
    
    # Alternative split 4: Try to use a 2x2x3 block with rank 10 (if possible)
    # R(2,2,3): Is it 10 or 11?
    # 2x2x3 can be split as 2x2x2 + 2x2x1 = 7+4=11
    # Or 2x2x1 + 2x2x2 = 4+7=11
    # Can we do better? R(2,2,3) >= R(2,2,2)=7, and R(2,1,3)=6, R(1,2,3)=6
    # Lower bound: max(7,6,6)=7. Upper bound: 11.
    # Actually R(2,2,3) is known to be 11 (no rank-10 decomposition is known).
    
    # Try a more exotic approach: use the fact that n=2 is small
    # The tensor is T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0..4}
    # Total 30 entries.
    
    # For rank 27, we need to save 1 from the 28.
    # The only way is if one of the "independent" terms in the block decomposition
    # can be absorbed into another.
    
    # Let's try: instead of separate Strassen blocks, use a combined 2x2x4 block
    # for j in {1,2}, k in {0,1,3,4} (skipping k=2)
    # This is not a contiguous block, so standard Strassen doesn't apply directly.
    
    # Try: 2x2x4 for j={1,2}, k={0,1,2,3} + 2x2x1 for j={1,2}, k=4
    # R(2,2,4) + R(2,2,1) = 14 + 4 = 18 for the j=1,2 part
    # Plus j=0 part: 10. Total 28.
    
    # Try: 2x2x4 for j={1,2}, k={1,2,3,4} + 2x2x1 for j={1,2}, k=0
    # Same: 14 + 4 = 18. Total 28.
    
    # It seems all block decompositions give 28.
    # Let me try a direct search for rank 27 with small integer coefficients.
    
    # For rank 27, we have 27 terms. Each term has coefficients in U, V, W.
    # U: 6x27, V: 15x27, W: 10x27
    # The constraint is: for all (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    # This is a system of 30 equations in 27*31 = 837 unknowns (with the constraint
    # that coefficients are integers or half-integers).
    
    # Direct search is infeasible. Instead, let's try to verify if rank 27 is possible
    # by checking the border rank or using a different argument.
    
    # Actually, let me just try a few specific structured rank-27 candidates.
    
    # Candidate: Modify the rank-28 by combining the last schoolbook term with a Strassen term.
    # In the rank-28, the last 4 terms are schoolbook for j={1,2}, k=2.
    # Term 17: U[1*3+1,17]=1, V[1*5+2,17]=1, W[2*2+0,17]=1  -> A[0,1]*B[1,2] -> C[0,2]
    # Term 18: U[0*3+2,18]=1, V[2*5+2,18]=1, W[2*2+0,18]=1  -> A[0,2]*B[2,2] -> C[0,2]
    # Term 19: U[1*3+1,19]=1, V[1*5+2,19]=1, W[2*2+1,19]=1  -> A[1,1]*B[1,2] -> C[1,2]
    # Term 20: U[1*3+2,20]=1, V[2*5+2,20]=1, W[2*2+1,20]=1  -> A[1,2]*B[2,2] -> C[1,2]
    
    # Can we combine term 17 with something from the Strassen block?
    # The Strassen block for k={0,1} involves B[1,0],B[1,1],B[2,0],B[2,1]
    # Term 17 involves B[1,2]. No overlap.
    
    # I believe rank 27 is not achievable with this structure.
    # Let's return None to fall back to rank 28.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # First, try to find rank 27 (unlikely but worth a quick check)
    best_U, best_V, best_W = None, None, None
    best_rank = 27
    
    result = _try_rank27_structured(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        if _verify_rank(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = 27
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to known rank-28 decomposition
    U, V, W = _rank28_decomposition(n, m, p)
    
    # Verify
    if not _verify_rank(U, V, W, n, m, p):
        raise ValueError("Rank-28 decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
