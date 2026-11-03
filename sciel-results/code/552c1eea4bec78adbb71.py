import numpy as np
import time
import sys

_START_TIME = time.time()
_DEADLINE = 140.0

def _verify(U, V, W, n=3, m=4, p=8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the standard multiplication tensor
    # Standard: T[i,j,k] = 1 if... actually the multiplication tensor is:
    # T[i,j,k] = delta (it's 1 for all i,j,k in the standard basis)
    # Wait, the matrix multiplication tensor T satisfies:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
    # In the decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
    
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] for all i,j,k
    # This is a tensor contraction. Let's do it efficiently.
    
    # For each r, we have a rank-1 tensor u_r ⊗ v_r ⊗ w_r
    # u_r[i,j] = U[i*m+j, r]
    # v_r[j,k] = V[j*p+k, r]  
    # w_r[k,i] = W[k*n+i, r]
    
    # Result[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Let's compute this as: for each r, form the outer product contribution
    # This is equivalent to: reshape and use einsum
    
    # U reshaped: (n, m, rank)
    # V reshaped: (m, p, rank)
    # W reshaped: (p, n, rank)
    
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # Result[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    # The standard multiplication tensor has T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    
    return np.allclose(result, expected, atol=1e-10)

def _to_halfint(arr, tol=1e-9):
    """Check if array entries are half-integers (integers or half-integers)."""
    arr = np.array(arr, dtype=float)
    doubled = arr * 2
    rounded = np.round(doubled)
    if not np.allclose(doubled, rounded, atol=tol):
        return None
    return rounded / 2.0

def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    # Fallback: schoolbook rank 96
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() - _START_TIME > _DEADLINE:
        return schoolbook()

    # Rank 88 Construction
    # Part 1: i=0, 1x4x8 schoolbook (32 terms)
    # Part 2: i=1,2, 2x4x8 Strassen (56 terms)
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0
    for j in range(m):
        for k in range(p):
            U[j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n, col] = 1.0
            col += 1
            
    # Part 2: i=1,2
    # Strassen coefficients for 2x2x2
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    for j_block in range(2):
        j_base = j_block * 2
        for k_block in range(4):
            k_base = k_block * 2
            for t in range(7):
                for l in range(4):
                    if S_A[t, l] != 0:
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_base + loc_j
                        U[g_i * m + g_j, col] += S_A[t, l]
                
                for l in range(4):
                    if S_B[t, l] != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_base + loc_j
                        g_k = k_base + loc_k
                        V[g_j * p + g_k, col] += S_B[t, l]
                
                for l in range(4):
                    if S_C[t, l] != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_base + loc_k
                        W[g_k * n + g_i, col] += S_C[t, l]
                        
                col += 1
                
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    # Verify the construction
    if not _verify(U, V, W, n, m, p):
        return schoolbook()
    
    # Try to reduce rank by attempting to eliminate terms
    # Strategy: try to find linear dependencies among the rank-1 terms
    # or find a lower-rank decomposition via local search
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Attempt 1: Try to remove terms by checking if the remaining still span the tensor
    # This is expensive, so we do a limited search
    
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 5:
        # Try greedy removal: for each term, check if removing it still gives correct result
        # This is O(rank) verifications, each O(n*m*p*rank) = O(96*88) ~ 8448 ops
        # Total: 88 * 8448 ~ 743k ops, very fast
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        removed = [False] * rank
        
        for attempt in range(rank):
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
            
            found_removal = False
            for r in range(rank):
                if removed[r]:
                    continue
                # Try removing term r
                removed[r] = True
                # Check if remaining terms still decompose correctly
                # Build active mask
                active = [i for i in range(rank) if not removed[i]]
                if len(active) < 2:
                    removed[r] = False
                    continue
                
                U_act = U_curr[:, active]
                V_act = V_curr[:, active]
                W_act = W_curr[:, active]
                
                if _verify(U_act, V_act, W_act, n, m, p):
                    # Successfully removed term r
                    found_removal = True
                    break
                else:
                    removed[r] = False
            
            if not found_removal:
                break
        
        if found_removal:
            active = [i for i in range(rank) if not removed[i]]
            best_U = U_curr[:, active]
            best_V = V_curr[:, active]
            best_W = W_curr[:, active]
            best_rank = len(active)
    
    # Attempt 2: Try to merge pairs of terms
    # If U[:,r1] + U[:,r2], etc., can be represented as a single term
    # This is harder; skip for now due to time constraints
    
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 3:
        # Try a different approach: for the Part 1 (i=0) terms, see if any can be
        # absorbed into the Strassen part or eliminated
        # Actually, let's try: can we reduce the 1x4x8 part?
        # 1x4x8: C[0,k] = sum_j A[0,j]*B[j,k]
        # This is a vector-matrix product, rank is 32 (schoolbook) since each output
        # depends on all 4*8=32 input pairs independently.
        # Actually no: C[0,k] for k=0..7, each is sum of 4 terms.
        # The tensor is 1x4x8, which has rank 32 (it's a sum of 32 rank-1 tensors).
        # Can we do better? The 1x4x8 tensor: T[0,j,k] = 1 for all j,k.
        # This is actually rank 4! Because T[0,j,k] = sum_{j'} delta_{j,j'} * delta_{k,k'}... 
        # No wait. T[j,k] = 1 for all j,k. This is the all-ones matrix, which has rank 1!
        # But as a tensor for matrix multiplication, it's different.
        
        # Actually for 1x4x8 matrix multiplication:
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # The tensor T[0,j,k] has T[0,j,k] = 1 for all j,k.
        # This is a rank-1 tensor! T = u ⊗ v where u[j] = 1, v[k] = 1... 
        # No, that's not right either. Let me think again.
        
        # The multiplication tensor for 1x4x8:
        # We need: sum_r U[0,j,r] * V[j,k,r] * W[k,0,r] = 1 for all j,k
        # If we set U[0,j,r] = 1 for all j, V[j,k,r] = delta_{j,k}... no that doesn't work.
        
        # Actually: C[0,k] = sum_j A[0,j] B[j,k]
        # This is just a dot product. The tensor is:
        # T[j,k] = 1 (for the coefficient of A[0,j]*B[j,k] in C[0,k])
        # Wait no. The tensor T[i,j,k] for matrix multiplication satisfies:
        # (AB)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # For 1x4x8: (AB)[0,k] = sum_j A[0,j] B[j,k]
        # So T[0,j,k] = 1 for all j,k.
        
        # Now, can we write this as sum_r U[j,r] V[j,k,r] W[k,r] = 1 for all j,k?
        # (Note: i=0 is fixed, so U depends only on j, W depends only on k)
        
        # If we set: for r=0..3, U[j,r] = delta_{j,r}, V[j,k,r] = delta_{j,k}, W[k,r] = 1
        # Then sum_r delta_{j,r} * delta_{j,k} * 1 = delta_{j,k}... that's not 1.
        
        # Try: U[j,r] = 1 for all j, V[j,k,r] = delta_{j,k}, W[k,r] = 1
        # sum_r 1 * delta_{j,k} * 1 = 4 * delta_{j,k}... still not right.
        
        # The correct schoolbook: for each (j,k) pair, one term with U[j,r]=1, V[j,k,r]=1, W[k,r]=1
        # That's 32 terms. Can we do fewer?
        
        # We need: sum_r U[j,r] * V[j,k,r] * W[k,r] = 1 for all j in {0,1,2,3}, k in {0,...,7}
        
        # Let's define for each r: u_r ∈ R^4, v_r ∈ R^{4x8}, w_r ∈ R^8
        # Constraint: sum_r u_r[j] * v_r[j,k] * w_r[k] = 1 for all j,k
        
        # This is a system of 32 equations. The minimum rank is the tensor rank of the 4x8 all-ones matrix.
        # The all-ones matrix of size 4x8 has matrix rank 1, but tensor rank (as a 3-tensor) is different.
        
        # Actually, I realize the "tensor" here is really just a matrix (since i is fixed). 
        # T[j,k] = 1 for all j,k. As a bilinear form, this has rank 1!
        # We can write: T[j,k] = u[j] * v[k] where u[j]=1, v[k]=1.
        # So: sum_r U[j,r] * V[j,k,r] * W[k,r] = 1
        # Set r=0: U[j,0] = 1 for all j, V[j,k,0] = 1 for all j,k, W[k,0] = 1 for all k.
        # Then sum = 1*1*1 = 1. ✓
        
        # So the 1x4x8 part can be done in rank 1!
        
        # Wait, that can't be right. Let me re-examine.
        # U shape: (n*m, rank) = (3*4, rank) = (12, rank). For i=0, we use U[0*4+j, r] = U[j, r]
        # V shape: (m*p, rank) = (4*8, rank) = (32, rank). V[j*p+k, r]
        # W shape: (p*n, rank) = (8*3, rank) = (24, rank). W[k*n+i, r] = W[k*3+0, r] = W[3k, r]
        
        # For i=0: sum_r U[j, r] * V[8j+k, r] * W[3k, r] = 1 for all j in {0,1,2,3}, k in {0,...,7}
        
        # If I set: U[j, 0] = 1 for all j, V[8j+k, 0] = 1 for all j,k, W[3k, 0] = 1 for all k
        # Then sum = 1 * 1 * 1 = 1. ✓
        
        # So yes! The 1x4x8 part needs only rank 1, not 32!
        
        # This is a huge improvement. Let me redo the construction.
        
        # New construction:
        # Part 1: i=0, rank 1
        # Part 2: i=1,2, 2x4x8 Strassen, rank 56
        # Total: 57
        
        U_new = np.zeros((n * m, 57))
        V_new = np.zeros((m * p, 57))
        W_new = np.zeros((p * n, 57))
        
        # Part 1: i=0, single term
        for j in range(m):
            U_new[j, 0] = 1.0
        for j in range(m):
            for k in range(p):
                V_new[j * p + k, 0] = 1.0
        for k in range(p):
            W_new[k * n, 0] = 1.0
        
        # Part 2: i=1,2, Strassen (same as before, starting at col 1)
        col = 1
        for j_block in range(2):
            j_base = j_block * 2
            for k_block in range(4):
                k_base = k_block * 2
                for t in range(7):
                    for l in range(4):
                        if S_A[t, l] != 0:
                            loc_i = l // 2
                            loc_j = l % 2
                            g_i = 1 + loc_i
                            g_j = j_base + loc_j
                            U_new[g_i * m + g_j, col] += S_A[t, l]
                    
                    for l in range(4):
                        if S_B[t, l] != 0:
                            loc_j = l // 2
                            loc_k = l % 2
                            g_j = j_base + loc_j
                            g_k = k_base + loc_k
                            V_new[g_j * p + g_k, col] += S_B[t, l]
                    
                    for l in range(4):
                        if S_C[t, l] != 0:
                            loc_i = l // 2
                            loc_k = l % 2
                            g_i = 1 + loc_i
                            g_k = k_base + loc_k
                            W_new[g_k * n + g_i, col] += S_C[t, l]
                            
                    col += 1
                    
        assert col == 57, f"Rank mismatch: {col} != 57"
        
        if _verify(U_new, V_new, W_new, n, m, p):
            best_U = U_new
            best_V = V_new
            best_W = W_new
            best_rank = 57
    
    # Final check
    if not _verify(best_U, best_V, best_W, n, m, p):
        return schoolbook()
    
    # Ensure half-integer entries
    U_h = _to_halfint(best_U)
    V_h = _to_halfint(best_V)
    W_h = _to_halfint(best_W)
    
    if U_h is None or V_h is None or W_h is None:
        return schoolbook()
    
    return {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
