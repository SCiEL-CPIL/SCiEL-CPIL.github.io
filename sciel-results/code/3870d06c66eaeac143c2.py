"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, tol=1e-10):
    """Check if U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_225_decomp(rank_target=17, max_attempts=1000):
    """
    Try to find a rank decomposition for 2x2 * 2x5 matrix multiplication.
    Returns (U, V, W) as lists of lists, or None if not found.
    U: 4 x rank
    V: 10 x rank
    W: 10 x rank
    """
    n_sub, m_sub, p_sub = 2, 2, 5
    for attempt in range(max_attempts):
        rng = np.random.RandomState(attempt * 101 + 7)
        U = rng.randn(n_sub*m_sub, rank_target)
        V = rng.randn(m_sub*p_sub, rank_target)
        W = rng.randn(p_sub*n_sub, rank_target)
        
        # Verify
        is_valid = True
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    val = 0.0
                    for c in range(rank_target):
                        val += U[i*m_sub+j, c] * V[j*p_sub+k, c] * W[k*n_sub+i, c]
                    if abs(val - 1.0) > 1e-10:
                        is_valid = False
                        break
                if not is_valid: break
            if not is_valid: break
        if is_valid:
            return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    """
    Find a low-rank decomposition for 2x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Schoolbook solution as fallback: rank = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Split middle dimension 4 into 2+2
    # A = [A1, A2] (2x2 each)
    # B = [B1; B2] (2x5 each)
    # C = A1*B1 + A2*B2
    # We need to decompose 2x2 * 2x5 multiplication.
    
    # Try to find a low rank decomposition for 2x2 * 2x5
    decomp_225 = None
    if time.time() < deadline - 10:
        decomp_225 = _find_225_decomp(rank_target=17, max_attempts=500)
        
    if decomp_225 is None and time.time() < deadline - 5:
        decomp_225 = _find_225_decomp(rank_target=18, max_attempts=500)
        
    if decomp_225 is None and time.time() < deadline - 2:
        decomp_225 = _find_225_decomp(rank_target=19, max_attempts=200)

    if decomp_225 is not None:
        U_225, V_225, W_225 = decomp_225
        r_225 = len(U_225[0])
        
        # Construct full U, V, W
        # U_full is 8 x (2 * r_225)
        # V_full is 20 x (2 * r_225)
        # W_full is 10 x (2 * r_225)
        
        U_full = [[0] * (2 * r_225) for _ in range(n * m)]
        V_full = [[0] * (2 * r_225) for _ in range(m * p)]
        W_full = [[0] * (2 * r_225) for _ in range(p * n)]
        
        # Part 1: A1 * B1
        # A1 is rows 0,1 of A (cols 0,1) -> indices 0,1,2,3 in U_flat (i*m+j)
        # B1 is rows 0,1 of B (cols 0..4) -> indices 0..9 in V_flat (j*p+k)
        # Output C1 is 2x5 -> indices 0..9 in W_flat (k*n+i)
        
        for c in range(r_225):
            # U mapping
            # U_225[i*m_sub+j, c] corresponds to A1[i,j]
            # In U_full, A1[i,j] is at index i*4 + j
            for i in range(2):
                for j in range(2):
                    U_full[i*4 + j][c] = U_225[i*2 + j][c]
            
            # V mapping
            # V_225[j*m_sub+k, c] corresponds to B1[j,k]
            # In V_full, B1[j,k] is at index j*5 + k
            for j in range(2):
                for k in range(5):
                    V_full[j*5 + k][c] = V_225[j*2 + k][c]
            
            # W mapping
            # W_225[k*n_sub+i, c] corresponds to C1[k,i]
            # In W_full, C1[k,i] is at index k*2 + i
            for k in range(5):
                for i in range(2):
                    W_full[k*2 + i][c] = W_225[k*2 + i][c]
        
        # Part 2: A2 * B2
        # A2 is rows 0,1 of A (cols 2,3) -> indices 2,3,6,7 in U_flat
        # B2 is rows 2,3 of B (cols 0..4) -> indices 10..19 in V_flat
        # Output C2 is 2x5 -> indices 0..9 in W_flat (added to C1)
        
        offset = r_225
        for c in range(r_225):
            # U mapping
            for i in range(2):
                for j in range(2):
                    U_full[i*4 + (j+2)][offset + c] = U_225[i*2 + j][c]
            
            # V mapping
            for j in range(2):
                for k in range(5):
                    V_full[(j+2)*5 + k][offset + c] = V_225[j*2 + k][c]
            
            # W mapping
            for k in range(5):
                for i in range(2):
                    W_full[k*2 + i][offset + c] = W_225[k*2 + i][c]
        
        # Verify the full decomposition
        if _verify(U_full, V_full, W_full):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = 2 * r_225

    # If the 2+2 split didn't work or wasn't found, try 2+1+1 split
    # This requires 2x2*2x5 (rank r1) and 2x1*2x5 (rank r2)
    # 2x1*2x5 is just outer product, rank 10 (schoolbook)
    # If we found 2x2*2x5 with rank 17, total is 17 + 10 + 10 = 37
    # This is worse than 34, so we only use this if 2+2 fails.
    
    if best_rank == rank_school and time.time() < deadline:
        # Fallback to a known rank 37 construction if 2+2 failed
        # We can construct 2x1*2x5 with rank 10 easily
        # And use the 2x2*2x5 found above if it exists, else schoolbook 20
        if decomp_225 is not None:
            U_225, V_225, W_225 = decomp_225
            r_225 = len(U_225[0])
            
            # We need to add two 2x1*2x5 blocks
            # Total rank = r_225 + 10 + 10
            # This is only better than 40 if r_225 < 20
            if r_225 + 20 < best_rank:
                U_full = [[0] * (r_225 + 20) for _ in range(n * m)]
                V_full = [[0] * (r_225 + 20) for _ in range(m * p)]
                W_full = [[0] * (r_225 + 20) for _ in range(p * n)]
                
                # Block 1: A1 (2x2) * B1 (2x5)
                for c in range(r_225):
                    for i in range(2):
                        for j in range(2):
                            U_full[i*4 + j][c] = U_225[i*2 + j][c]
                    for j in range(2):
                        for k in range(5):
                            V_full[j*5 + k][c] = V_225[j*2 + k][c]
                    for k in range(5):
                        for i in range(2):
                            W_full[k*2 + i][c] = W_225[k*2 + i][c]
                
                # Block 2: A2 (2x1) * B2 (1x5)
                # A2 is col 2 of A. B2 is row 2 of B.
                # A2[i] = A[i,2]. B2[k] = B[2,k].
                # Product C2[i,k] = A[i,2] * B[2,k].
                # Rank 10 decomposition:
                # U[i*4+2, 10+c] = A[i,2] coeff?
                # Actually, for 2x1*1x5, we can use rank 5?
                # No, rank 10 is safe.
                # Let's use rank 5?
                # C2[i,k] = u_i v_k.
                # We can set U[i*4+2, 10+k] = 1, V[2*5+k, 10+k] = 1, W[k*2+i, 10+k] = 1?
                # No, that's not right.
                # For 2x1*1x5, the tensor is T[i,k] = u_i v_k.
                # This is a rank-1 tensor in the output space?
                # No, the bilinear form is sum_c U[i,c] V[k,c] W[i,k,c]?
                # Indices: U[2x1], V[1x5], W[5x2].
                # T[i,k] = sum_c U[i,c] V[k,c] W[k,i,c].
                # If we set c=k (0..4):
                # U[i,k] = 1, V[k,k] = 1, W[k,i,k] = 1?
                # Then sum is 1.
                # So rank 5 is sufficient for 2x1*1x5.
                
                r_215 = 5
                # Update total rank
                total_rank = r_225 + 2 * r_215
                if total_rank < best_rank:
                     # Re-allocate
                     U_full = [[0] * total_rank for _ in range(n * m)]
                     V_full = [[0] * total_rank for _ in range(m * p)]
                     W_full = [[0] * total_rank for _ in range(p * n)]
                     
                     # Copy Block 1
                     for c in range(r_225):
                        for i in range(2):
                            for j in range(2):
                                U_full[i*4 + j][c] = U_225[i*2 + j][c]
                        for j in range(2):
                            for k in range(5):
                                V_full[j*5 + k][c] = V_225[j*2 + k][c]
                        for k in range(5):
                            for i in range(2):
                                W_full[k*2 + i][c] = W_225[k*2 + i][c]
                     
                     # Block 2: A2 (col 2) * B2 (row 2)
                     # U indices: i*4+2
                     # V indices: 2*5+k = 10+k
                     # W indices: k*2+i
                     offset2 = r_225
                     for k in range(5):
                         U_full[0*4+2][offset2+k] = 1
                         U_full[1*4+2][offset2+k] = 1
                         V_full[2*5+k][offset2+k] = 1
                         W_full[k*2+0][offset2+k] = 1
                         W_full[k*2+1][offset2+k] = 1
                     
                     # Block 3: A3 (col 3) * B3 (row 3)
                     # U indices: i*4+3
                     # V indices: 3*5+k = 15+k
                     # W indices: k*2+i
                     offset3 = r_225 + r_215
                     for k in range(5):
                         U_full[0*4+3][offset3+k] = 1
                         U_full[1*4+3][offset3+k] = 1
                         V_full[3*5+k][offset3+k] = 1
                         W_full[k*2+0][offset3+k] = 1
                         W_full[k*2+1][offset3+k] = 1
                     
                     if _verify(U_full, V_full, W_full):
                         best_U, best_V, best_W = U_full, V_full, W_full
                         best_rank = total_rank

    # Return the best result found
    U_out = [[round(val, 10) for val in row] for row in best_U]
    V_out = [[round(val, 10) for val in row] for row in best_V]
    W_out = [[round(val, 10) for val in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
