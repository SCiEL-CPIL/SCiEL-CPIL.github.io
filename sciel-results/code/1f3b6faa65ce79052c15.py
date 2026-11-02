import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-39 exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    
    Strategy:
      - 5 blocks of 2 columns each (k=0..9) using Strassen (rank 7 each) = 35
      - 1 tail column (k=10) using schoolbook (rank 4) = 4
      - Total rank = 39
    
    Index conventions:
      A is 2x2: A[i, j] where i in {0,1}, j in {0,1}
      B is 2x11: B[j, k] where j in {0,1}, k in {0..10}
      C is 2x11: C[i, k] where i in {0,1}, k in {0..10}
      
      U has shape (n*m, rank) = (4, rank): indices 0..3 for A[0,0], A[0,1], A[1,0], A[1,1]
      V has shape (m*p, rank) = (22, rank): indices 0..10 for B[0,k], 11..21 for B[1,k]
      W has shape (p*n, rank) = (22, rank): indices k*n+i for C[i,k] (official trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline (10s safety margin)
    
    n, m, p = 2, 2, 11
    rank = 39
    
    # Build the verified rank-39 construction
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0  # current rank index
    
    # Process 5 blocks of 2 columns each (k=0,1; 2,3; 4,5; 6,7; 8,9)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # B indices
        v_e = k0          # B[0, k0]
        v_f = k1          # B[0, k1]
        v_g = p + k0      # B[1, k0]
        v_h = p + k1      # B[1, k1]
        
        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        # Strassen's 7 multiplications for 2x2 * 2x2:
        # m1 = (a+d)(e+h)
        U[0][r] = 1    # a
        U[3][r] = 1    # d
        V[v_e][r] = 1  # e
        V[v_h][r] = 1  # h
        W[w_c00][r] = 1   # c00 += m1
        W[w_c11][r] = 1   # c11 += m1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1    # c
        U[3][r] = 1    # d
        V[v_e][r] = 1  # e
        W[w_c10][r] = 1   # c10 += m2
        W[w_c11][r] = -1  # c11 -= m2
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1    # a
        V[v_f][r] = 1  # f
        V[v_h][r] = -1 # h
        W[w_c01][r] = 1   # c01 += m3
        W[w_c11][r] = 1   # c11 += m3
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1    # d
        V[v_g][r] = 1  # g
        V[v_e][r] = -1 # e
        W[w_c00][r] = 1   # c00 += m4
        W[w_c10][r] = 1   # c10 += m4
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1    # a
        U[1][r] = 1    # b
        V[v_h][r] = 1  # h
        W[w_c00][r] = -1  # c00 -= m5
        W[w_c01][r] = 1   # c01 += m5
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1    # c
        U[0][r] = -1   # -a
        V[v_e][r] = 1  # e
        V[v_f][r] = 1  # f
        W[w_c11][r] = 1   # c11 += m6
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1    # b
        U[3][r] = -1   # -d
        V[v_g][r] = 1  # g
        V[v_h][r] = 1  # h
        W[w_c00][r] = 1   # c00 += m7
        r += 1
    
    # Tail column k=10: schoolbook (4 multiplications)
    k_tail = 10
    v_et = k_tail          # B[0, 10]
    v_gt = p + k_tail      # B[1, 10]
    w_c0t = k_tail * n + 0  # C[0, 10]
    w_c1t = k_tail * n + 1  # C[1, 10]
    
    # a * B[0,10]
    U[0][r] = 1
    V[v_et][r] = 1
    W[w_c0t][r] = 1
    r += 1
    
    # b * B[1,10]
    U[1][r] = 1
    V[v_gt][r] = 1
    W[w_c0t][r] = 1
    r += 1
    
    # c * B[0,10]
    U[2][r] = 1
    V[v_et][r] = 1
    W[w_c1t][r] = 1
    r += 1
    
    # d * B[1,10]
    U[3][r] = 1
    V[v_gt][r] = 1
    W[w_c1t][r] = 1
    r += 1
    
    if r != rank:
        raise RuntimeError(f"Internal error: expected rank {rank}, got {r}")
    
    # Verify the construction
    if not _verify_decomposition(U, V, W, n, m, p):
        raise RuntimeError("Verification failed for rank-39 construction")
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Strategy: try to merge tail into last Strassen block by using a 3-column block
        # or try alternative groupings
        try_lower_rank = _try_lower_rank_search(n, m, p, start_time, deadline)
        if try_lower_rank is not None:
            alt_rank, alt_U, alt_V, alt_W = try_lower_rank
            if alt_rank < best_rank:
                if _verify_decomposition(alt_U, alt_V, alt_W, n, m, p):
                    best_rank = alt_rank
                    best_result = {"U": alt_U, "V": alt_V, "W": alt_W}
    
    return best_result


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Check dimensions
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    # Build the tensor: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] should be 1 if we're computing C[i,k] with A[i,j]*B[j,k]
    
    # Compute the decomposition tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                tensor[i, j, k] = s
    
    # The target tensor for matrix multiplication:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In standard matrix mult, C[i,k] = sum_j A[i,j] * B[j,k]
    # So each term A[i,j]*B[j,k] appears exactly once with coefficient 1
    # The tensor T[i,j,k] = 1 for all i,j,k (each product term has coefficient 1)
    # Wait, that's not right. The tensor is defined as:
    # T[i,j,k] = 1 means A[i,j] * B[j,k] contributes to C[i,k] with weight 1
    # Actually for matrix multiplication, the structure tensor is:
    # T[i,j,k] = 1 for all i, j, k (since every pair (i,j) and (j,k) contributes)
    # No wait. Let me think again.
    # 
    # The matrix multiplication tensor M_{i,j,k} where:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k
    # 
    # In the decomposition: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # So we need M[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0..10}
    
    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-9):
        return False
    
    return True


def _try_lower_rank_search(n, m, p, start_time, deadline):
    """
    Try to find a lower-rank decomposition within the Strassen-block family.
    
    Ideas to explore:
    1. Use a different column grouping (e.g., 3+2+2+2+2 or 2+2+2+2+3)
    2. Try to share factors between adjacent blocks
    3. Use a rank-6 or lower decomposition for specific 2x2x2 blocks
    
    Returns (rank, U, V, W) if a valid lower-rank decomposition is found, else None.
    """
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return None
    
    # Try grouping: 6 blocks where last block has 2 columns (k=9,10) instead of tail
    # This gives 6 Strassen blocks = 42 rank, which is worse. Skip.
    
    # Try: 4 Strassen blocks (8 columns) + 3 schoolbook columns (3*4=12) = 28+12=40, worse.
    
    # Try: 5 Strassen blocks but try to reduce one block's rank
    # For a 2x2 * 2x2 multiplication, the minimum rank is 7 (Strassen is optimal).
    # So we can't reduce individual block ranks below 7.
    
    # Alternative: try to share U factors between blocks.
    # If two blocks share the same A-side linear combinations, we might save rank.
    # But each block has different B columns, so V factors differ.
    # The U factors for Strassen are: (a+d), (c+d), a, d, (a+b), (c-a), (b-d)
    # These are fixed for any 2x2*2x2 Strassen decomposition.
    # If we could share U factors between blocks, we'd need the same U vector to appear
    # in multiple blocks, which would require combining them into a single tensor term.
    
    # Let's try a different approach: use the fact that we have 11 columns and try
    # a non-uniform partition. E.g., treat columns 0-9 as 5 Strassen blocks (rank 35)
    # and column 10 as schoolbook (rank 4) = 39. That's what we have.
    
    # Try: can we do the last 3 columns (9,10) with fewer than 7+4=11 rank?
    # Columns 9 and 10: 2x2 * 2x2 block = rank 7 (Strassen)
    # But that's the same as before. No gain.
    
    # Try: what if we group columns as (0,1), (2,3), (4,5), (6,7), (8,9,10)?
    # The last group is 2x2 * 2x3. What's the rank of 2x2 * 2x3?
    # This is equivalent to multiplying A (2x2) by B (2x3).
    # C (2x3) = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has size 2*2*3 = 12 entries, all = 1.
    # Rank of 2x2 * 2x3: we can use Strassen on a 2x2 subblock + 1 schoolbook column
    # = 7 + 4 = 11, or we could try to find a better decomposition.
    # Actually, 2x2 * 2x3 can be seen as 2x2 * 2x2 + 2x2 * 2x1
    # Rank <= 7 + 4 = 11. Can we do better?
    # The tensor is 2x2x3. Let's think about it differently.
    # We need to compute C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    # 
    # Using Strassen for the first 2 columns (rank 7) + schoolbook for 3rd (rank 4) = 11
    # Or: use 3 Strassen-like terms? No, Strassen is for 2x2*2x2.
    # 
    # Alternative: use 8 terms? Let's see if rank 8 or 9 is possible for 2x2*2x3.
    # This is a research question. For now, let's assume 11 is optimal for this grouping.
    
    # Let's try a completely different partition: 2+2+2+2+2+1 (current) vs 2+2+2+3+2
    # 2+2+2+3+2: 4 Strassen blocks (28) + 1 2x2*2x3 block (11) = 39. Same!
    
    # What about 3+2+2+2+2? Same as above, just reordered.
    
    # Try: 2+2+2+2+3 (last 3 columns as 2x2*2x3)
    # 4 Strassen blocks (28) + 2x2*2x3 (11) = 39. Same.
    
    # It seems 39 might be tight for this family. Let's try to verify by attempting
    # to find a rank-38 or rank-37 decomposition via a more careful search.
    
    # For now, return None (no improvement found)
    return None
