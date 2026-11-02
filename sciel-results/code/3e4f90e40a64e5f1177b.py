"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check some invalid entries (spot check)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets using Strassen.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    U_full = np.zeros((n*m, 7))
    V_full = np.zeros((m*p, 7))
    W_full = np.zeros((p*n, 7))
    
    a_map = [offset_a, offset_a + 1, offset_a + m, offset_a + m + 1]
    b_map = [offset_b, offset_b + 1, offset_b + p, offset_b + p + 1]
    c_map = [offset_c, offset_c + 1, offset_c + n, offset_c + n + 1]
    
    for r in range(7):
        for local_idx in range(4):
            if U[local_idx, r] != 0:
                global_idx = a_map[local_idx]
                if 0 <= global_idx < n*m:
                    U_full[global_idx, r] = U[local_idx, r]
            if V[local_idx, r] != 0:
                global_idx = b_map[local_idx]
                if 0 <= global_idx < m*p:
                    V_full[global_idx, r] = V[local_idx, r]
            if W[local_idx, r] != 0:
                global_idx = c_map[local_idx]
                if 0 <= global_idx < p*n:
                    W_full[global_idx, r] = W[local_idx, r]
    
    return U_full, V_full, W_full

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    Rank-2 decomposition: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    """
    U_full = np.zeros((n*m, 2))
    V_full = np.zeros((m*p, 2))
    W_full = np.zeros((p*n, 2))
    
    # Term r=0: A[0,0] * B[0,:] -> C[0,:]
    U_full[offset_a, 0] = 1
    V_full[offset_b, 0] = 1
    V_full[offset_b + p, 0] = 1
    W_full[offset_c, 0] = 1
    W_full[offset_c + n, 0] = 1
    
    # Term r=1: A[0,1] * B[1,:] -> C[0,:]
    U_full[offset_a + 1, 1] = 1
    V_full[offset_b + 1, 1] = 1
    V_full[offset_b + p + 1, 1] = 1
    W_full[offset_c, 1] = 1
    W_full[offset_c + n, 1] = 1
    
    return U_full, V_full, W_full

def _dot_1x4x4(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x4x4 block at given offsets.
    Rank-4 decomposition: C[0,k] = sum_j A[0,j]*B[j,k]
    Each term r=j: U[offset_a+j, j]=1, V[offset_b+j, j]=1, W[offset_c+k, j]=1 for all k
    """
    U_full = np.zeros((n*m, 4))
    V_full = np.zeros((m*p, 4))
    W_full = np.zeros((p*n, 4))
    
    for j in range(4):
        U_full[offset_a + j, j] = 1
        V_full[offset_b + j, j] = 1
        for k in range(4):
            W_full[offset_c + k, j] = 1
    
    return U_full, V_full, W_full

def _try_strassen_blocks():
    """
    Construct a block-based Strassen decomposition for 3x4x8.
    Total rank: 2*14 + 2*4 = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 16, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    U_b, V_b, W_b = _strassen_2x2x2(0, 4, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 16, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    U_b, V_b, W_b = _dot_1x2x2(8, 4, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return _schoolbook(), n * m * p

def _try_optimized_blocks():
    """
    Try to reduce rank by sharing Strassen factors across blocks.
    
    Key insight: When two 2x2x2 blocks share the same A submatrix (e.g., A[0:2, 0:2])
    but have different B submatrices, we can potentially merge their Strassen decompositions.
    
    For A[0:2, 0:2] * B[0:2, 0:4] (2x2x4):
      This is two 2x2x2 blocks: A[0:2,0:2]*B[0:2,0:2] and A[0:2,0:2]*B[0:2,2:4]
      Each uses 7 rank, total 14.
      
    Alternative: Use a 2x2x4 decomposition directly.
      C[0:2, 0:4] = A[0:2,0:2] * B[0:2,0:4]
      
      We can write:
      C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
      
      For fixed i, k: this is a 2-term dot product.
      
      One approach: For each i in {0,1}, for each j in {0,1}:
        term (i,j): A[i,j] * B[j, :] -> contributes to C[i, :]
      
      That's 4 terms of rank 1 each? No, each term A[i,j]*B[j,:] is rank 1 in the tensor sense
      only if we fix i. Actually:
      
      C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
      
      For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
      For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
      
      We can decompose as:
      r0: A[0,0] * B[0,0:4] -> C[0, 0:4]
      r1: A[0,1] * B[1,0:4] -> C[0, 0:4]
      r2: A[1,0] * B[0,0:4] -> C[1, 0:4]
      r3: A[1,1] * B[1,0:4] -> C[1, 0:4]
      
      That's rank 4 for a 2x2x4 block! Much better than 14.
      
      Wait, let me verify. For a 2x2x4 tensor:
      T[i,j,k] = delta_{i,j} for the multiplication tensor? No.
      
      The matrix multiplication tensor for 2x2x4:
      T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B produce (i,k) entry of C.
      So T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,3} such that... 
      
      Actually the tensor is: T[i, j, k] = 1 if j is the shared index, i.e., 
      T[i, j, k] = 1 for all valid (i,j,k) where j is the inner product index.
      
      For 2x2x4: A is 2x2, B is 2x4, C is 2x4.
      C[i,k] = sum_j A[i,j] B[j,k]
      
      The tensor T has entries T[i, j, k] = 1 for all i in {0,1}, j in {0,1}, k in {0,3}.
      That's 2*2*4 = 16 entries, all 1.
      
      Rank of this tensor: 
      We can write T = sum_{r=0}^{3} u_r v_r w_r where:
      u_0[i] = 1 if i=0, j=0... 
      
      Actually, let's think of it as:
      T[i,j,k] = sum_{r} U[i,r] V[j,r] W[k,r]
      
      One decomposition:
      r=0: U[0,0]=1, V[0,0]=1, W[0:4,0]=1  -> contributes to T[0,0,k] for all k
      r=1: U[0,1]=1, V[1,1]=1, W[0:4,1]=1  -> contributes to T[0,1,k] for all k
      r=2: U[1,2]=1, V[0,2]=1, W[0:4,2]=1  -> contributes to T[1,0,k] for all k
      r=3: U[1,3]=1, V[1,3]=1, W[0:4,3]=1  -> contributes to T[1,1,k] for all k
      
      So T[i,j,k] = 1 if (i,j) matches (0,0) or (0,1) or (1,0) or (1,1), which is always true.
      And W[k,r] = 1 for all k, so the sum over r gives the correct value.
      
      Yes! Rank 4 for 2x2x4.
      
      Similarly, 2x2x2 has rank 4 with this method, but Strassen gives rank 7 which is worse.
      Wait, that can't be right. The 2x2x2 multiplication tensor has rank 7 (Strassen), not 4.
      
      Let me re-examine. For 2x2x2:
      T[i,j,k] = 1 if j is the shared index.
      i in {0,1}, j in {0,1}, k in {0,1}.
      
      T[0,0,0] = 1 (A[0,0]*B[0,0] -> C[0,0])
      T[0,0,1] = 1 (A[0,0]*B[0,1] -> C[0,1])
      T[0,1,0] = 1 (A[0,1]*B[1,0] -> C[0,0])
      T[0,1,1] = 1 (A[0,1]*B[1,1] -> C[0,1])
      T[1,0,0] = 1 (A[1,0]*B[0,0] -> C[1,0])
      T[1,0,1] = 1 (A[1,0]*B[0,1] -> C[1,1])
      T[1,1,0] = 1 (A[1,1]*B[1,0] -> C[1,0])
      T[1,1,1] = 1 (A[1,1]*B[1,1] -> C[1,1])
      
      All 8 entries are 1.
      
      My decomposition above would give:
      r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1
      r=1: U[0,1]=1, V[1,1]=1, W[0,1]=1, W[1,1]=1
      r=2: U[1,2]=1, V[0,2]=1, W[0,2]=1, W[1,2]=1
      r=3: U[1,3]=1, V[1,3]=1, W[0,3]=1, W[1,3]=1
      
      T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + ... = 1*1*1 + 0 + 0 + 0 = 1 ✓
      T[0,0,1] = U[0,0]*V[0,0]*W[1,0] + ... = 1*1*1 + 0 + 0 + 0 = 1 ✓
      T[0,1,0] = U[0,1]*V[1,1]*W[0,1] + ... = 1*1*1 + 0 + 0 + 0 = 1 ✓
      T[0,1,1] = 1*1*1 = 1 ✓
      T[1,0,0] = U[1,2]*V[0,2]*W[0,2] = 1*1*1 = 1 ✓
      T[1,0,1] = 1*1*1 = 1 ✓
      T[1,1,0] = U[1,3]*V[1,3]*W[0,3] = 1*1*1 = 1 ✓
      T[1,1,1] = 1*1*1 = 1 ✓
      
      So rank 4 works for 2x2x2?! But Strassen says rank 7...
      
      Wait, I'm confusing the tensor. The matrix multiplication tensor is NOT all-ones.
      The tensor T_M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
      That means T_M[i,j,k] = 1 for ALL i,j,k in the valid range, because every (i,j,k) triple
      corresponds to a valid term in the matrix product.
      
      So yes, the 2x2x2 multiplication tensor is the all-ones tensor of size 2x2x2, and its
      rank is indeed 4 (not 7). Strassen's rank 7 is for the 2x2x2 matrix multiplication, but
      that's a different tensor!
      
      Actually no. Let me reconsider. The matrix multiplication tensor for multiplying an
      m x n matrix by an n x p matrix is a tensor of size (m*n) x (n*p) x (m*p) where
      T[(i,j), (j,k), (i,k)] = 1 and all other entries are 0.
      
      For 2x2x2: A is 2x2, B is 2x2, C is 2x2.
      T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', and k=k'.
      So T[(0,0), (0,0), (0,0)] = 1
      T[(0,0), (0,1), (0,1)] = 1
      T[(0,1), (1,0), (0,0)] = 1
      T[(0,1), (1,1), (0,1)] = 1
      T[(1,0), (0,0), (1,0)] = 1
      T[(1,0), (0,1), (1,1)] = 1
      T[(1,1), (1,0), (1,0)] = 1
      T[(1,1), (1,1), (1,1)] = 1
      
      All other entries are 0. So the tensor has 8 non-zero entries, not all 8 entries of a
      2x2x2 tensor. It's a sparse tensor.
      
      In my earlier analysis, I was treating it as if T[i,j,k] = 1 for all i,j,k, but that's
      wrong. The indices are (i,j), (j,k), (i,k) - the j in the first and second indices must
      match, and the i in the first and third must match, and the k in the second and third must match.
      
      So the tensor is:
      T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i, and 0 otherwise.
      
      For 2x2x2:
      a in {0,1,2,3} = (i,j) with i,j in {0,1}
      b in {0,1,2,3} = (j,k) with j,k in {0,1}
      c in {0,1,2,3} = (k,i) with k,i in {0,1}
      
      T[0,0,0] = 1  (i=0,j=0,k=0)
      T[0,1,1] = 1  (i=0,j=0,k=1)
      T[1,2,0] = 1  (i=0,j=1,k=0)
      T[1,3,1] = 1  (i=0,j=1,k=1)
      T[2,0,2] = 1  (i=1,j=0,k=0)
      T[2,1,3] = 1  (i=1,j=0,k=1)
      T[3,2,2] = 1  (i=1,j=1,k=0)
      T[3,3,3] = 1  (i=1,j=1,k=1)
      
      This is NOT the all-ones tensor. Its rank is 7 (Strassen).
      
      So my earlier "rank 4" decomposition was incorrect because I was decomposing the wrong tensor.
      
      OK so the Strassen decomposition with rank 7 is correct for 2x2x2.
      
      Now, for the 3x4x8 problem, the current approach gives rank 36. Can we do better?
      
      One idea: instead of decomposing into 2x2x2 blocks, use a different blocking strategy.
      
      For example, decompose A (3x4) as a sum of rank-1 matrices, or use a different
      factorization approach.
      
      Actually, let's think about this differently. The matrix multiplication tensor for
      3x4x8 has a known rank. Let me try to find a better decomposition.
      
      One approach: use the fact that 3 = 2+1 and 8 = 4+4, and decompose accordingly.
      
      C = A * B where A is 3x4, B is 4x8, C is 3x8.
      
      Split A = [A1; A2] where A1 is 2x4, A2 is 1x4.
      Split B = [B1 B2] where B1 is 4x4, B2 is 4x4.
      
      C = [A1*B1  A1*B2]
          [A2*B1  A2*B2]
      
      A1*B1 is 2x4x4, A1*B2 is 2x4x4, A2*B1 is 1x4x4, A2*B2 is 1x4x4.
      
      For 2x4x4: split into two 2x2x2 blocks, each rank 7, total 14.
      For 1x4x4: split into two 1x2x2 blocks, each rank 2, total 4.
      
      Total: 2*14 + 2*4 = 28 + 8 = 36.
      
      Can we do better for 2x4x4? 
      
      2x4x4: A is 2x4, B is 4x4, C is 2x4.
      C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
      
      One decomposition:
      For each j in {0,1,2,3}:
        term j: A[:,j] * B[j,:] -> C
        This is rank 1 in the sense that U[i,j] = A[i,j], V[j,k] = B[j,k], W[i,k] = C[i,k].
        But we need a fixed decomposition, not dependent on the matrix values.
      
      The tensor for 2x4x4:
      T[(i,j), (j,k), (i,k)] = 1
      i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
      
      Number of non-zero entries: 2*4*4 = 32.
      
      Rank of this tensor: at least max(2*4, 4*4, 2*4) = 16.
      Strassen-like: split into two 2x2x2 blocks (j=0,1 and j=2,3), each rank 7, total 14.
      
      Can we do better than 14 for 2x4x4?
      
      Actually, I recall that the rank of the 2xn x n x 2n matrix multiplication tensor is
      related to the rank of the 2x2x2 tensor. For 2x4x4, it might be possible to achieve
      rank less than 14.
      
      Let me try a different approach: use the fact that we can share factors.
      
      For 2x4x4:
      C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
      C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
      
      One decomposition:
      r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
      r=1: U[0,1]=1, V[1,1]=1, W[0,1]=1, W[1,1]=1, W[2,1]=1, W[3,1]=1
      r=2: U[0,2]=1, V[2,2]=1, W[0,2]=1, W[1,2]=1, W[2,2]=1, W[3,2]=1
      r=3: U[0,3]=1, V[3,3]=1, W[0,3]=1, W[1,3]=1, W[2,3]=1, W[3,3]=1
      r=4: U[1,4]=1, V[0,4]=1, W[0,4]=1, W[1,4]=1, W[2,4]=1, W[3,4]=1
      r=5: U[1,5]=1, V[1,5]=1, W[0,5]=1, W[1,5]=1, W[2,5]=1, W[3,5]=1
      r=6: U[1,6]=1, V[2,6]=1, W[0,6]=1, W[1,6]=1, W[2,6]=1, W[3,6]=1
      r=7: U[1,7]=1, V[3,7]=1, W[0,7]=1, W[1,7]=1, W[2,7]=1, W[3,7]=1
      
      That's rank 8 for 2x4x4! Better than 14.
      
      Wait, let me verify. T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
      
      T[(0,0), (0,0), (0,0)] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
      T[(0,0), (0,1), (0,1)] = U[0,0]*V[0,1]*W[0,1] = 1*0*1 = 0 ✗
      
      That's wrong. V[0,1] should be 0, but we need T[(0,0), (0,1), (0,1)] = 1.
      
      The issue is that B[0,1] is at index (j=0, k=1), which is b = 0*4+1 = 1.
      And C[0,1] is at index (k=1, i=0), which is c = 1*2+0 = 2.
      
      So T[(0,0), (0,1), (0,1)] means a=0, b=1, c=2.
      T[0,1,2] = 1.
      
      In my decomposition:
      T[0,1,2] = sum_r U[0,r]*V[1,r]*W[2,r]
      = U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[2,1] + ...
      = 1*0*1 + 1*1*1 + 0 + 0 + ...
      = 1 ✓
      
      Let me re-examine. a=0 is (i=0,j=0), b=1 is (j=0,k=1), c=2 is (k=1,i=0).
      T[a,b,c] = 1 if the indices are consistent: i_a = i_c, j_a = j_b, k_b = k_c.
      i_a=0, i_c=0 ✓
      j_a=0, j_b=0 ✓
      k_b=1, k_c=1 ✓
      So T[0,1,2] = 1.
      
      In my decomposition:
      sum_r U[0,r]*V[1,r]*W[2,r]
      r=0: U[0,0]=1, V[1,0]=0, W[2,0]=1 -> 0
      r=1: U[0,1]=1, V[1,1]=1, W[2,1]=1 -> 1
      r=2: U[0,2]=1, V[1,2]=0, W[2,2]=1 -> 0
      r=3: U[0,3]=1, V[1,3]=0, W[2,3]=1 -> 0
      r=4: U[0,4]=0, V[1,4]=0, W[2,4]=1 -> 0
      r=5: U[0,5]=0, V[1,5]=1, W[2,5]=1 -> 0
      r=6: U[0,6]=0, V[1,6]=0, W[2,6]=1 -> 0
      r=7: U[0,7]=0, V[1,7]=0, W[2,7]=1 -> 0
      Sum = 1 ✓
      
      Let me check another entry: T[(0,1), (1,0), (0,0)] = T[1,2,0]
      a=1 is (i=0,j=1), b=2 is (j=1,k=0), c=0 is (k=0,i=0).
      i_a=0, i_c=0 ✓
      j_a=1, j_b=1 ✓
      k_b=0, k_c=0 ✓
      T[1,2,0] = 1.
      
      sum_r U[1,r]*V[2,r]*W[0,r]
      r=0: U[1,0]=0, V[2,0]=0, W[0,0]=1 -> 0
      r=1: U[1,1]=0, V[2,1]=0, W[0,1]=1 -> 0
      r=2: U[1,2]=0, V[2,2]=1, W[0,2]=1 -> 0
      r=3: U[1,3]=0, V[2,3]=0, W[0,3]=1 -> 0
      r=4: U[1,4]=1, V[2,4]=0, W[0,4]=1 -> 0
      r=5: U[1,5]=1, V[2,5]=0, W[0,5]=1 -> 0
      r=6: U[1,6]=1, V[2,6]=1, W[0,6]=1 -> 1
      r=7: U[1,7]=1, V[2,7]=0, W[0,7]=1 -> 0
      Sum = 1 ✓
      
      Let me check a zero entry: T[(0,0), (1,0), (0,0)] = T[0,2,0]
      a=0 is (i=0,j=0), b=2 is (j=1,k=0), c=0 is (k=0,i=0).
      i_a=0, i_c=0 ✓
      j_a=0, j_b=1 ✗
      So T[0,2,0] = 0.
      
      sum_r U[0,r]*V[2,r]*W[0,r]
      r=0: 1*0*1 = 0
      r=1: 1*0*1 = 0
      r=2: 1*1*1 = 1 ✗
      
      That's wrong! T[0,2,0] should be 0 but my decomposition gives 1.
      
      The issue: V[2,2] = 1, but b=2 corresponds to (j=1, k=0), and r=2 has V[2,2]=1 which means
      V[j=1, k=0, r=2] = 1. But U[0,2] = 1 means U[i=0, j=0, r=2] = 1. So the j indices don't match.
      
      The problem is that my decomposition doesn't enforce the j-index matching. I need to be more careful.
      
      Let me redesign. For 2x4x4:
      T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
      
      Decomposition:
      For each (i, j, k) triple, we need a term that contributes to T[(i,j), (j,k), (i,k)].
      
      One term per (i,j,k):
      r_{i,j,k}: U[i, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
      
      That gives rank 2*4*4 = 32, which is the schoolbook.
      
      To reduce rank, we need to share terms. The key is that for a fixed (i,k), the sum over j
      is linear, so we can potentially combine terms with the same (i,k).
      
      Actually, the standard approach is to use the Strassen-like decomposition for the 2x2x2 blocks.
      
      Let me just go with the rank 36 decomposition and see if I can find a better one through
      a different blocking.
      
      Alternative: decompose A (3x4) as a sum of 3 rank-1 matrices (trivially), and use that.
      
      Actually, let me try a different approach: use the fact that 3x4x8 can be decomposed as
      (2+1)x4x(4+4), and use the 2x4x4 rank 14 and 1x4x4 rank 4 as before, but try to share
      factors between the two 2x4x4 blocks.
      
      The two 2x4x4 blocks are:
      Block 1: A[0:2, :] * B[:, 0:4]
      Block 2: A[0:2, :] * B[:, 4:8]
      
      They share the same A submatrix A[0:2, :]. Can we share any Strassen factors?
      
      In the Strassen decomposition of 2x2x2, the U factors depend on A, the V factors on B,
      and the W factors on C. Since A is the same for both blocks, the U factors could potentially
      be shared.
      
      But the Strassen decomposition is not linear in a way that allows direct sharing. Each
      2x2x2 block has its own set of 7 U, V, W factors.
      
      I think the rank 36 is hard to beat with simple blocking. Let me just implement the rank 36
      decomposition and return it.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 16, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    U_b, V_b, W_b = _strassen_2x2x2(0, 4, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 16, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    U_b, V_b, W_b = _dot_1x2x2(8, 4, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return _schoolbook(), n * m * p

def construct():
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 36)
    try:
        result, rank = _try_strassen_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Try the optimized blocks (same rank 36 for now)
    try:
        result, rank = _try_optimized_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
