"""Matrix multiplication 4x4 by 4x7: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition as (U, V, W) numpy int arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if s != 1:
                    return False
    return True

def _strassen_2x2_core():
    """
    Return the 7 rank-1 tensors for 2x2 * 2x2 matrix multiplication (Strassen).
    Each tensor is a 2x2x2 array with entries in {-1, 0, 1}.
    """
    # Strassen's 7 products for 2x2*2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each of the 7 rank-1 terms, we need U (2x2), V (2x2), W (2x2)
    # such that U[i,j]*V[j,k]*W[k,i] summed over the 7 terms gives the correct C[i,k]
    
    # The 7 terms as (U, V, W) pairs where each is a 2x2 matrix:
    # We encode each term as three 2x2 matrices.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U1: rows i=0,1; cols j=0,1. U1[i,j] = 1 if (i==0 and j==0) or (i==1 and j==1)
    # V1: V1[j,k] = 1 if (j==0 and k==0) or (j==1 and k==1)
    # W1: W1[k,i] = 1 if (k==0 and i==0) or (k==1 and i==1)
    # This gives C[0,0] += 1, C[1,1] += 1 (diagonal)
    
    # Actually, let me think more carefully. The tensor T[i,j,k] for matmul is 1 for all i,j,k.
    # We need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    # For 2x2*2x2, the Strassen decomposition gives:
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    
    # The 7 Strassen products and their contributions to C:
    # M1 = (A00+A11)(B00+B11): contributes to C00 (+1), C01 (0), C10 (0), C11 (+1)
    #   Wait, that's not right. Let me use the standard formulation.
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Mr, we need to express it as a sum of rank-1 terms U_r[i,j]*V_r[j,k]*W_r[k,i]
    # where i,j,k are in {0,1}.
    
    # M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is 4 rank-1 terms, but we want to combine them.
    # U1 = [[1,0],[0,1]] (diagonal), V1 = [[1,1],[0,1]]... no.
    
    # Actually, for the tensor decomposition, each rank-1 term is:
    # U_r is a 2x2 matrix (indices i,j), V_r is 2x2 (indices j,k), W_r is 2x2 (indices k,i)
    
    # Let me just enumerate the 7 Strassen terms as their U,V,W matrices.
    
    # M1 = (A00+A11)(B00+B11)
    # Contribution to C[i,k]: 
    #   C[0,0] += A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   C[0,1] += 0
    #   C[1,0] += 0  
    #   C[1,1] += A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Hmm, that's not quite right either. Let me be very explicit.
    
    # M1 = (A00+A11)(B00+B11) is a scalar. It contributes to C[0,0] and C[1,1] with +1 coefficient.
    # So for rank-1 decomposition:
    # We need U1, V1, W1 such that:
    # sum over r of U_r[i,j]*V_r[j,k]*W_r[k,i] gives the right coefficients.
    
    # For M1: coefficient of A[i,j]*B[j,k] in the contribution to C[i',k'] is:
    #   +1 when (i'=0,k'=0) and (i,j) in {(0,0),(0,1),(1,0),(1,1)}... no.
    
    # Let me use a different approach. The Strassen algorithm computes 7 scalar products.
    # Each scalar product Mr = sum_{j} (linear combo of A) * (linear combo of B)
    # Then C[i,k] = sum_r coeff[r][i,k] * Mr
    
    # For tensor rank decomposition, we need:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # where U_r[i,j] is the coefficient of A[i,j] in the r-th "left" factor,
    # V_r[j,k] is the coefficient of B[j,k] in the r-th "right" factor,
    # W_r[k,i] is the coefficient for output C[i,k] in the r-th term.
    
    # For Strassen M1 = (A00+A11)(B00+B11):
    # Left factor: A00 + A11, so U1[0,0]=1, U1[1,1]=1, others 0
    # Right factor: B00 + B11, so V1[0,0]=1, V1[1,1]=1, others 0
    # Output: M1 contributes +1 to C[0,0] and +1 to C[1,1]
    # So W1[0,0]=1, W1[1,1]=1, others 0
    
    # M2 = (A10+A11)B00:
    # U2[1,0]=1, U2[1,1]=1
    # V2[0,0]=1
    # W2[1,0]=1 (contributes to C[1,0])
    
    # M3 = A00(B10-B11):
    # U3[0,0]=1
    # V3[1,0]=1, V3[1,1]=-1
    # W3[0,1]=1 (contributes to C[0,1])
    
    # M4 = A11(B01-B00):
    # U4[1,1]=1
    # V4[0,1]=1, V4[0,0]=-1
    # W4[0,0]=1, W4[1,0]=1 (contributes to C[0,0] and C[1,0])
    
    # M5 = (A00+A01)B11:
    # U5[0,0]=1, U5[0,1]=1
    # V5[1,1]=1
    # W5[0,1]=1, W5[1,1]=-1 (contributes +1 to C[0,1], -1 to C[1,1])
    
    # M6 = (A10-A00)(B00+B01):
    # U6[1,0]=1, U6[0,0]=-1
    # V6[0,0]=1, V6[0,1]=1
    # W6[1,1]=1 (contributes to C[1,1])
    
    # M7 = (A01-A11)(B10+B11):
    # U7[0,1]=1, U7[1,1]=-1
    # V7[1,0]=1, V7[1,1]=1
    # W7[0,0]=1, W7[1,1]=-1 (contributes +1 to C[0,0], -1 to C[1,1])
    
    U_strassen = [
        np.array([[1,0],[0,1]]),  # M1
        np.array([[0,0],[1,1]]),  # M2
        np.array([[1,0],[0,0]]),  # M3
        np.array([[0,0],[0,1]]),  # M4
        np.array([[1,1],[0,0]]),  # M5
        np.array([[-1,0],[1,0]]), # M6
        np.array([[0,1],[-1,0]]), # M7
    ]
    V_strassen = [
        np.array([[1,0],[0,1]]),  # M1
        np.array([[1,0],[0,0]]),  # M2
        np.array([[0,0],[1,-1]]), # M3
        np.array([[-1,1],[0,0]]), # M4
        np.array([[0,0],[0,1]]),  # M5
        np.array([[1,1],[0,0]]),  # M6
        np.array([[0,0],[1,1]]),  # M7
    ]
    W_strassen = [
        np.array([[1,0],[0,1]]),  # M1: C[0,0]+=1, C[1,1]+=1
        np.array([[0,0],[1,0]]),  # M2: C[1,0]+=1
        np.array([[0,1],[0,0]]),  # M3: C[0,1]+=1
        np.array([[1,0],[1,0]]),  # M4: C[0,0]+=1, C[1,0]+=1
        np.array([[0,1],[-1,0]]), # M5: C[0,1]+=1, C[1,1]-=1
        np.array([[0,0],[0,1]]),  # M6: C[1,1]+=1
        np.array([[1,0],[0,-1]]), # M7: C[0,0]+=1, C[1,1]-=1
    ]
    
    return U_strassen, V_strassen, W_strassen

def _build_tensor_train_decomposition(n, m, p, deadline):
    """
    Build a decomposition using tensor train / hierarchical approach.
    
    Strategy: Split the problem into 2x2 blocks.
    For 4x4 * 4x7:
    - Split A (4x4) into four 2x2 blocks: A00, A01, A10, A11
    - Split B (4x7) into: B_top (2x7) and B_bot (2x7)
    - C (4x7) = [[A00*B_top + A01*B_bot, ...], [A10*B_top + A11*B_bot, ...]]
    
    Actually, let's think of it as:
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01, B02, B03, B04, B05, B06],
         [B10, B11, B12, B13, B14, B15, B16]]
    where B0* is the top row block (2x7) and B1* is bottom (2x7)
    
    C[0:2, :] = A00 @ B0 + A01 @ B1  (2x7)
    C[2:4, :] = A10 @ B0 + A11 @ B1  (2x7)
    
    For the top half: C_top = A00@B0 + A01@B1
    This is like computing two 2x2 * 2x7 products.
    
    We can use Strassen on the 2x2 * 2x2 part and extend to 2x7.
    
    Actually, a cleaner approach:
    Use the fact that M_{4,4,7} can be decomposed using the 2x2 Strassen kernel.
    
    Let's build the full 4x4*4x7 tensor and try to find a structured decomposition.
    
    Alternative: Use the "divide and conquer" approach:
    1. Split n=4 into n1=2, n2=2
    2. Split m=4 into m1=2, m2=2
    3. p=7 stays as is (or split into 3+4)
    
    The multiplication:
    C[0:2, :] = A[0:2, 0:2] @ B[0:2, :] + A[0:2, 2:4] @ B[2:4, :]
    C[2:4, :] = A[2:4, 0:2] @ B[0:2, :] + A[2:4, 2:4] @ B[2:4, :]
    
    For each 2x2 * 2x7 product, we can use a Strassen-like decomposition.
    2x2 * 2x7: This is equivalent to 2x2 * 2x7.
    We can think of B as 7 columns of 2x1 vectors.
    For each column, we need 2x2 * 2x1.
    
    2x2 * 2x1: C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
               C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    This has rank 4 (schoolbook). Can we do better?
    The tensor is 2x2x2 (i,j,k where k=0 only for 2x1). Actually it's 2x2x1.
    Rank of 2x2x1 matmul tensor is 4 (no known improvement).
    
    Hmm, so 2x2*2x7 has rank at least 7*4 = 28? No, that's not right.
    The rank of M_{2,2,7} is at most 7 * rank(M_{2,2,1}) = 7*4 = 28.
    But we can do better by sharing the A factors.
    
    Actually, M_{2,2,p} for any p: the rank is at most 2p (use each A[i,j] with all p B[j,k]).
    But we can also use Strassen-like tricks.
    
    For M_{2,2,7}, the known upper bound is... let me think.
    We can compute C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    This is 4 scalar-vector products, so rank <= 4*7 = 28? No.
    Each term A[i,j]*B[j,:] is a vector of length 7.
    The tensor M_{2,2,7} has entries T[i,j,k] = 1 for all i,j,k.
    Rank: We need sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    One decomposition: For each (i,j), use r corresponding to (i,j), with U_r[i,j]=1, V_r[j,k]=1 for all k, W_r[k,i]=1 for all k.
    That gives rank 4 (one for each i,j pair). Wait, that's not right.
    
    If U_r[i,j] = delta_{r,(i,j)} (1 if r corresponds to (i,j), 0 otherwise),
    V_r[j,k] = 1 for all k (when r corresponds to (i,j)),
    W_r[k,i] = 1 for all k (when r corresponds to (i,j)),
    Then sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1*1*1 = 1 for the matching (i,j) term.
    But we only have one term, so we get 1. 
    
    Wait, that means rank 4 for M_{2,2,7}? Let me verify.
    U has shape (2*2, 4) = (4, 4). U[r, i*m+j] where r is the rank index.
    Actually, U has shape (n*m, rank) = (4, rank).
    
    For r=0: U[0,0]=1 (i=0,j=0), V[0,k]=1 for all k, W[k,0]=1 for all k.
    For r=1: U[1,1]=1 (i=0,j=1), V[1,k]=1 for all k, W[k,0]=1 for all k.
    For r=2: U[2,2]=1 (i=1,j=0), V[0,k]=1 for all k, W[k,1]=1 for all k.
    For r=3: U[3,3]=1 (i=1,j=1), V[1,k]=1 for all k, W[k,1]=1 for all k.
    
    Check: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    For i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] + U[1,0]*V[1,0]*W[0,0] + U[2,0]*V[0,0]*W[0,0] + U[3,0]*V[1,0]*W[0,0]
    = 1*1*1 + 0 + 0 + 0 = 1. Correct.
    For i=0,j=1,k=0: U[0,1]*V[0,0]*W[0,0] + U[1,1]*V[1,0]*W[0,0] + U[2,1]*V[0,0]*W[0,0] + U[3,1]*V[1,0]*W[0,0]
    = 0 + 1*1*1 + 0 + 0 = 1. Correct.
    
    So M_{2,2,7} has rank at most 4! That's great.
    
    Similarly, M_{2,2,4} has rank at most 4.
    
    Now for M_{4,4,7}:
    C[0:2,:] = A[0:2,0:2]@B[0:2,:] + A[0:2,2:4]@B[2:4,:]
    C[2:4,:] = A[2:4,0:2]@B[0:2,:] + A[2:4,2:4]@B[2:4,:]
    
    Each 2x2 * 2x7 product has rank <= 4.
    So total rank <= 4 + 4 + 4 + 4 = 16? 
    
    Wait, but we need to be careful. The rank of the sum is at most the sum of the ranks.
    So M_{4,4,7} <= 4 * M_{2,2,7} in terms of rank? No, it's:
    M_{4,4,7} rank <= rank(M_{2,2,7}) * 4 = 4 * 4 = 16.
    
    Let me construct this explicitly.
    
    For the term A[0:2,0:2]@B[0:2,:]:
    This is a 2x2 * 2x7 multiplication.
    Using the rank-4 decomposition:
    r=0: U[i*m+j, r] = 1 if (i,j)=(0,0), V[j*p+k, r] = 1 for all k, W[k*n+i, r] = 1 for all k
    But wait, this is for the sub-matrix. We need to map to the full indices.
    
    Full indices: i in 0..3, j in 0..3, k in 0..6.
    For A[0:2,0:2]@B[0:2,:]:
    i_full in 0..1, j_full in 0..1, k_full in 0..6.
    U_full[i_full*4 + j_full, r] = 1 if (i_full, j_full) matches.
    V_full[j_full*7 + k_full, r] = 1 for all k_full.
    W_full[k_full*4 + i_full, r] = 1 for all k_full.
    
    For the 4 sub-products:
    1. A[0:2,0:2]@B[0:2,:]: i in 0..1, j in 0..1
    2. A[0:2,2:4]@B[2:4,:]: i in 0..1, j in 2..3
    3. A[2:4,0:2]@B[0:2,:]: i in 2..3, j in 0..1
    4. A[2:4,2:4]@B[2:4,:]: i in 2..3, j in 2..3
    
    Each has rank 4, total rank 16.
    
    Let me verify this works.
    """
    n, m, p = 4, 4, 7
    rank = 16
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 4 blocks, each with 4 rank terms
    block_defs = [
        (0, 2, 0, 2),  # (i_start, i_end, j_start, j_end)
        (0, 2, 2, 4),
        (2, 4, 0, 2),
        (2, 4, 2, 4),
    ]
    
    for block_idx, (i_s, i_e, j_s, j_e) in enumerate(block_defs):
        base_r = block_idx * 4
        for r_sub in range(4):
            r = base_r + r_sub
            # r_sub 0: (i,j) = (i_s, j_s)
            # r_sub 1: (i,j) = (i_s, j_s+1)
            # r_sub 2: (i,j) = (i_s+1, j_s)
            # r_sub 3: (i,j) = (i_s+1, j_s+1)
            if r_sub == 0:
                i_loc, j_loc = i_s, j_s
            elif r_sub == 1:
                i_loc, j_loc = i_s, j_s + 1
            elif r_sub == 2:
                i_loc, j_loc = i_s + 1, j_s
            else:
                i_loc, j_loc = i_s + 1, j_s + 1
            
            U[i_loc * m + j_loc, r] = 1
            # V[j*p+k, r] = 1 for all k in 0..p-1
            for k in range(p):
                V[j_loc * p + k, r] = 1
            # W[k*n+i, r] = 1 for all k in 0..p-1
            for k in range(p):
                W[k * n + i_loc, r] = 1
    
    return U, V, W

def _verify_rank(U, V, W):
    """Verify and return rank if valid."""
    if _verify(U, V, W):
        return U.shape[1]
    return None

def _try_improved_decompositions(n, m, p, deadline):
    """Try various structured decompositions."""
    best_rank = n * m * p
    best_result = None
    
    # Try the block decomposition with rank 16
    U, V, W = _build_tensor_train_decomposition(n, m, p, deadline)
    r = _verify_rank(U, V, W)
    if r is not None and r < best_rank:
        best_rank = r
        best_result = (U, V, W)
    
    # Try Strassen-based decomposition for 4x4*4x4 part + 4x4*4x3 part
    # 4x4*4x4: Use 2x2 Strassen at two levels
    # Level 1: Split into 2x2 blocks of 2x2 matrices
    # 7 products of 2x2*2x2, each of rank 7, total rank 49
    # But we can do better by sharing.
    
    # Actually, let's try: M_{4,4,4} <= 7 * M_{2,2,2} = 7*7 = 49? No.
    # M_{4,4,4}: split A, B, C into 2x2 blocks.
    # C = A*B where A,B,C are 2x2 block matrices with 2x2 entries.
    # Using Strassen at block level: 7 block multiplications.
    # Each block multiplication is 2x2*2x2, rank 7.
    # Total rank = 7 * 7 = 49.
    
    # For M_{4,4,7}:
    # C = A * B where B is 4x7.
    # Split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2
    # rank(M_{4,4,7}) <= rank(M_{4,4,4}) + rank(M_{4,4,3})
    
    # M_{4,4,4} <= 49 (Strassen at two levels)
    # M_{4,4,3}: Split B into 4x3.
    # C = A * B where B is 4x3.
    # We can compute this as: for each column of B (2x1 vectors), do 4x4 * 2x1.
    # 4x4 * 2x1: rank <= 4 (as shown above for 2x2*2x1 extended).
    # Actually, 4x4 * 4x3:
    # C[i,k] = sum_j A[i,j] * B[j,k] for j=0..3, k=0..2.
    # rank <= 4*3 = 12 (schoolbook).
    # Can we do better?
    # Using the block approach: 4x4 * 4x3 = (4x4 * 4x4) truncated? No.
    # Let's just use schoolbook for 4x4*4x3: rank 12.
    # Total: 49 + 12 = 61.
    
    # But our block decomposition gave rank 16! Let's make sure that's correct.
    
    # Let's also try a hybrid: use Strassen for the 4x4*4x4 part and schoolbook for 4x4*4x3.
    # But 16 is already much better than 61.
    
    # Let's try to improve below 16.
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28.
    # Wait, that can't be right. The lower bound for tensor rank of matmul is:
    # rank >= max(n*m, m*p, p*n) ? No, that's not a standard lower bound.
    # Actually, the rank of the matmul tensor M_{n,m,p} satisfies:
    # rank >= max(n*m, m*p, p*n) is NOT true in general.
    # For 2x2*2x2, rank = 7 < max(4,4,4)=4? No, 7 > 4.
    # The lower bound is: rank >= max(n*m, m*p, p*n) is false.
    # Actually, I think the lower bound is just max(n, m, p) in some cases, but that's weak.
    
    # Let me reconsider. The tensor M_{n,m,p} has n*m*p non-zero entries (all 1s).
    # A rank-1 tensor U[i,j]*V[j,k]*W[k,i] has at most min(n*m, m*p, p*n) non-zero entries? No.
    # A rank-1 tensor can have all n*m*p entries non-zero.
    # So the lower bound from sparsity is 1, which is useless.
    
    # The actual lower bound for matmul tensor rank is not well-known for small sizes.
    # For 2x2*2x2, it's 7 (proven by Strassen).
    # For 4x4*4x4, it's at least 49? No, it's at least... I'm not sure.
    
    # Anyway, let's see if we can beat 16.
    # The block decomposition with rank 16 uses the fact that each 2x2*2x7 has rank 4.
    # Can we do better for 2x2*2x7?
    # M_{2,2,7}: rank 4 is achieved. Can it be lower?
    # rank >= 2 (since we need at least 2 terms to cover all i,j).
    # Actually, rank >= max(n, m, p)? No.
    # For M_{2,2,7}, I believe rank 4 is optimal, but I'm not 100% sure.
    
    # Let's try to find a rank-3 or rank-4 decomposition for M_{2,2,7} and see.
    # Actually, we already have rank 4. Let's stick with it.
    
    # Total rank 16 for M_{4,4,7}. Let's verify it's correct.
    
    return best_rank, best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline, leaving 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p  # 112
    
    # Try structured decompositions
    best_rank, best_result = _try_improved_decompositions(n, m, p, deadline)
    if best_rank is not None and best_rank < current_rank:
        U, V, W = best_result
        current_rank = best_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
