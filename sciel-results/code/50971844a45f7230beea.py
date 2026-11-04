"""
Matrix multiplication tensor decomposition for M_{2,3,10}.
Uses Strassen's algorithm on 2x2 sub-blocks to achieve rank 55.
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-55 exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    The decomposition exploits the structure:
    C = A * B
    where A is 2x3 and B is 3x10.
    
    We split B into 5 blocks of 2 columns: B_0, B_1, ..., B_4.
    Each B_i is 3x2.
    We split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
    
    C_i (2x2) = A_left * B_i_top (2x2) + A_right * B_i_bottom (1x2)
    
    The term A_right * B_i_bottom is rank 2 (schoolbook for 2x1 * 1x2 -> 2x2).
    Wait, A_right is 2x1, B_i_bottom is 1x2. The product is 2x2.
    Rank of 2x1 * 1x2 tensor is 2.
    So for each of the 5 blocks, we need:
    - Rank 7 for A_left (2x2) * B_i_top (2x2) using Strassen.
    - Rank 2 for A_right (2x1) * B_i_bottom (1x2) using schoolbook.
    Total per block: 9.
    Total rank: 5 * 9 = 45?
    
    Let's re-evaluate the dimensions.
    A: 2x3. Rows i=0,1. Cols j=0,1,2.
    B: 3x10. Rows j=0,1,2. Cols k=0..9.
    C: 2x10. Rows i=0,1. Cols k=0..9.
    
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    Group k into pairs (2t, 2t+1).
    For a fixed pair t, let k0=2t, k1=2t+1.
    C[:, k0:k1+1] is a 2x2 matrix.
    C[:, k0:k1+1] = A[:, 0:2] * B[0:2, k0:k1+1] + A[:, 2:3] * B[2:3, k0:k1+1]
    
    Let A_L = A[:, 0:2] (2x2).
    Let A_R = A[:, 2:3] (2x1).
    Let B_T = B[0:2, k0:k1+1] (2x2).
    Let B_B = B[2:3, k0:k1+1] (1x2).
    
    Term 1: A_L * B_T. This is 2x2 matrix multiplication. Rank 7 (Strassen).
    Term 2: A_R * B_B. This is (2x1) * (1x2) = 2x2 outer product.
    A_R = [a0, a1]^T. B_B = [b0, b1].
    Result: [[a0*b0, a0*b1], [a1*b0, a1*b1]].
    This tensor has rank 2?
    T[i, j, k] = A_R[i] * B_B[k] (where j is the dummy index from the middle dimension, which is 1 here).
    Actually, the tensor for 2x1 * 1x2 is:
    T[i, j, k] = delta_{j,0} * A_R[i] * B_B[k].
    This can be decomposed into 2 rank-1 terms:
    r=0: U[0]=1, U[1]=1? No.
    U[i, r], V[j, r], W[k, r].
    For rank 2:
    r=0: U=[1,0], V=[1], W=[1,0] -> contributes A_R[0]*B_B[0] to C[0,0]
    r=1: U=[0,1], V=[1], W=[0,1] -> contributes A_R[1]*B_B[1] to C[1,1]? No.
    
    Let's check the rank of 2x1 * 1x2.
    The tensor is T_{i, j, k} = A_{i, j} B_{j, k}. Here j is only 0.
    T_{i, 0, k} = A_{i, 0} B_{0, k}.
    This is an outer product of vector A_col (size 2) and vector B_row (size 2).
    The rank of an outer product of two vectors is 1.
    Wait. The tensor is defined by indices i (row of C), j (col of A / row of B), k (col of C).
    For the sub-problem A_R (2x1) * B_B (1x2):
    Indices: i in {0,1}, j in {0}, k in {0,1}.
    T[i, j, k] = A_R[i, j] * B_B[j, k].
    Since j is fixed to 0, T[i, 0, k] = A_R[i, 0] * B_B[0, k].
    This is indeed rank 1.
    U[i, 0] = A_R[i, 0]
    V[0, 0] = 1
    W[k, 0] = B_B[0, k]
    
    So, per block:
    Strassen part: Rank 7.
    Schoolbook part (for j=2): Rank 1.
    Total per block: 8.
    Total rank: 5 * 8 = 40.
    
    Is this correct?
    Let's verify the "Rank 1" claim.
    We need to represent the contribution of A[:, 2] * B[2, :] to C.
    C[i, k] += A[i, 2] * B[2, k].
    In the tensor decomposition T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    For the term corresponding to j=2:
    We need T[i, 2, k] to include A[i, 2] * B[2, k].
    If we add a rank-1 term:
    U[i, r] = A[i, 2]
    V[j, r] = delta_{j, 2}
    W[k, r] = B[2, k]
    Then sum_r ... U[i, r] V[j, r] W[k, r] = A[i, 2] * delta_{j, 2} * B[2, k].
    This matches the required contribution to the tensor T at index (i, 2, k).
    Yes, this is rank 1.
    
    So the total rank is 5 * (7 + 1) = 40.
    
    Let's double check if Strassen works for the 2x2 block.
    A_L is 2x2, B_T is 2x2.
    Strassen gives rank 7 for 2x2 matrix multiplication.
    Yes.
    
    So we can achieve rank 40!
    
    Let's construct the matrices for rank 40.
    
    n=2, m=3, p=10.
    U: 6x40
    V: 30x40
    W: 20x40
    
    For each block t in 0..4:
      k0 = 2t, k1 = 2t+1.
      
      # Strassen for A[:, 0:2] * B[0:2, k0:k1+1]
      # A indices: 0, 1, 2, 3 (i*m+j). 
      # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4].
      # B indices: j*p+k.
      # B[0,k0]=V[k0], B[0,k1]=V[k1], B[1,k0]=V[10+k0], B[1,k1]=V[10+k1].
      # C indices: k*n+i.
      # C[0,k0]=W[0], C[1,k0]=W[1], C[0,k1]=W[2], C[1,k1]=W[3].
      
      # Strassen 7 products:
      # M1 = (A00 + A11) * (B00 + B11)
      # M2 = (A10 + A11) * B00
      # M3 = A00 * (B01 - B11)
      # M4 = A11 * (B10 - B00)
      # M5 = (A00 + A01) * B11
      # M6 = (A10 - A00) * (B00 + B01)
      # M7 = (A01 - A11) * (B10 + B11)
      
      # C00 = M1 + M4 - M5 + M7
      # C01 = M3 + M5
      # C10 = M2 + M4
      # C11 = M1 - M2 + M3 + M6
      
      # We map these to 7 columns in U, V, W.
      
      # Schoolbook for A[:, 2] * B[2, k0:k1+1]
      # This is rank 1.
      # U[i, r] = A[i, 2] -> U[2]=1, U[5]=1? No, A[i,2] is variable.
      # In the tensor decomposition, U, V, W are fixed coefficients.
      # The "variables" A and B are inputs.
      # The tensor T is the structure of the multiplication.
      # T[i, j, k] = 1 if j is the common index.
      # Specifically, T[i, j, k] = 1 if we are multiplying A[i, j] and B[j, k].
      # Wait, the standard tensor for matrix multiplication M_{n,m,p} has T[i, j, k] = 1 if the indices align?
      # No, the tensor T is such that T(A, B) = C.
      # T is defined by T_{i, j, k} = 1 if we sum over j?
      # Actually, the task says "exact tensor decomposition for multiplying...".
      # The tensor is T where T[i, j, k] = 1 if the multiplication A[i, j] * B[j, k] contributes to C[i, k].
      # For standard matrix multiplication, T[i, j, k] = 1 for all valid i, j, k?
      # No, that's the identity tensor.
      # The multiplication tensor is T_{i, j, k} = 1 if the indices are "compatible" in the sense that they form a path i->j->k.
      # For M_{2,3,10}, T is 6x30x20.
      # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
      # This implies j is determined by a and b.
      # j = a % m. j = b // p.
      # If a % m == b // p, then T[a, b, c] is non-zero.
      # Specifically, if a % m == b // p, let j = a % m.
      # Then i = a // m. k = b % p.
      # c must be k * n + i.
      # So T[a, b, c] = 1 if a % m == b // p and c == (b % p) * n + (a // m).
      # Otherwise 0.
      
      # So we need to decompose this specific T.
      
      # My Strassen construction above assumes we are decomposing the tensor for 2x2 * 2x2.
      # Let's verify the Strassen decomposition for the 2x2 * 2x2 tensor.
      # T_{2x2} has indices i in {0,1}, j in {0,1}, k in {0,1}.
      # T[i, j, k] = 1 if c = k*2 + i matches the "output" index?
      # In the 2x2 * 2x2 case, m=2.
      # a = i*2 + j. b = j*2 + k. c = k*2 + i.
      # T[a, b, c] = 1 if a%2 == b//2 and c == (b%2)*2 + (a//2).
      
      # Strassen's algorithm provides a rank-7 decomposition for this tensor.
      # The coefficients U, V, W for Strassen are well-known.
      
      # Now for the j=2 part (A[:, 2] * B[2, :]).
      # a = i*3 + 2. b = 2*10 + k. c = k*2 + i.
      # T[a, b, c] = 1 if a%3 == 2 (which is true for a=2,5) and b//10 == 2 (which is true for b=20..29).
      # And c == (b%10)*2 + (a//3).
      # Let i = a//3 (0 or 1). k = b%10.
      # c = k*2 + i.
      # This matches the condition.
      # So for each k, there is a 1 at T[i*3+2, 20+k, k*2+i].
      # This is a set of 2*10 = 20 ones.
      # Can we decompose this with rank 5?
      # We grouped k into pairs.
      # For a pair (k0, k1), we have 4 ones:
      # (0*3+2, 20+k0, k0*2+0) -> (2, 20+k0, 2*k0)
      # (1*3+2, 20+k0, k0*2+1) -> (5, 20+k0, 2*k0+1)
      # (0*3+2, 20+k1, k1*2+0) -> (2, 20+k1, 2*k1)
      # (1*3+2, 20+k1, k1*2+1) -> (5, 20+k1, 2*k1+1)
      
      # My previous "Rank 1" argument was:
      # U[i, r] = A[i, 2] ... wait, U, V, W are constants.
      # The tensor entry T[a, b, c] is 1.
      # We want sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c].
      # For the j=2 slice:
      # T[2, 20+k, 2k] = 1
      # T[5, 20+k, 2k+1] = 1
      # Others 0.
      
      # Let's try to represent this with 1 rank term?
      # U[2, r] V[20+k, r] W[2k, r] = 1
      # U[5, r] V[20+k, r] W[2k+1, r] = 1
      # This looks like it requires dependence on k.
      # If we use 1 term for the whole block (k0, k1):
      # We have 4 non-zeros.
      # If we use 2 terms (one for k0, one for k1)?
      # Term for k0:
      # U[2, r0]=1, U[5, r0]=1?
      # V[20+k0, r0]=1, V[20+k1, r0]=0?
      # W[2k0, r0]=1, W[2k0+1, r0]=1?
      # Then T[2, 20+k0, 2k0] = 1*1*1 = 1.
      # T[5, 20+k0, 2k0+1] = 1*1*1 = 1.
      # T[2, 20+k0, 2k0+1] = 1*1*0 = 0.
      # T[5, 20+k0, 2k0] = 1*1*0 = 0.
      # This works for k0!
      # Similarly for k1 with r1.
      # So we need 2 rank terms per block for the j=2 part.
      # Total per block: 7 (Strassen) + 2 (Schoolbook j=2) = 9.
      # Total rank: 5 * 9 = 45.
      
      # Can we do better?
      # Is the rank of the j=2 slice 2 per block?
      # The sub-tensor for j=2 is block diagonal in k.
      # Each 2x1 (k) block has rank 2?
      # Matrix of non-zeros for k0:
      # Indices: a in {2,5}, b in {20+k0}, c in {2k0, 2k0+1}.
      # This is a 2x1x2 tensor.
      # Rank of 2x1x2 tensor?
      # It's an outer product of vector u (size 2) and vector w (size 2), with v (size 1) being [1].
      # T[a, b, c] = u[a] * v[b] * w[c].
      # Here v is fixed to 1 at the single index.
      # So it's effectively u * w.
      # The rank of an outer product of two vectors is 1.
      # Wait.
      # T[2, 20+k0, 2k0] = 1
      # T[5, 20+k0, 2k0+1] = 1
      # T[2, 20+k0, 2k0+1] = 0
      # T[5, 20+k0, 2k0] = 0
      # This is NOT an outer product.
      # Outer product u=[1,0], w=[1,0] gives [[1,0],[0,0]].
      # Outer product u=[0,1], w=[0,1] gives [[0,0],[0,1]].
      # Sum of these two is [[1,0],[0,1]]. Rank 2.
      # Our tensor is [[1,0],[0,1]]?
      # Rows a=2,5. Cols c=2k0, 2k0+1.
      # Entry (2, 2k0) is 1.
      # Entry (5, 2k0+1) is 1.
      # Entry (2, 2k0+1) is 0.
      # Entry (5, 2k0) is 0.
      # Yes, it is the identity matrix I_2.
      # The rank of the identity matrix as a tensor (with the middle index fixed) is 2.
      # So we need 2 rank terms per block for the j=2 part.
      
      # Total rank = 5 * (7 + 2) = 45.
      
      # Can we combine the Strassen and Schoolbook parts?
      # Unlikely, as they act on disjoint sets of indices in A and B.
      # Strassen uses A[:, 0:2] and B[0:2, :].
      # Schoolbook uses A[:, 2] and B[2, :].
      # The indices in A and B are disjoint.
      # So the tensors are "orthogonal" in the A and B modes.
      # The sum of ranks is the rank of the sum.
      
      # So Rank 45 is achievable.
      
    """
    n, m, p = DIMENSIONS
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # --- Strassen 7 products for A[:, 0:2] * B[0:2, k0:k1+1] ---
        # A indices:
        # A00 = U[0]
        # A01 = U[1]
        # A10 = U[3]
        # A11 = U[4]
        
        # B indices:
        # B0k0 = V[k0]
        # B0k1 = V[k1]
        # B1k0 = V[10+k0]
        # B1k1 = V[10+k1]
        
        # C indices:
        # C0k0 = W[2*k0]
        # C1k0 = W[2*k0+1]
        # C0k1 = W[2*k1]
        # C1k1 = W[2*k1+1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        V[10+k1][r] += 1
        W[2*k0][r] += 1  # C00
        W[2*k1+1][r] += 1 # C11
        r += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        W[2*k0+1][r] += 1 # C10
        W[2*k1+1][r] -= 1 # -C11
        r += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][r] += 1
        V[k1][r] += 1
        V[10+k1][r] -= 1
        W[2*k1][r] += 1 # C01
        W[2*k1+1][r] += 1 # C11
        r += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][r] += 1
        V[10+k0][r] += 1
        V[k0][r] -= 1
        W[2*k0][r] += 1 # C00
        W[2*k0+1][r] += 1 # C10
        r += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][r] += 1
        U[1][r] += 1
        V[10+k1][r] += 1
        W[2*k0][r] -= 1 # -C00
        W[2*k1][r] += 1 # C01
        r += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][r] += 1
        U[0][r] -= 1
        V[k0][r] += 1
        V[k1][r] += 1
        W[2*k1+1][r] += 1 # C11
        r += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r] += 1
        U[4][r] -= 1
        V[10+k0][r] += 1
        V[10+k1][r] += 1
        W[2*k0][r] += 1 # C00
        r += 1
        
        # --- Schoolbook 2 products for A[:, 2] * B[2, k0:k1+1] ---
        # This part corresponds to the identity matrix contribution for each k.
        # For k0:
        # C[0, k0] += A[0, 2] * B[2, k0]
        # C[1, k0] += A[1, 2] * B[2, k0]
        # Tensor entries:
        # T[0*3+2, 2*10+k0, k0*2+0] = 1
        # T[1*3+2, 2*10+k0, k0*2+1] = 1
        
        # Term 1: Covers C[0, k0]
        U[2][r] += 1
        V[20+k0][r] += 1
        W[2*k0][r] += 1
        r += 1
        
        # Term 2: Covers C[1, k0]
        U[5][r] += 1
        V[20+k0][r] += 1
        W[2*k0+1][r] += 1
        r += 1
        
        # For k1:
        # Term 3: Covers C[0, k1]
        U[2][r] += 1
        V[20+k1][r] += 1
        W[2*k1][r] += 1
        r += 1
        
        # Term 4: Covers C[1, k1]
        U[5][r] += 1
        V[20+k1][r] += 1
        W[2*k1+1][r] += 1
        r += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Verification
    n, m, p = 2, 3, 10
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    print("Error:", err)
