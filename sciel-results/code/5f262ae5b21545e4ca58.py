import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W, r):
        """Verify that U, V, W with rank r exactly decompose the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Construct a Strassen-like decomposition
    # 2x4x7 can be seen as 2 blocks of 2x2x7 or similar.
    # R(2,2,2)=7. R(2,2,1)=4.
    # Let's try to decompose 2x4x7 using 2x2x2 and 2x2x1 blocks.
    # 4 = 2+2. 7 = 2+2+2+1.
    # So we have 2 (rows) x 2 (col blocks) x 4 (row blocks of p)
    # This is 4 blocks of 2x2x2 and 2 blocks of 2x2x1?
    # Wait, dimensions are (n, m, p) = (2, 4, 7).
    # Split m into 2+2. Split p into 2+2+2+1.
    # Blocks:
    # (i in 0..1, j in 0..1, k in 0..1) -> 2x2x2
    # (i in 0..1, j in 0..1, k in 2..3) -> 2x2x2
    # (i in 0..1, j in 0..1, k in 4..5) -> 2x2x2
    # (i in 0..1, j in 0..1, k in 6..6) -> 2x2x1
    # (i in 0..1, j in 2..3, k in 0..1) -> 2x2x2
    # (i in 0..1, j in 2..3, k in 2..3) -> 2x2x2
    # (i in 0..1, j in 2..3, k in 4..5) -> 2x2x2
    # (i in 0..1, j in 2..3, k in 6..6) -> 2x2x1
    
    # Total blocks: 6 of 2x2x2 and 2 of 2x2x1.
    # Rank(2x2x2) = 7 (Strassen)
    # Rank(2x2x1) = 4 (Gauß? No, R(2,2,1)=4 is known, e.g. via 2x2x1 tensor)
    # Actually R(2,2,1) is the rank of the bilinear map (x,y) -> x*y^T (2x1 vector).
    # This is just element-wise product of two 2-vectors. Rank is 2?
    # Wait. Tensor T_{ijk} where i,j in {0,1}, k in {0}.
    # T_{ijk} = 1 if i==j? No, matrix multiplication C = A B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # If B is 2x1, B_{j0}. C_{i0} = A_{i0}B_{00} + A_{i1}B_{10}.
    # The tensor is sum_j e_i^A e_j^B e_j^C? No.
    # The term is A_{ij} B_{jk}.
    # For fixed i, k, sum over j.
    # The tensor is T_{i j k} = delta_{i, i'} delta_{j, j'} delta_{k, k'}?
    # The multiplication tensor is sum_{i,j,k} e_i^C e_j^A e_k^B (delta_{jk} is not there, it's the contraction).
    # Actually, the tensor is T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^C \otimes e_j^A \otimes e_k^B (with indices mapped correctly).
    # Specifically, T_{i j k} = 1 if the index in C is i, A is j, B is k?
    # No, the standard tensor is T = \sum_{i,j,k} e_i^{(C)} \otimes e_j^{(A)} \otimes e_k^{(B)} is wrong.
    # The tensor for matrix multiplication C=AB is:
    # T = \sum_{i=0}^{n-1} \sum_{j=0}^{m-1} \sum_{k=0}^{p-1} e_i^{(C)} \otimes e_{j,k}^{(A,B)}?
    # Let's stick to the definition in the prompt:
    # W[k*n+i, r] corresponds to C_{i,k}.
    # U[i*m+j, r] corresponds to A_{i,j}.
    # V[j*p+k, r] corresponds to B_{j,k}.
    # We need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is a valid triple?
    # No, the equation is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T_{i j k} such that T(A,B) = C is defined by:
    # T_{i j k} = 1 if the indices match the contraction?
    # The prompt says: "exactly decompose the matrix multiplication tensor".
    # Usually this means:
    # \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} = 1 for all i,j,k?
    # No, that would imply C_{i,k} = sum_r ... which is not how it works.
    # The decomposition is of the tensor T where T_{i, j, k} = 1 if we are computing the term A_{i,j} B_{j,k} contributing to C_{i,k}.
    # So for each term in the sum, we have a specific i, j, k.
    # The tensor is T = \sum_{i,j,k} e_i^{(C)} \otimes e_j^{(A)} \otimes e_k^{(B)} ?
    # Let's check the verify function in the seed:
    # val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # It checks if val == 1.0 for ALL i, j, k.
    # This implies that the tensor is simply T_{i,j,k} = 1 for all i,j,k?
    # That can't be right for matrix multiplication.
    # Matrix multiplication is C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    # The tensor T is such that T(A,B) = C.
    # In tensor notation, T = \sum_{i,j,k} E_{i,k}^{(C)} \otimes E_{i,j}^{(A)} \otimes E_{j,k}^{(B)}?
    # No.
    # Let's look at the seed's verify again.
    # It iterates i, j, k.
    # It computes sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # And requires it to be 1.
    # This means the tensor T has T_{i,j,k} = 1 for ALL i,j,k.
    # This is NOT matrix multiplication. This is the "all ones" tensor.
    # Wait. If T_{i,j,k} = 1 for all i,j,k, then the decomposition is trivial?
    # No, the rank of the all-ones tensor of size n x m x p is 1.
    # U=1, V=1, W=1.
    # But the seed returns rank 56.
    # Why?
    # Let's re-read the prompt carefully.
    # "Matrix multiplication 2,4,7"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # The seed's verify function checks:
    # `if abs(val - 1.0) > 1e-9: return False`
    # This implies the target tensor has entries 1.0 at every (i,j,k) position?
    # If so, the rank is 1.
    # But the seed uses schoolbook (rank 56) as fallback and searches for lower.
    # If the target was all-ones, the seed would find rank 1 immediately.
    # It didn't (score 1.0 means it returned schoolbook or something similar).
    # Therefore, my interpretation of the verify function in the seed might be wrong, OR the seed is broken, OR the tensor is NOT all-ones.
    
    # Let's look at the standard definition of the matrix multiplication tensor.
    # T = \sum_{i=0}^{n-1} \sum_{j=0}^{m-1} \sum_{k=0}^{p-1} e_i^{(C)} \otimes e_{i,j}^{(A)} \otimes e_{j,k}^{(B)}
    # Here, the indices for A are (i,j), for B are (j,k), for C are (i,k).
    # The tensor T is a 3-tensor of size (n*p) x (n*m) x (m*p).
    # T_{(i,k), (i',j), (j',k')} = 1 if i=i', k=k', j=j'.
    # Otherwise 0.
    # So T_{i,k, i',j, j',k'} = delta_{i,i'} delta_{k,k'} delta_{j,j'}.
    # This tensor is sparse.
    # The seed's verify function:
    # `val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]`
    # It checks if this sum is 1.
    # But for the actual MM tensor, the value should be 1 ONLY if the indices correspond to a valid term in the sum?
    # No, the decomposition is:
    # C_{i,k} = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} ?
    # No, the standard decomposition is:
    # C_{i,k} = \sum_{j} A_{i,j} B_{j,k}.
    # We want to write this as \sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)}?
    # No, the factors are:
    # U has indices (i,j) and r.
    # V has indices (j,k) and r.
    # W has indices (i,k) and r.
    # The contraction is over r.
    # So C_{i,k} = \sum_r W_{i,k}^{(r)} ( \sum_j U_{i,j}^{(r)} A_{i,j} ? ) No.
    # The bilinear map is:
    # C = \sum_r W^{(r)} \otimes ( U^{(r)} \circ A ) \otimes ( V^{(r)} \circ B )?
    # No.
    # The tensor T is defined by T(A,B) = C.
    # T is a 3-linear form? No, it's a tensor such that:
    # T_{i,k, i',j, j',k'} = 1 if i=i', k=k', j=j'.
    # The rank decomposition is:
    # T = \sum_{r=1}^R U^{(r)} \otimes V^{(r)} \otimes W^{(r)}
    # Where U^{(r)} is a vector of size n*m, V^{(r)} is size m*p, W^{(r)} is size p*n.
    # So T_{a,b,c} = \sum_r U_a^{(r)} V_b^{(r)} W_c^{(r)}.
    # Here a is index for A (i*m+j), b is index for B (j*p+k), c is index for C (k*n+i).
    # We require T_{a,b,c} = 1 if the indices are consistent with the MM structure, 0 otherwise.
    # Specifically, T_{i*m+j, j'*p+k', k''*n+i''} = 1 if i=i'' and k=k'' and j=j'.
    # Otherwise 0.
    
    # The seed's verify function:
    # `val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]`
    # It checks if val == 1.
    # But it does this for ALL i, j, k.
    # If the tensor is the MM tensor, then for a given i, j, k,
    # the index for A is i*m+j.
    # The index for B is j*p+k.
    # The index for C is k*n+i.
    # The MM tensor value at these indices is 1 IF AND ONLY IF the indices are "linked" in the MM sense?
    # Wait. In the MM tensor, the non-zero entries are exactly those where the A-index (i,j), B-index (j,k), and C-index (i,k) share the same i, j, k.
    # So for ANY i, j, k, the triplet (A_idx, B_idx, C_idx) = (i*m+j, j*p+k, k*n+i) corresponds to a NON-ZERO entry in the MM tensor?
    # Yes!
    # The MM tensor is:
    # T = \sum_{i,j,k} e_{i*m+j}^{(A)} \otimes e_{j*p+k}^{(B)} \otimes e_{k*n+i}^{(C)}
    # So T_{a,b,c} = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # And 0 otherwise.
    # Since every triplet (i,j,k) generates a unique triplet of indices, and these are the only non-zero entries,
    # the verify function in the seed IS CORRECT for the MM tensor.
    # It checks that for every i,j,k, the sum of products is 1.
    # It does NOT check that for OTHER index combinations (which would be 0 in the tensor), the sum is 0.
    # BUT, if the decomposition is exact, it must be 0 for all other combinations.
    # The seed's verify function is INCOMPLETE. It only checks the 1s, not the 0s.
    # This means the seed might accept invalid decompositions that are 1 on the diagonal but non-zero elsewhere.
    # HOWEVER, the task says "exactly decompose".
    # My program MUST verify the 0s as well to be safe, or assume the structure enforces it.
    # Given the small size, I should verify the full tensor.

    def verify_full_decomposition(U, V, W, r):
        """Verify full tensor decomposition."""
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some 0s (random sampling or exhaustive if small)
        # Total indices: (n*m) * (m*p) * (p*n) = 8 * 28 * 14 = 3136.
        # Non-zero: 56.
        # We should check a few 0s to ensure no leakage.
        # Let's check all 0s since 3136 is small.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) corresponds to a valid (i,j,k)
                    # a = i*m+j => i = a//m, j = a%m
                    # b = j'*p+k => j' = b//p, k = b%p
                    # c = k'*n+i' => k' = c//n, i' = c%n
                    i = a // m
                    j = a % m
                    j_prime = b // p
                    k = b % p
                    k_prime = c // n
                    i_prime = c % n
                    
                    is_nonzero = (i == i_prime and j == j_prime and k == k_prime)
                    
                    val = 0.0
                    for col in range(r):
                        val += U[a, col] * V[b, col] * W[c, col]
                    
                    if is_nonzero:
                        if abs(val - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(val - 0.0) > 1e-9:
                            return False
        return True

    # Construct Strassen-like decomposition
    # We will build the decomposition by combining smaller known decompositions.
    # 2x2x2 Strassen:
    # A, B are 2x2. C = AB.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M2+M3-M4+M6
    # C10 = M1-M3+M5-M7
    # C11 = M2+M4-M6+M7
    
    # We need to map this to U, V, W matrices.
    # For a 2x2x2 block, U is 4x7, V is 4x7, W is 4x7.
    # Indices:
    # U[i*2+j, r]
    # V[j*2+k, r]
    # W[k*2+i, r]
    
    def get_strassen_2x2x2():
        """Returns U, V, W for 2x2x2 Strassen."""
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        
        # M1: (A00+A11)(B00+B11)
        # A00 is idx 0, A11 is idx 3. B00 is idx 0, B11 is idx 3.
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C10 += M1
        # C00 is idx 0 (k=0, i=0), C10 is idx 2 (k=0, i=1)
        W[0, 0] = 1; W[2, 0] = 1
        
        # M2: (A10+A11)B00
        # A10 is idx 2, A11 is idx 3. B00 is idx 0.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C01 += M2, C11 += M2
        # C01 is idx 1 (k=1, i=0), C11 is idx 3 (k=1, i=1)
        W[1, 1] = 1; W[3, 1] = 1
        
        # M3: A00(B10-B11)
        # A00 is idx 0. B10 is idx 2, B11 is idx 3.
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C01 -= M3, C10 -= M3
        W[1, 2] = -1; W[2, 2] = -1
        
        # M4: A11(B01-B00)
        # A11 is idx 3. B01 is idx 1, B00 is idx 0.
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        # C00 -= M4, C11 += M4
        W[0, 3] = -1; W[3, 3] = 1
        
        # M5: (A00+A10)B11
        # A00 is idx 0, A10 is idx 2. B11 is idx 3.
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        # C00 -= M5, C10 += M5
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6: (A10-A00)(B00+B10)
        # A10 is idx 2, A00 is idx 0. B00 is idx 0, B10 is idx 2.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        # C01 += M6, C11 -= M6
        W[1, 5] = 1; W[3, 5] = -1
        
        # M7: (A01-A11)(B01+B11)
        # A01 is idx 1, A11 is idx 3. B01 is idx 1, B11 is idx 3.
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        # C00 += M7, C10 -= M7
        W[0, 6] = 1; W[2, 6] = -1
        
        return U, V, W

    def get_strassen_2x2x1():
        """Returns U, V, W for 2x2x1.
        C0 = A00 B00 + A01 B10
        C1 = A10 B00 + A11 B10
        This is just 2 independent products? No.
        C = A (2x2) * B (2x1).
        C0 = A00*B00 + A01*B10
        C1 = A10*B00 + A11*B10
        We can use 4 products:
        P1 = A00 B00
        P2 = A01 B10
        P3 = A10 B00
        P4 = A11 B10
        C0 = P1 + P2
        C1 = P3 + P4
        Rank 4.
        """
        r = 4
        U = np.zeros((4, r))
        V = np.zeros((4, r)) # B is 2x1, so size 2*1=2? No, V size is m*p. Here m=2, p=1. Size 2.
        # Wait, V shape is (m*p, r). m=2, p=1. So V is 2x4.
        # U shape is (n*m, r). n=2, m=2. So U is 4x4.
        # W shape is (p*n, r). p=1, n=2. So W is 2x4.
        
        # P1: A00 B00
        U[0, 0] = 1 # A00
        V[0, 0] = 1 # B00
        W[0, 0] = 1 # C0
        # W index: k*n+i. k=0, i=0 -> 0.
        
        # P2: A01 B10
        U[1, 1] = 1 # A01
        V[1, 1] = 1 # B10
        W[0, 1] = 1 # C0
        
        # P3: A10 B00
        U[2, 2] = 1 # A10
        V[0, 2] = 1 # B00
        W[1, 2] = 1 # C1
        # W index: k=0, i=1 -> 1.
        
        # P4: A11 B10
        U[3, 3] = 1 # A11
        V[1, 3] = 1 # B10
        W[1, 3] = 1 # C1
        
        return U, V, W

    # Now assemble the full 2x4x7 decomposition.
    # Blocks:
    # (i,j,k) ranges:
    # i in {0,1}
    # j in {0,1,2,3}
    # k in {0,1,2,3,4,5,6}
    
    # We split j into j1 in {0,1} and j2 in {2,3}.
    # We split k into k1 in {0,1}, k2 in {2,3}, k3 in {4,5}, k4 in {6}.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 (Strassen)
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 (Strassen)
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> 2x2x2 (Strassen)
    # Block 4: i in {0,1}, j in {0,1}, k in {6}    -> 2x2x1 (Rank 4)
    # Block 5: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2 (Strassen)
    # Block 6: i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2 (Strassen)
    # Block 7: i in {0,1}, j in {2,3}, k in {4,5} -> 2x2x2 (Strassen)
    # Block 8: i in {0,1}, j in {2,3}, k in {6}    -> 2x2x1 (Rank 4)
    
    # Total Rank = 6*7 + 2*4 = 42 + 8 = 50.
    
    U_full = np.zeros((n * m, 50))
    V_full = np.zeros((m * p, 50))
    W_full = np.zeros((p * n, 50))
    
    col_offset = 0
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        """Add a block decomposition to the full matrices."""
        nonlocal col_offset
        r_block = U_block.shape[1]
        
        for r in range(r_block):
            # U indices
            for i in range(2):
                for j in range(2):
                    idx_u = (i_offset + i) * m + (j_offset + j)
                    U_full[idx_u, col_offset + r] = U_block[i * 2 + j, r]
            
            # V indices
            for j in range(2):
                for k in range(2 if V_block.shape[0] == 4 else 1):
                    # V_block shape is (2*2, r) or (2*1, r)
                    # If 2x2x2, V is 4x7. j in 0,1. k in 0,1.
                    # If 2x2x1, V is 2x4. j in 0,1. k in 0.
                    if V_block.shape[0] == 4:
                        idx_v = (j_offset + j) * p + (k_offset + k)
                        V_full[idx_v, col_offset + r] = V_block[j * 2 + k, r]
                    else:
                        idx_v = (j_offset + j) * p + (k_offset)
                        V_full[idx_v, col_offset + r] = V_block[j, r]
            
            # W indices
            for k in range(2 if W_block.shape[0] == 4 else 1):
                for i in range(2):
                    if W_block.shape[0] == 4:
                        idx_w = (k_offset + k) * n + (i_offset + i)
                        W_full[idx_w, col_offset + r] = W_block[k * 2 + i, r]
                    else:
                        idx_w = (k_offset) * n + (i_offset + i)
                        W_full[idx_w, col_offset + r] = W_block[k, r]
        
        col_offset += r_block

    # Get blocks
    U_s, V_s, W_s = get_strassen_2x2x2()
    U_1, V_1, W_1 = get_strassen_2x2x1()
    
    # Add blocks
    # j1=0, k1=0
    add_block(U_s, V_s, W_s, 0, 0, 0)
    # j1=0, k2=2
    add_block(U_s, V_s, W_s, 0, 0, 2)
    # j1=0, k3=4
    add_block(U_s, V_s, W_s, 0, 0, 4)
    # j1=0, k4=6
    add_block(U_1, V_1, W_1, 0, 0, 6)
    
    # j2=2, k1=0
    add_block(U_s, V_s, W_s, 0, 2, 0)
    # j2=2, k2=2
    add_block(U_s, V_s, W_s, 0, 2, 2)
    # j2=2, k3=4
    add_block(U_s, V_s, W_s, 0, 2, 4)
    # j2=2, k4=6
    add_block(U_1, V_1, W_1, 0, 2, 6)
    
    # Verify
    if verify_full_decomposition(U_full, V_full, W_full, 50):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook if construction failed
        return best_result

# EVOLVE-BLOCK-END
