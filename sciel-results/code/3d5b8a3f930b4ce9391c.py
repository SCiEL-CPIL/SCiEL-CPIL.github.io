"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Block-wise decomposition for 2x4 * 4x7 matrix multiplication.
    
    We partition the middle dimension (size 4) into two blocks of size 2:
    A = [A1; A2] where A1, A2 are 2x2
    B = [B1 B2] where B1, B2 are 2x7
    C = A1*B1 + A2*B2
    
    Each 2x2 * 2x7 product requires 2*2*7 = 28 scalar multiplications schoolbook.
    But we can use a rank-8 decomposition for 2x2*2x2 (per column pair) and extend.
    
    Actually, let's use a simpler structural improvement:
    For each pair (i, k) in the output (2*7=14 entries), we compute:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    We can group terms: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    Each inner term is a dot product of length 2, which has rank 4 schoolbook,
    but we can use rank 3 via (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)... no that's not right.
    
    Actually for 2-term dot product: a*c + b*d has tensor rank 4 (schoolbook) and cannot be reduced.
    
    Let me try a different approach: use the known result that the matrix multiplication tensor
    for 2x4*4x7 has rank at most 50 by using a block Strassen-like method.
    
    Strategy: View as 2x2 blocks in the middle:
    Let j = 2*j2 + j1 where j2 in {0,1}, j1 in {0,1}
    
    C[i,k] = sum_{j2=0}^{1} sum_{j1=0}^{1} A[i, 2*j2+j1] * B[2*j2+j1, k]
           = sum_{j2=0}^{1} (A[i,2*j2] * B[2*j2,k] + A[i,2*j2+1] * B[2*j2+1,k])
    
    For each j2, the inner term is a 2-term dot product. We can express this using
    the identity: x*y + u*v = ((x+u)*(y+v) + (x-u)*(y-v)) / 2
    
    This uses 2 multiplications per j2 instead of 2, but with additions. However,
    for tensor rank, we need to count the bilinear forms.
    
    Using the identity: x*y + u*v = (1/2)[(x+u)(y+v) + (x-u)(y-v)]
    
    This reduces the rank! For each j2, instead of 2 separate rank-1 terms,
    we use 2 rank-1 terms but with combined inputs.
    
    Total: 2 (j2 values) * 2 (terms from identity) * 2 (i values) * 7 (k values) = 56... same.
    
    Wait, that's not reducing it. Let me think differently.
    
    The key insight from the cell findings: block-wise Strassen reduces rank from 56 to 50.
    
    Let me implement a specific rank-50 decomposition.
    
    For 2x4 * 4x7, we can use the following approach:
    Partition into 2x2 blocks in the middle dimension.
    
    A = [A_top; A_bottom] where A_top, A_bottom are 2x2
    B = [B_left B_right] where B_left, B_right are 2x7
    
    C = A_top * B_left + A_bottom * B_right
    
    Now, for each 2x2 * 2x7 product, we can use a rank-14 decomposition
    (since 2*2*7/2 = 14 if we use Strassen-like for the 2x2 part).
    
    Actually, for 2x2 * 2x2, Strassen uses 7 multiplications.
    For 2x2 * 2x7, we can apply Strassen to each pair of columns:
    7 columns -> 3 pairs + 1 single = 3*7 + 4 = 25? No, that's worse.
    
    Let me just implement a known rank-50 construction.
    
    After research: The rank of the 2x4*4x7 matrix multiplication tensor is at most 50.
    One construction uses the fact that we can share computations across the 7 output columns.
    
    Here's a concrete rank-50 decomposition:
    
    For i in {0,1}, k in {0,...,6}:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We define 50 rank-1 terms as follows:
    
    Group the j-index: j in {0,1,2,3}
    
    Use the following bilinear forms (each is a rank-1 term with U, V, W coefficients):
    
    For each i in {0,1}:
      For each k in {0,...,6}:
        Term 1: A[i,0]*B[0,k] -> U[i*4+0]=1, V[0*7+k]=1, W[k*2+i]=1
        Term 2: A[i,1]*B[1,k] -> U[i*4+1]=1, V[1*7+k]=1, W[k*2+i]=1
        Term 3: A[i,2]*B[2,k] -> U[i*4+2]=1, V[2*7+k]=1, W[k*2+i]=1
        Term 4: A[i,3]*B[3,k] -> U[i*4+3]=1, V[3*7+k]=1, W[k*2+i]=1
    
    That's 2*7*4 = 56 terms (schoolbook).
    
    To reduce to 50, we need to eliminate 6 terms using linear combinations.
    
    One approach: for fixed i, combine terms for j=0,1 and j=2,3 using:
    A[i,0]*B[0,k] + A[i,1]*B[1,k] = 0.5 * [(A[i,0]+A[i,1])*(B[0,k]+B[1,k]) + (A[i,0]-A[i,1])*(B[0,k]-B[1,k])]
    
    This uses 2 rank-1 terms instead of 2, so no savings in count, but the coefficients are different.
    
    Actually, this doesn't reduce the rank. The tensor rank of the 2-term dot product is 4.
    
    Let me try a completely different structural approach: use a search to find a lower-rank decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Start with schoolbook rank 56
    # Try to find a rank-50 decomposition using a heuristic search
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Known: rank 50 is achievable. Let's construct it explicitly.
    # 
    # The following is a rank-50 decomposition based on the structure of the problem.
    # We use the fact that the matrix multiplication tensor T_{2,4,7} can be decomposed
    # with rank 50 by exploiting the rectangular structure.
    #
    # Construction:
    # For the 2x4 * 4x7 product, we can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # We use 50 bilinear terms. The construction groups the 4 middle indices into
    # two pairs and uses a shared structure across the 7 output columns.
    
    # Let me implement a verified rank-50 construction.
    # Based on the algebraic structure, here is a rank-50 decomposition:
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    # We'll use a systematic construction.
    # The idea: for each of the 14 output positions (i,k), we need to compute
    # a sum of 4 products. We can share some of the bilinear forms.
    
    # Here's a concrete rank-50 construction:
    # 
    # We partition the computation as follows:
    # For i in {0,1}:
    #   For k in {0,...,6}:
    #     C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    #
    # We use the following 50 rank-1 terms:
    
    # Terms 0-13: For i=0, k=0..6, handle j=0,1 using 2 terms each (4 terms per k, but we share)
    # Actually, let me use a simpler verified construction.
    
    # After careful analysis, here is a rank-50 decomposition:
    
    col = 0
    # For each output position (i,k), use 4 terms but share across i for some j-pairs
    
    # Alternative: use the known fact that r(2x4, 4x7) <= 50
    # and construct it via a greedy approach.
    
    # For now, let's return the schoolbook decomposition as a fallback,
    # but attempt to verify if we can do better with a quick search.
    
    # Since we need to be safe, let's return schoolbook (rank 56) which is known correct.
    # But the task says rank 50 is achievable, so let's try to construct it.
    
    # Concrete rank-50 construction:
    # 
    # We use the following identity for each pair of middle indices:
    # For j-pair (0,1): A[i,0]B[0,k] + A[i,1]B[1,k]
    # For j-pair (2,3): A[i,2]B[2,k] + A[i,3]B[3,k]
    #
    # We can write each as:
    # 0.5 * [(A[i,j]+A[i,j+1])*(B[j,k]+B[j+1,k]) + (A[i,j]-A[i,j+1])*(B[j,k]-B[j+1,k])]
    #
    # This uses 2 terms per j-pair per (i,k), so 2*2*2*7 = 56 terms. Same as schoolbook.
    #
    # The savings come from sharing across i. For fixed j-pair and k, the terms for i=0 and i=1
    # can be combined.
    
    # Let me just return schoolbook to be safe, but with the correct interface.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
