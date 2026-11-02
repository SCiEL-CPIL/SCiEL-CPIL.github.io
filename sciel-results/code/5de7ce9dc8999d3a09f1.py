"""Matrix multiplication 2x4 x 4x7 - rank-minimal exact tensor decomposition."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a low-rank exact decomposition for 2x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition using block structure.
    # A is 2x4, B is 4x7, C is 2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Strategy: Use the fact that we can combine terms.
    # Let's try a direct construction with rank reduction.
    
    # Schoolbook has rank 56. Let's try to do better.
    
    # Partition: split the 4 middle indices into two groups of 2.
    # Group 1: j=0,1; Group 2: j=2,3
    # C = A_g1 * B_g1 + A_g2 * B_g2
    # where A_g1 is 2x2, B_g1 is 2x7, etc.
    
    # For 2x2 * 2x7: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This can be done with rank 14 (2 per k-value, 7 k-values)
    # So total rank = 14 + 14 = 28. Still not better than 56? 
    # Wait, 28 < 56, so this IS better!
    
    # Actually wait: schoolbook for 2x2*2x7 is 2*2*7=28, and I'm saying rank 14.
    # Let me verify: the tensor T[i,k, i',j, j'] = delta(i,i')*delta(j,j')*delta... 
    # No. For 2x2 * 2x7: output C[i,k], input A[i,j], B[j,k].
    # T[i,k, (i_a, j_a), (j_b, k_b)] = delta(i, i_a) * delta(j_a, j_b) * delta(k, k_b)
    # This is the standard matrix multiplication tensor for 2x2 * 2x7.
    # Its rank is at most 2*7 = 14 by the trivial decomposition:
    # For each (j, k) pair... no.
    
    # Trivial: C[i,k] = sum_j A[i,j] B[j,k]
    # Term for fixed (j,k): contributes A[i,j]*B[j,k] to C[i,k] for all i.
    # But A[i,j] depends on i, so we can't separate it that way for a single term.
    
    # Correct trivial decomposition: for each (i,j,k), term is A[i,j]*B[j,k] contributing to C[i,k].
    # That's 2*2*7 = 28 terms.
    
    # Can we do better? The rank of the 2x2x2 tensor is 7 (Strassen).
    # For 2x2 * 2x7, I think the rank is 14.
    
    # Let me just use the schoolbook for now and see if I can improve.
    
    # Actually, let me implement a known low-rank decomposition.
    # For 2x4 * 4x7, I'll use a construction based on polynomial multiplication tricks.
    
    # Simpler: just return schoolbook and verify it works, then try to improve.
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
# EVOLVE-BLOCK-END
