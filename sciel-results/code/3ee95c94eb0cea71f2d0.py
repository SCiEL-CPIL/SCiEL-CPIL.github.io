"""Matrix multiplication 2x4x8 with structural Strassen-based decomposition."""
import time
import itertools

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen's algorithm on the 4x8 matrix B by splitting it
    into two 4x4 blocks. Each 4x4 block can be multiplied with the 2x4 matrix
    A using a Strassen-like approach.
    
    Actually, let's think more carefully. We have A (2x4) * B (4x8) = C (2x8).
    
    We can split B into two 4x4 blocks: B = [B1 | B2] where B1, B2 are 4x4.
    Then C = [A*B1 | A*B2].
    
    For each 2x4 * 4x4 multiplication, we can use a rank-14 decomposition:
    - Split A into two 2x2 blocks: A = [A11 A12]
    - Split each 4x4 block into four 2x2 blocks
    - Use Strassen's 7 multiplications for each 2x2*2x2 sub-product
    
    But we need to be careful about the indices.
    
    Let's use a direct approach: enumerate all possible rank-r decompositions
    and verify correctness. Start with known lower bounds and work up.
    
    For 2x4x8 matrix multiplication, the trivial rank is 2*4*8 = 64.
    Using Strassen ideas, we can do better.
    
    Let's try to build a decomposition of rank 28:
    - 2 rows of A, each row interacts with 4 columns of B
    - Split into 4 blocks of 2x2x2 (each contributing rank 7 via Strassen)
    - Total: 4 * 7 = 28
    
    Actually, let me construct this explicitly.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Build the multiplication tensor T[i, j, k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # We need U, V, W such that for all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other (i,j,k) combinations that don't contribute, the sum is 0.
    # Actually, the tensor is T[i,j,k] = delta(i,j,k) where the product A*B gives
    # C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    
    # Wait, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because every term A[i,j]*B[j,k] contributes to C[i,k].
    
    # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for indices that don't form valid triples... actually all (i,j,k) with
    # i<n, j<m, k<p are valid. The tensor is all-ones on the valid domain.
    
    # Let me verify: the multiplication tensor for n x m * m x p has entries
    # T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p, else 0.
    # And the condition is that for the decomposition to be exact:
    # For each (i,k): C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need rank-r matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try rank 28 using block Strassen.
    # Split the 4x8 B matrix into four 2x4 blocks? No, let's split differently.
    
    # Alternative: use the fact that we can split the middle dimension.
    # A is 2x4, B is 4x8. Split the middle dimension into two parts of size 2:
    # A = [A_left | A_right] where A_left, A_right are 2x2
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    # C = A_left * B_top + A_right * B_bottom
    
    # Each 2x2 * 2x8 multiplication can be done with rank... 
    # 2x2*2x8: split B_top into four 2x2 blocks.
    # For 2x2*2x2, Strassen gives rank 7.
    # So 2x2*2x8 = 2x2*2x2 (4 blocks) = 4*7 = 28? No, that's not right either.
    
    # Let me just do a direct search for a low-rank decomposition.
    # Start with rank 28 and try to verify.
    
    # Actually, let me construct a specific decomposition.
    # 
    # For 2x4 * 4x8, let's split the 4 middle dimension into two 2s:
    # A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x8)
    # C = A1*B1 + A2*B2
    #
    # Now A1*B1 is 2x2 * 2x8. Split B1 into four 2x2 blocks: B1 = [B1a B1b B1c B1d]
    # A1*B1 = A1*B1a + A1*B1b + A1*B1c + A1*B1d
    # Each A1*B1x is 2x2*2x2, which can be done with rank 7 (Strassen).
    # So A1*B1 needs rank 28? No, that's 4*7=28 just for A1*B1.
    # And A2*B2 similarly needs 28. Total 56. That's the same as schoolbook.
    
    # Hmm, that's not helping. Let me think differently.
    #
    # The key insight: we should share the Strassen intermediates.
    #
    # Let's split B (4x8) into two 4x4 blocks: B = [B_left | B_right]
    # C = [A*B_left | A*B_right]
    #
    # For A*B_left (2x4 * 4x4):
    # Split A into [A11 A12] (2x2 each), B_left into [[B11 B12]; [B21 B22]] (2x2 each)
    # A*B_left = [A11 B11 + A12 B21, A11 B12 + A12 B22]
    #
    # Using Strassen for the 2x2*2x2 products, we need 7 scalar multiplications
    # per 2x2*2x2 block. But we have 4 such blocks per A*B_left, so 28.
    # Same for A*B_right: 28. Total 56 again.
    
    # The problem is that Strassen doesn't help when the matrices are too small
    # to benefit from the sharing.
    
    # Let me try a different approach: use the known optimal rank for 2x4*4x8.
    # Actually, let me just search for a valid decomposition with decreasing rank.
    
    # First, let's get the schoolbook solution as a fallback.
    def schoolbook():
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
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    best = schoolbook()
    
    # Now try to find a lower rank decomposition.
    # The tensor T[i,j,k] = 1 for all i,j,k in valid ranges.
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is equivalent to finding a rank-r decomposition of the all-ones tensor
    # of size (n*m) x (m*p) x (p*n) with the specific index mapping.
    
    # Let me try rank 28 with a structured approach.
    # 
    # Idea: Use the 7 Strassen terms for each "chunk" of the computation.
    # 
    # Split the 4x8 matrix B into 4 blocks of 2x2:
    # B = [[B00 B01 B02 B03],
    #      [B10 B11 B12 B13],
    #      [B20 B21 B22 B23],
    #      [B30 B31 B32 B33]]
    # 
    # A = [[A00 A01 A02 A03],
    #      [A10 A11 A12 A13]]
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Let's group the j index: j = 0,1,2,3
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # This is a linear combination. The tensor is all-ones on [0,2)x[0,4)x[0,8).
    
    # Let me try to use a rank-7 decomposition for each 2x2x2 subproblem
    # and see if we can share terms.
    
    # Actually, let me just try to find ANY decomposition with rank < 64.
    # Start with rank 32 and see if we can construct one.
    
    # For now, return the schoolbook solution. The search for better
    # decompositions would require solving a system of polynomial equations,
    # which is complex.
    
    # Let me try a simple improvement: use the fact that the tensor is all-ones
    # to find a lower rank decomposition.
    
    # The all-ones tensor of size a x b x c has rank min(a,b,c) if we allow
    # arbitrary entries, but here the index mapping is specific.
    
    # Actually, for the multiplication tensor, the rank is at least max(n*m, m*p, p*n)
    # = max(8, 32, 16) = 32. So rank >= 32.
    
    # Let me try to construct a rank-32 decomposition.
    # 
    # If we can find U (8x32), V (32x32), W (16x32) such that
    # sum_c U[i*4+j,c] * V[j*8+k,c] * W[k*2+i,c] = 1 for all i,j,k.
    
    # One approach: set V to be the identity (32x32), then we need
    # U[i*4+j,c] * W[k*2+i,c] = 1 for all i,j,k with c=j*8+k... no that doesn't work.
    
    # Let me think about this differently. The condition is:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # If we set c = (j, k) i.e., c ranges over j in [0,4), k in [0,8), that's 32 values.
    # Then U[i*m+j,c] = U[i*4+j, j*8+k] should be 1 if the index matches, 0 otherwise.
    # But c is a single index from 0 to 31. Let c = j*8+k.
    # U[i*4+j, j*8+k] = 1 for all i, else 0? No, U[i*4+j, c] should depend on i and j and c.
    
    # Let's set: U[i*4+j, j*8+k] = 1 for all i, 0 otherwise.
    # V[j*8+k, j*8+k] = 1 (diagonal), 0 otherwise.
    # W[k*2+i, j*8+k] = 1 for all j, 0 otherwise? No, W[k*2+i, c] should give 1 when
    # the product U*V*W = 1.
    
    # With V diagonal: sum_c U[i*4+j,c] * V[c,c] * W[k*2+i,c] = sum_c U[i*4+j,c] * V[c,c] * W[k*2+i,c]
    # If V[c,c]=1 for all c, then we need sum_c U[i*4+j,c] * W[k*2+i,c] = 1.
    # 
    # If U[i*4+j, j*8+k] = 1 and 0 otherwise, and W[k*2+i, j*8+k] = 1 and 0 otherwise,
    # then sum_c U[i*4+j,c] * W[k*2+i,c] = sum_k U[i*4+j, j*8+k] * W[k*2+i, j*8+k]
    # = sum_k 1 * 1 = 8. That's not 1, it's 8.
    
    # So we need to scale. Set U[i*4+j, j*8+k] = 1/8, W[k*2+i, j*8+k] = 1.
    # Then sum = sum_k (1/8)*1 = 8*(1/8) = 1. 
    
    # But wait, we need this to work for ALL (i,j,k), not just the diagonal c.
    # With the above, for a given (i,j,k), the sum is over c=0..31:
    # U[i*4+j,c] * W[k*2+i,c]
    # U[i*4+j,c] is nonzero only when c = j*8+k' for some k'. So c = j*8+k' for k'=0..7.
    # W[k*2+i, c] is nonzero only when c = j'*8+k for some j'. So c = j'*8+k for j'=0..3.
    # The overlap is when c = j*8+k (i.e., k'=k and j'=j).
    # So the sum has exactly one nonzero term: U[i*4+j, j*8+k] * W[k*2+i, j*8+k] = (1/8)*1 = 1/8.
    # That's not 1.
    
    # I made an error. Let me redo.
    # U[i*4+j, c] is nonzero only when c is of the form j*8+k' (for fixed j).
    # Specifically, U[i*4+j, j*8+k'] = 1/8 for all k', 0 otherwise.
    # W[k*2+i, c] is nonzero only when c is of the form j'*8+k (for fixed k).
    # Specifically, W[k*2+i, j'*8+k] = 1 for all j', 0 otherwise.
    # 
    # Product U[i*4+j,c] * W[k*2+i,c] is nonzero when c = j*8+k' = j'*8+k,
    # which means k'=k and j'=j. So c = j*8+k.
    # The product at c = j*8+k is (1/8)*1 = 1/8.
    # Sum over c: only c=j*8+k contributes, so sum = 1/8. Not 1.
    
    # So we need U[i*4+j, j*8+k] = 1 and W[k*2+i, j*8+k] = 1.
    # Then sum = 1*1 = 1. But wait, is U[i*4+j, j*8+k'] = 1 for all k'?
    # Yes, by my definition. And W[k*2+i, j'*8+k] = 1 for all j'.
    # So the product is 1*1 = 1 only at c = j*8+k, and 0 elsewhere.
    # Sum = 1. 
    
    # But we need to check that U and W have the right shapes and that this works.
    # U is 8x32. U[i*4+j, c] for i in [0,2), j in [0,4), c in [0,32).
    # U[i*4+j, j*8+k'] = 1 for all k' in [0,8).
    # So row i*4+j of U has 1s in columns j*8, j*8+1, ..., j*8+7, and 0s elsewhere.
    
    # W is 16x32. W[k*2+i, c] for k in [0,8), i in [0,2), c in [0,32).
    # W[k*2+i, j'*8+k] = 1 for all j' in [0,4).
    # So row k*2+i of W has 1s in columns 0*8+k, 1*8+k, 2*8+k, 3*8+k, and 0s elsewhere.
    # i.e., columns k, 8+k, 16+k, 24+k.
    
    # V is 32x32, diagonal with 1s.
    
    # Let's verify: for (i,j,k), sum_c U[i*4+j,c] * V[c,c] * W[k*2+i,c]
    # = sum_c U[i*4+j,c] * W[k*2+i,c]  (since V is identity)
    # U[i*4+j,c] = 1 iff c in {j*8, j*8+1, ..., j*8+7}
    # W[k*2+i,c] = 1 iff c in {k, 8+k, 16+k, 24+k}
    # Intersection: c = j*8+k (since j*8+k is in both sets).
    # Any other c in the intersection? j*8+k' = j'*8+k implies k'=k and j'=j.
    # So exactly one c: c = j*8+k. Product = 1*1 = 1. Sum = 1. 
    
    # This gives rank 32! And it's exact.
    
    # Let's construct it.
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = j * p + k
                U[i * m + j][c] = 1
                V[c][c] = 1
                W[k * n + i][c] = 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        nm = n * m
        mp = m * p
        pn = p * n
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None
    
    ok, err = verify(U, V, W, n, m, p)
    if ok:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    sb = schoolbook()
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}

# EVOLVE-BLOCK-END
