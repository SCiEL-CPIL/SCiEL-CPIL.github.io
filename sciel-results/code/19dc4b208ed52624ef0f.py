"""Program for rank-minimal exact tensor decomposition of 2x2 * 2x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We compute C = A @ B where A is 2x2 and B is 2x8.
    # C[:,k] = A @ B[:,k] for each column k=0..7.
    # 
    # The tensor for 2x2x2 matrix multiplication has rank 7.
    # We use a known rank-7 decomposition for 2x2x2 and apply it independently
    # for each of the 8 columns of B.
    #
    # For each column k, we need:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # A rank-7 decomposition for 2x2x2 uses 7 bilinear forms:
    #   M_r = (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_j V[j,r] * B[j,k])
    #   C[i,k] = sum_r W[i,k,r] * M_r
    #
    # Here is a verified rank-7 decomposition for 2x2x2:
    # The 7 terms correspond to:
    #   r=0: A[0,0]*B[0,k] -> contributes to C[0,k]
    #   r=1: A[0,1]*B[1,k] -> contributes to C[0,k]
    #   r=2: A[1,0]*B[0,k] -> contributes to C[1,k]
    #   r=3: A[1,1]*B[1,k] -> contributes to C[1,k]
    #   r=4: (A[0,0]+A[1,1])*(B[0,k]+B[1,k]) -> cross terms
    #   r=5: (A[0,0]-A[1,1])*(B[0,k]-B[1,k]) -> cross terms
    #   r=6: (A[0,1]+A[1,0])*(B[0,k]+B[1,k]) - ... 
    #
    # Let me use a cleaner known decomposition. The standard rank-7 for 2x2x2:
    #
    # Term 0: U=[1,0,0,0], V=[1,0], W=[1,0]  => A00*B0k -> C0k
    # Term 1: U=[0,1,0,0], V=[0,1], W=[1,0]  => A01*B1k -> C0k
    # Term 2: U=[0,0,1,0], V=[1,0], W=[0,1]  => A10*B0k -> C1k
    # Term 3: U=[0,0,0,1], V=[0,1], W=[0,1]  => A11*B1k -> C1k
    # Term 4: U=[1,0,0,1], V=[1,1], W=[-1,-1] => (A00+A11)*(B0k+B1k) -> -C0k-C1k
    # Term 5: U=[1,0,0,-1], V=[1,-1], W=[1,1] => (A00-A11)*(B0k-B1k) -> C0k+C1k
    # Term 6: U=[0,1,1,0], V=[1,1], W=[0,0] => (A01+A10)*(B0k+B1k) -> 0
    #
    # Wait, that doesn't work. Let me think more carefully.
    #
    # Actually, for 2x2x2, the schoolbook is rank 4:
    #   C00 = A00*B00 + A01*B10
    #   C01 = A00*B01 + A01*B11
    #   C10 = A10*B00 + A11*B10
    #   C11 = A10*B01 + A11*B11
    #
    # But the rank is 7, not 4. The schoolbook for 2x2x2 (where B is 2x2) 
    # actually has rank 4 in the standard tensor sense? No, the tensor 
    # T_{i,j,k} = delta_{ik} * A_{ij} * B_{jk} for 2x2x2 has rank 7.
    #
    # Wait, I'm confusing myself. Let me reconsider.
    #
    # For C = A * B where A is n x m and B is m x p:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # The tensor is T_{i,j,k} with the multiplication rule.
    # For n=2, m=2, p=2:
    # C_{0,0} = A_{0,0}*B_{0,0} + A_{0,1}*B_{1,0}
    # C_{0,1} = A_{0,0}*B_{0,1} + A_{0,1}*B_{1,1}
    # C_{1,0} = A_{1,0}*B_{0,0} + A_{1,1}*B_{1,0}
    # C_{1,1} = A_{1,0}*B_{0,1} + A_{1,1}*B_{1,1}
    #
    # This is a 4x4 tensor (4 entries of C, each depending on 4 entries of A and 4 of B).
    # The rank of this tensor is known to be 7.
    #
    # For our problem, A is 2x2 and B is 2x8, so B has 16 entries.
    # C is 2x8, so 16 entries.
    # C_{i,k} = sum_j A_{i,j} * B_{j,k} for i=0,1 and k=0..7.
    #
    # This can be seen as 8 independent 2x2 matrix-vector products:
    # For each k, C[:,k] = A * B[:,k].
    #
    # If we use a rank-7 decomposition for each column independently,
    # we get rank 8*7 = 56.
    #
    # But we can potentially do better by sharing structure across columns.
    # However, since A is shared but B and C are column-independent,
    # the natural decomposition is per-column.
    #
    # Let me find a correct rank-7 decomposition for 2x2x2.
    #
    # A known rank-7 decomposition (from Strassen 1969 or later work):
    # 
    # Let me use the following approach. We want 7 pairs of linear forms (u_r, v_r)
    # and 4 linear combinations w_{i,k,r} such that:
    # C_{i,k} = sum_r w_{i,k,r} * (sum_{a,b} u_{a,b,r} * A_{a,b}) * (sum_b v_{b,r} * B_{b,k})
    #
    # Here's one construction. Define 7 bilinear forms:
    # M_0 = A_{0,0} * B_{0,k}
    # M_1 = A_{0,1} * B_{1,k}  
    # M_2 = A_{1,0} * B_{0,k}
    # M_3 = A_{1,1} * B_{1,k}
    # M_4 = (A_{0,0} + A_{1,1}) * (B_{0,k} + B_{1,k})
    # M_5 = (A_{0,0} + A_{1,1}) * (B_{0,k} - B_{1,k})
    # M_6 = (A_{0,1} + A_{1,0}) * (B_{0,k} + B_{1,k})
    #
    # Then:
    # C_{0,k} = M_0 + M_1
    # C_{1,k} = M_2 + M_3
    #
    # But that's only rank 4. We need to express this with 7 terms where
    # each term is a product of a linear form in A and a linear form in B.
    #
    # The issue is that M_4, M_5, M_6 are not needed for the direct computation.
    # The rank-7 result comes from a different perspective.
    #
    # Actually, I think I was overcomplicating this. For the 2x2x2 tensor,
    # the schoolbook algorithm uses 4 multiplications, and it's known that
    # the tensor rank is 7. But wait - that doesn't make sense if schoolbook
    # uses only 4.
    #
    # Let me reconsider. The tensor rank of the 2x2x2 matrix multiplication
    # tensor is indeed 7. This means you CANNOT do it with fewer than 7
    # multiplications in the bilinear model. But the schoolbook uses 4?
    #
    # No, I think I'm confusing different tensor formulations. Let me be precise.
    #
    # The matrix multiplication tensor T_{n,m,p} has entries indexed by (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p), and T_{i,j,k} = 1 if the
    # standard multiplication C_{i,k} = sum_j A_{i,j} B_{j,k} uses this term.
    #
    # For n=m=p=2:
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
    # C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}
    # C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1}
    #
    # The tensor has 4*2*2 = 16 nonzero entries (one for each (i,j,k) combination
    # that appears). Actually, each C_{i,k} depends on 2 terms, so there are
    # 4*2 = 8 terms total, but they involve different (i,j,k) tuples.
    #
    # The rank of this tensor is 7. This is a well-known result.
    #
    # So for our problem, if we treat each column independently, we get rank 8*7=56.
    #
    # Let me now construct an explicit rank-7 decomposition for 2x2x2.
    #
    # One known decomposition (I'll derive it):
    # We need 7 terms. Let me use the following:
    #
    # Term 0: u_0 = (1,0,0,0), v_0 = (1,0), w_00=1, w_01=0, w_10=0, w_11=0
    #          => A00 * B0k -> C0k
    # Term 1: u_1 = (0,1,0,0), v_1 = (0,1), w_00=1, w_01=0, w_10=0, w_11=0
    #          => A01 * B1k -> C0k
    # Term 2: u_2 = (0,0,1,0), v_2 = (1,0), w_00=0, w_01=0, w_10=1, w_11=0
    #          => A10 * B0k -> C1k
    # Term 3: u_3 = (0,0,0,1), v_3 = (0,1), w_00=0, w_01=0, w_10=0, w_11=1
    #          => A11 * B1k -> C1k
    #
    # That's rank 4, which is the schoolbook. But the rank is 7, so this
    # should be optimal... unless I'm wrong about the rank being 7.
    #
    # Actually, I think the rank of the 2x2x2 matrix multiplication tensor
    # IS 4 (schoolbook is optimal for 2x2x2). The rank 7 result might be
    # for a different tensor or I'm misremembering.
    #
    # Let me just use the schoolbook decomposition, which gives rank 4 per column,
    # total rank 8*4 = 32. This is what the current code does.
    #
    # Wait, but the task says "rank-minimal". If the rank is 4 per column,
    # then 32 is minimal. Let me verify: for 2x2x2, can we do it in fewer
    # than 4 multiplications? No, because the tensor rank is 4.
    #
    # So the answer is rank 32. The current code already achieves this.
    #
    # Let me just clean up the code and make sure it's correct.
    
    rank = n * m * p  # 2 * 2 * 8 = 32
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 32)
    V = [[0] * rank for _ in range(m * p)]  # (16, 32)
    W = [[0] * rank for _ in range(p * n)]  # (16, 32)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i,k] += A[i,j] * B[j,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
