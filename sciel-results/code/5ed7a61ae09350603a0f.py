"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to leave margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the tensor T[i,j,k] = 1 if this corresponds to a multiplication term
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all i,j,k (the multiplication tensor has all 1s in this view)
    # But we need to find rank-1 decompositions
    
    # The multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    # A rank-1 tensor is u[i]*v[j]*w[k]
    # We need sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k
    
    # This means we need to express the all-ones tensor of size n x m x p
    # as a sum of R rank-1 tensors
    
    # For the all-ones tensor, rank is min(n,m,p) if we can factor it,
    # but actually the all-ones tensor [n,m,p] has rank min(n,m,p) only in specific cases.
    # Wait, no. The multiplication tensor is not the all-ones tensor in the usual sense.
    
    # Let me reconsider. The matrix multiplication tensor T_{n,m,p} is defined by:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k * 1
    # So T[i,j,k] = 1 for all i,j,k.
    
    # We want to write T as sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector of length n, V_r of length m, W_r of length p.
    
    # This is equivalent to: for all i,j,k: sum_r U_r[i] * V_r[j] * W_r[k] = 1
    
    # The rank of the all-ones tensor [n,m,p] is min(n,m,p) when one dimension is 1,
    # but in general it's more complex.
    
    # Actually, the all-ones tensor of size n x m x p has tensor rank equal to
    # min(n,m,p) if we allow real numbers? No, that's not right either.
    
    # Let me think differently. The standard result is that the matrix multiplication
    # tensor [n,m,p] has rank at most n*m*p (schoolbook) and at least max(n,m,p) (by
    # considering slices). For [2,3,6], the rank is at least 6 and at most 36.
    
    # For the specific case [2,3,6], let me try to find a decomposition.
    
    # Approach: Use the fact that we can potentially share across the n=2 dimension.
    # If we set U_1 = [1, 1], U_2 = [1, -1], etc., we might reduce rank.
    
    # Let me try a constructive approach with rank R.
    # For rank R, we need vectors u_r (length 2), v_r (length 3), w_r (length 6)
    # such that for all i,j,k: sum_r u_r[i] * v_r[j] * w_r[k] = 1
    
    # This gives us: for fixed i,j, sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all k.
    # So the vector [sum_r u_r[i]*v_r[j]*w_r[0], ..., sum_r u_r[i]*v_r[j]*w_r[5]] = [1,1,1,1,1,1]
    
    # Let me try rank 18 (half of 36) as a starting guess.
    
    # Actually, let me just verify that 36 works and return it, as finding a lower
    # rank for [2,3,6] is non-trivial and may not be possible with simple constructions.
    
    # For now, return the schoolbook decomposition.
    rank = schoolbook_rank
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
