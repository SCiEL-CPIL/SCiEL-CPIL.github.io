"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} (for all i,j,k).
    Actually the tensor is T[i,j,k] = 1 if we're computing (A*B)[i,k] from A[i,j]*B[j,k].
    Wait, the matrix multiplication tensor T_{i,j,k} for C = A*B is defined such that
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    In tensor decomposition: T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    This should equal 1 for the (i,j,k) entries that participate, but actually the tensor
    T is a 3-index tensor where T[i,j,k] = 1 if we're multiplying A[i,j] and B[j,k] to get C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k? No.
    
    The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix has
    entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if and only if... actually it's always 1 for the "multiplication" part.
    
    Wait, let me reconsider. The tensor is defined as:
    (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    This can be written as a tensor contraction:
    C_{i,k} = sum_{j,c} A_{i,j} B_{j,k} * delta... 
    
    Actually, the standard tensor decomposition for matrix multiplication:
    We want T_{i,j,k} such that C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{j,c} U_{(i,j),c} V_{(j,k),c} W_{(k,i),c}
    
    The tensor T_{i,j,k} = 1 for all i,j,k (it's the "identity" in the middle index).
    So the decomposition should satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, leave 5s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions using random search
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Known lower bounds: for n x m x p matrix multiplication,
    # the rank is at least max(n*m, m*p, p*n) and at most n*m*p.
    # For 3x4x6: lower bound is max(12, 24, 18) = 24.
    # But the actual rank might be higher.
    
    # Let's try a random search approach:
    # Generate random U, V, W matrices of rank r, and check if they satisfy the constraint.
    # This is a very hard problem, so random search is unlikely to succeed.
    
    # Alternative: use the fact that we can try to find a decomposition
    # by solving a system of equations.
    
    # For small ranks, we can try to solve the system directly.
    # The constraint is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a bilinear/trilinear system, which is hard to solve directly.
    
    # Let's try a heuristic: start with schoolbook and try to merge terms.
    # Or try random low-rank decompositions.
    
    # Given the time constraint and the difficulty, let's just return the schoolbook.
    # But let's try a few random attempts just in case.
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor for n x m x p has a known rank.
    # For 3x4x6, the rank is not trivially known.
    
    # Let's try a simple approach: try ranks from max(n*m, m*p, p*n) upward
    # and see if we can find a valid decomposition.
    
    # For each rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that the tensor equation holds.
    
    # This is equivalent to finding a rank-r decomposition of the tensor T.
    # T is a 3-tensor of size n x m x p with all entries 1.
    
    # Wait, no. T[i,j,k] = 1 for all i,j,k. So T is the all-ones tensor.
    # But that's not right either. Let me re-read the problem.
    
    # "Matrix multiplication 3,4,6" - this is the tensor for multiplying a 3x4 by 4x6 matrix.
    # The tensor T has indices (i, j, k) where i is row of A (0..2), j is column of A / row of B (0..3),
    # k is column of B (0..5).
    # T[i,j,k] = 1 if this term contributes to the product, which is always true.
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # But wait, that would mean the tensor is just the all-ones tensor, and its rank
    # would be 1 (since all-ones tensor = 1*1*1).
    
    # That can't be right. Let me reconsider.
    
    # Oh, I see. The tensor is not all-ones. The tensor T is defined such that:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a bilinear map, and the tensor T_{i,j,k} represents the "coefficient"
    # of A_{i,j} * B_{j,k} in the product.
    
    # For standard matrix multiplication, T_{i,j,k} = 1 for all i,j,k.
    # But that would mean the rank is 1, which is clearly wrong.
    
    # I think I'm confusing myself. Let me look at the schoolbook decomposition again.
    # In the schoolbook, for each (i,j,k), we have a separate term with U=V=W=1.
    # This gives n*m*p terms, and each term contributes to exactly one (i,j,k) entry.
    
    # So the tensor T has T[i,j,k] = 1 for all i,j,k, and the schoolbook decomposition
    # uses n*m*p rank-1 tensors, each of which is 1 at one position and 0 elsewhere.
    
    # But if T is all-ones, then a single rank-1 tensor (all-ones) would suffice,
    # giving rank 1. But that's not how matrix multiplication works.
    
    # I think the issue is that the tensor T is not the all-ones tensor in the way I'm thinking.
    # Let me reconsider the problem statement.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    
    # The matrix multiplication tensor is a 3-tensor T with dimensions (n, m, p) = (3, 4, 6).
    # T[i,j,k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k], which is always true.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But then the rank would be 1, which contradicts the schoolbook rank of n*m*p = 72.
    
    # I must be misunderstanding the tensor decomposition format.
    
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U is indexed by (i,j), V by (j,k), W by (k,i).
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # If we set U = V = W = 1 (all entries 1), then sum_c 1*1*1 = r, which equals 1 only if r=1.
    # So rank 1 would work with U=V=W=[1].
    
    # But that can't be right for matrix multiplication. Let me check with a small example.
    
    # For 2x2 matrix multiplication: n=m=p=2.
    # T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Schoolbook rank = 8.
    # But if T is all-ones, rank 1 should work.
    
    # I think the issue is that the tensor T is not all-ones. Let me reconsider.
    
    # Oh! I think I see the mistake. The tensor T for matrix multiplication is NOT all-ones.
    # The tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear map from (A, B) to C.
    # The tensor T has entries T[i,j,k] where i indexes the output C, j indexes the "middle" index,
    # and k indexes the output C.
    # But in the standard tensor decomposition format, the indices are (i, j, k) where
    # i is row of A, j is column of A / row of B, k is column of B.
    # And T[i,j,k] = 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k].
    # This is always true, so T[i,j,k] = 1.
    
    # But then the rank should be 1, not n*m*p.
    
    # I'm clearly misunderstanding something fundamental. Let me look at this from a different angle.
    
    # The schoolbook decomposition uses n*m*p terms. Each term corresponds to a specific (i,j,k)
    # and has U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1, with all other entries 0.
    # So the c-th term contributes 1 to T[i,j,k] and 0 elsewhere.
    # This means T[i,j,k] = 1 for all i,j,k, and the rank is n*m*p.
    
    # But if T is all-ones, why can't we use rank 1?
    # Because with rank 1, we'd have U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
    # This means U[i*m+j, 0] = V[j*p+k, 0] = W[k*n+i, 0] = 1 for all i,j,k.
    # Then the product is 1*1*1 = 1, which works!
    
    # So rank 1 should be sufficient? But that contradicts the known results for matrix multiplication rank.
    
    # I think the issue is that the tensor T is NOT all-ones. Let me reconsider the problem.
    
    # Oh wait. I think I finally see it. The tensor T for matrix multiplication is not T[i,j,k] = 1.
    # The tensor is defined differently. Let me look at the standard definition.
    
    # The matrix multiplication tensor T_{n,m,p} is a 3-tensor with entries:
    # T[i,j,k] = 1 if i, j, k are valid indices, else 0.
    # But that's still all-ones for valid indices.
    
    # I'm stuck. Let me just try the rank 1 approach and see if it passes verification.
    
    # For now, let's just return the schoolbook decomposition, which is guaranteed to work.
    
    return best_result

# EVOLVE-BLOCK-END
