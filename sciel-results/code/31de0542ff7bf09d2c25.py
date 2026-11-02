"""Find a low-rank exact tensor decomposition for 3x3 x 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    """Return a dict with keys U, V, W giving a tensor decomposition.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1,
    and for all other (i,j,k) combinations the sum is 0.
    """
    t_start = time.time()
    deadline = t_start + 130  # safety margin before 150s kill

    n, m, p = 3, 3, 4

    # Build the tensor T with indices (i, j, k) where T[i,j,k] = delta_{i,0}*delta_{j,0}*delta_{k,0}? No.
    # Actually the matrix multiplication tensor M[i,j,k] = 1 if we multiply A[i,j] * B[j,k] -> C[i,k].
    # In the decomposition: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Wait, the standard decomposition is:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # Tensor form: T[i, j, k] = 1 if the indices are compatible, i.e., T is the "multiplication tensor"
    # T[i,j,k] = 1 if we can pair A[i,j] with B[j,k]. But in the rank decomposition:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]  -- no, the factors are:
    # U indexed by (i,j) i.e. (n*m), V indexed by (j,k) i.e. (m*p), W indexed by (k,i) i.e. (p*n)
    # and T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) where the multiplication happens.
    # Actually T[i,j,k] = 1 if the three indices form a valid "path" i->j->k, but for matrix mult
    # every (i,j,k) triple has T[i,j,k] = 1? No.
    
    # Let me re-read the contract:
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The tensor being decomposed has entries T[i,j,k] where the matrix mult tensor is:
    # M_{ijk} = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes to C[i,k])
    # Wait no. The tensor is: for fixed i,k, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because every (i,j,k) combination is a valid term in the matrix product.
    
    # So T is an n x m x p all-ones tensor? No, that's not right either.
    # The matrix multiplication tensor M has M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    # For standard matrix multiplication, M[i,j,k] = 1 for ALL i, j, k.
    # Yes! Because C[i,k] = sum_j A[i,j] B[j,k], so every (i,j,k) triple contributes.
    
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And we want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The schoolbook solution uses rank = n*m*p with each term corresponding to one (i,j,k):
    # U[i*m+j, c(i,j,k)] = 1, V[j*p+k, c(i,j,k)] = 1, W[k*n+i, c(i,j,k)] = 1
    
    # But we can do better using Strassen-like ideas or other decompositions.
    
    # For 3x3 x 3x4, the schoolbook rank is 3*3*4 = 36.
    # Known lower bounds and constructions:
    # - For 2x2 x 2x2, rank 7 (Strassen)
    # - For 3x3 x 3x3, rank is at most 24 (or lower?)
    # - For 3x3 x 3x4, we can try to find a rank < 36 decomposition.
    
    # Let me try a simple approach: use the schoolbook decomposition but try to merge terms.
    # Or use a known decomposition for 3x3 x 3x3 and extend it.
    
    # Actually, let me just provide the schoolbook decomposition for now, since I don't have
    # a verified lower-rank decomposition for 3x3 x 3x4.
    
    # But wait - the task says "find a rank-minimal exact tensor decomposition". The score is the rank.
    # If I can find a rank 34 or 33 decomposition, that would be better.
    
    # Let me think about this differently. The tensor T is all-ones of shape (3,3,4).
    # T[i,j,k] = 1 for all i,j,k.
    
    # A rank-1 term is U[i*m+j]*V[j*p+k]*W[k*n+i].
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a 3-way tensor of all ones. The rank of the all-ones tensor of shape (n,m,p) is...
    # Actually, the all-ones tensor has rank 1 if we allow U,V,W to be all-ones vectors!
    # U[i*m+j, c] = 1 for all i,j,c=0
    # V[j*p+k, c] = 1 for all j,k,c=0  
    # W[k*n+i, c] = 1 for all k,i,c=0
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 for all i,j,k.
    # 
    # Wait, that gives rank 1! But that can't be right because the matrix multiplication tensor
    # is NOT the all-ones tensor. Let me re-read.
    
    # The matrix multiplication tensor M_{ijk} is defined such that:
    # (A*B)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor M has M_{ijk} = 1 for all i,j,k? No, that would mean
    # (A*B)_{ik} = sum_j sum_c U_{ij}^c V_{jk}^c W_{ki}^c
    # and if M_{ijk} = 1 for all i,j,k, then the rank-1 all-ones decomposition works.
    
    # But that's clearly wrong because matrix multiplication has structure - the j index links A and B.
    # The tensor M_{ijk} should be 1 only when the indices are "connected" in the right way.
    
    # I think I'm confusing myself. Let me look at the schoolbook construction:
    # For each (i,j,k): U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # This means the tensor entry T[i,j,k] = sum_col U[i*m+j,col]*V[j*p+k,col]*W[k*n+i,col]
    # For the schoolbook, each (i,j,k) has its own column where only that triple gets value 1.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # So the tensor IS all-ones. And the all-ones tensor of shape (3,3,4) does have rank 1!
    # U = ones(9,1), V = ones(12,1), W = ones(12,1)
    # Then T[i,j,k] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1.
    
    # But wait, that would mean matrix multiplication can be done in rank 1, which is absurd.
    # The issue is that the tensor M_{ijk} for matrix multiplication is NOT the all-ones tensor.
    
    # Let me reconsider. The matrix multiplication tensor is:
    # M_{ijk} = 1 if the indices satisfy some condition related to matrix multiplication.
    # Actually, I think the standard definition is that M_{ijk} = 1 for all valid (i,j,k),
    # and the "validity" is just that i,j,k are in range. The structure of matrix multiplication
    # comes from how the indices are shared between the three matrices, not from the tensor values.
    
    # Hmm, but if the tensor is all-ones, then rank 1 works. That can't be right.
    
    # Let me look at this more carefully. In the tensor decomposition for matrix multiplication:
    # (A * B)_{ik} = sum_j A_{ij} B_{jk}
    # This can be written as: C_{ik} = sum_{j,c} A_{ij} B_{jk} * M_{ijk} * W_{ki}^c ... no.
    
    # The standard way to view matrix multiplication as a tensor:
    # Define M as a 3-tensor with M_{ijk} = 1 if and only if the indices (i,j,k) form a valid
    # "path" in the multiplication. For standard matrix multiplication of n x m and m x p matrices,
    # M_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # But then the tensor is all-ones, and rank 1 works. This contradicts the known result that
    # 2x2 x 2x2 matrix multiplication has rank 7.
    
    # I think I'm misunderstanding the tensor decomposition format. Let me re-read the contract.
    
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor M is such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = M[i,j,k]
    
    # For matrix multiplication, M[i,j,k] = 1 for all i,j,k? Or is it something else?
    
    # I believe M[i,j,k] = 1 for all valid (i,j,k), meaning the all-ones tensor. But then rank 1 works.
    
    # UNLESS the tensor is defined differently. Let me think about the 2x2 case.
    # 2x2 x 2x2: U has shape (4, r), V has shape (4, r), W has shape (4, r).
    # U[i*2+j, c] for i,j in {0,1}
    # V[j*2+k, c] for j,k in {0,1}
    # W[k*2+i, c] for k,i in {0,1}
    
    # If M[i,j,k] = 1 for all i,j,k, then U=ones(4,1), V=ones(4,1), W=ones(4,1) gives rank 1.
    # But we know Strassen gives rank 7, and the rank is between 7 and 8.
    
    # So M[i,j,k] is NOT the all-ones tensor. What is it?
    
    # Ah, I think I finally understand. The tensor M[i,j,k] represents the coefficient of
    # A[i,j] * B[j,k] in the product C[i,k]. For standard matrix multiplication, this coefficient
    # is 1 for all i,j,k. So M is indeed all-ones.
    
    # But then why does Strassen work? Because Strassen is not decomposing the all-ones tensor.
    # Strassen is finding a different representation of matrix multiplication.
    
    # I think the key insight is that the tensor decomposition is not unique, and the "matrix
    # multiplication tensor" is a specific tensor that encodes the computation, not just the
    # coefficients.
    
    # Let me look at this from a different angle. The rank of the matrix multiplication tensor
    # for 2x2 x 2x2 is known to be 7. If the tensor were all-ones, its rank would be 1.
    # So the tensor must be something else.
    
    # I think the correct definition is:
    # M[i,j,k] = 1 if i,j,k are all in range, but the tensor is over the space of
    # (A,B) -> C, where the bilinear map is (A,B) -> A*B.
    # The tensor M is the structure tensor of this bilinear map.
    
    # For the bilinear map (A,B) -> A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The structure tensor M has M[i,j,k] = 1 for all i,j,k.
    
    # But this gives rank 1, which is wrong. So I must be misunderstanding the tensor.
    
    # Let me try a completely different approach. I'll just implement the schoolbook decomposition
    # which is guaranteed to be correct, and return that.
    
    rank = n * m * p  # 36 for 3x3 x 3x4
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
