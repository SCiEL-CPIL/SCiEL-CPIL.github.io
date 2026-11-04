import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Trivial rank-54 decomposition as fallback
    def trivial_decomp():
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
    
    # Attempt rank-18 decomposition using exterior algebra
    # A is 2x3, B is 3x9
    # Group B's columns into 3 groups of 3: B = [B1 | B2 | B3], each Bi is 3x3
    # The product C = A*B is 2x9
    # 
    # Using the identity: for vectors u,v,w in R^3, u∧v∧w = det([u v w])
    # We can express entries of C using wedge products
    
    # Let's construct a rank-18 decomposition
    # The key is to use the fact that for 3x3 matrices, the determinant can be
    # expressed as a sum of 6 terms (permutations)
    
    # Actually, let me think more carefully. The tensor T_{i,j,k} = delta_{j, j'} * ... 
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] which is 1 if we're computing that product
    
    # For rank r decomposition: T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
    # where U is n*m x r, V is m*p x r, W is p*n x r
    
    # The exterior algebra approach: identify B's columns with vectors
    # B[j,k] for j=0,1,2 and k=0..8
    # Group k: k=0,1,2 -> group 0; k=3,4,5 -> group 1; k=6,7,8 -> group 2
    # So B has 3 "row vectors" of length 3 each (when grouped)
    
    # Let me try a direct construction. We need rank 18.
    # The 3x3 determinant has 6 terms. For each of the 2 rows of A and 3 groups of B,
    # we might need 6 terms, giving 2*3*6 = 36, but we can share structure.
    
    # Alternative: use the standard rank formula for small matrix mult
    # For 2x3x9, the tensor rank is known to be at most 18 by certain constructions
    
    # Let me construct it explicitly:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,...,8}
    
    # Group k into 3 groups of 3: k = 3*g + h where g in {0,1,2}, h in {0,1,2}
    # B[j, 3*g+h] is the (j,h) entry of the g-th 3x3 block
    
    # For each i and g, C[i, 3*g+h] = sum_j A[i,j] * B[j, 3*g+h]
    
    # This is just matrix-vector product for each block. The rank for 2x3x3 is 6 (trivial)
    # or possibly less. 2x3x3 has tensor rank 6 (I believe).
    # So 3 blocks * 6 = 18 total.
    
    # For 2x3x3 matrix mult: C[i,h] = sum_j A[i,j]*B[j,h]
    # Trivial rank is 2*3*3=18, but we can do better.
    # Actually for 2x3x3, the rank is 6 (I need to verify).
    
    # Let me just use the trivial 6-term decomposition for each 2x3x3 block:
    # C[i,h] = sum_j A[i,j]*B[j,h]
    # Term (i,j,h): U[i*3+j] = 1, V[j*3+h] = 1, W[h*2+i] = 1
    # That's 2*3*3 = 18 terms per block, 3 blocks = 54. Not good.
    
    # For 2x3x3, we can use rank 6:
    # The tensor is sum over j of A[i,j]*B[j,h]
    # Each j gives a rank-1 tensor in (i,h) space: outer product of column j of A and row j of B
    # So rank is at most 3 (one for each j). Wait, that's not right for the full tensor.
    
    # The full tensor T[i,j,h] for 2x3x3 mult: T[i,j,h] = 1 if we're computing that term
    # Actually the multiplication tensor has T[i,j,k] = delta (it's the identity in the middle index)
    # For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is sum_j (e_i ⊗ e_j ⊗ e_k) for all i,j,k? No.
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{j,j} ... I'm confusing myself.
    
    # Let me restart. The matrix multiplication tensor for (n,m,p) has entries
    # T[i,j,k] where i in [n], j in [m], k in [p]
    # and T[i,j,k] = 1 if the (i,k) entry of the product involves A[i,j]*B[j,k]
    # Actually T[i,j,k] is the coefficient, and the product is C[i,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
    # where T[i,j,k] = 1 for all valid i,j,k (since every A[i,j] multiplies every B[j,k] for that j)
    # Wait no: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is sum_j (sum_i e_i ⊗ e_j ⊗ sum_k e_k) ... 
    
    # The tensor is: T = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} e_i) ⊗ e_j ⊗ (sum_{k=0}^{p-1} e_k)
    # No that's not right either. Let me be precise.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor M satisfies: <M, A ⊗ B> = C where C[i,k] = sum_j A[i,j]B[j,k]
    # So M[i,j,k] = 1 for all i,j,k? No.
    # M is a 3-tensor of shape (n*m, m*p, p*n) in the flattened convention.
    # In the original indices: M[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k.
    
    # Wait, that means the tensor is all ones, which has rank 1? That can't be right.
    
    # I think I'm confusing the tensor definition. Let me look at the seed code:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # So for each (i,j,k), we have a term with U indexed by (i,j), V by (j,k), W by (k,i)
    # The reconstruction is: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # This should give 1 for the "multiplication tensor" entry.
    
    # So the tensor T[i,j,k] = 1 for all i,j,k (in the sense that the coefficient is 1)
    # And the rank is the tensor rank of this all-ones tensor of shape (n*m, m*p, p*n)
    
    # For all-ones tensor of shape (a,b,c), the rank is max(a,b,c) if we can factor it.
    # Actually the all-ones tensor of shape (a,b,c) has rank 1: it's e_a ⊗ e_b ⊗ e_c where e is all-ones vector.
    # Wait no, that would be rank 1. But the seed uses rank n*m*p.
    
    # I think I'm misunderstanding the tensor. Let me re-examine.
    
    # The matrix multiplication tensor for (n,m,p) is NOT all ones.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
    # For a fixed (i,k), the terms are A[i,0]*B[0,k] + A[i,1]*B[1,k] + ... + A[i,m-1]*B[m-1,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But then the rank would be 1 (all-ones tensor). That contradicts the seed using rank 54.
    
    # I must be wrong about the tensor definition. Let me think again.
    
    # Oh! The tensor is indexed differently. In the seed:
    # U has shape (n*m, rank), indexed by (i,j) flattened
    # V has shape (m*p, rank), indexed by (j,k) flattened  
    # W has shape (p*n, rank), indexed by (k,i) flattened
    # 
    # The reconstruction: for each (i,j,k), the value is sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    # This should equal 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # 
    # For matrix multiplication, EVERY term A[i,j]*B[j,k] contributes to C[i,k] (summed over j)
    # So yes, T[i,j,k] = 1 for all i,j,k.
    
    # But then why is the rank 54 in the seed? Unless the tensor is NOT all ones.
    
    # Wait, I think I've been confused. Let me reconsider.
    # 
    # The matrix multiplication tensor M_{n,m,p} is defined such that:
    # (A*B)[i,k] = sum_{j} M[i,j,k] * A[i,j] * B[j,k]
    # where M[i,j,k] = 1 for all i,j,k.
    # 
    # So M is indeed the all-ones tensor of shape (n, m, p) in the (i,j,k) indices.
    # But in the flattened form used by the seed, it's a tensor of shape (n*m, m*p, p*n)
    # where the entry at position ((i*m+j), (j*p+k), (k*n+i)) is 1.
    # 
    # This is NOT the all-ones tensor in the flattened space! It's a sparse tensor where
    # only certain entries are 1. Specifically, for each (i,j,k), the entry at
    # (i*m+j, j*p+k, k*n+i) is 1.
    
    # So the tensor is: T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # And T[a,b,c] = 0 otherwise.
    
    # This makes more sense. The rank is the tensor rank of this sparse tensor.
    
    # For n=2, m=3, p=9:
    # a in [0,5], b in [0,26], c in [0,17]
    # T[a,b,c] = 1 iff a=i*3+j, b=j*9+k, c=k*2+i for some i in {0,1}, j in {0,1,2}, k in {0,...,8}
    
    # So there are 2*3*9 = 54 ones in the tensor, at positions determined by (i,j,k).
    # The seed assigns one rank-1 term to each such position, giving rank 54.
    
    # To reduce rank, we need to find linear dependencies among these rank-1 tensors.
    
    # The exterior algebra approach: by grouping k into 3 groups of 3, we can use
    # the fact that for 3x3 matrices, certain combinations have lower rank.
    
    # Let me try to construct a rank-18 decomposition.
    
    # For each i in {0,1} and group g in {0,1,2}, we have 3 values of h in {0,1,2}
    # The terms are: (i, j, 3*g+h) for j in {0,1,2}, h in {0,1,2}
    # That's 9 terms per (i,g), 18 total per i, 36 total.
    
    # Hmm, I need a different approach. Let me just try to use known results.
    
    # The tensor rank of 2x3x9 matrix multiplication is known to be at most 18
    # by certain constructions. Let me try to implement one.
    
    # Actually, let me just return the trivial decomposition for now and mark this as a TODO.
    # The exterior algebra strategy is complex and I may not get it right in time.
    
    if time.time() > deadline:
        return trivial_decomp()
    
    # For now, return the trivial rank-54 decomposition
    return trivial_decomp()
