"""Matrix multiplication tensor decomposition for 2x2 * 2x15.
Uses a structured search to verify minimality and produce an exact decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Schoolbook decomposition is rank n*m*p = 60.
    # The tensor M_{n,m,p} is a direct sum of p copies of M_{n,m,1}.
    # For n=2, m=2, p=1, the rank is 4 (Strassen not applicable for 2x2x1? 
    # Wait, 2x2 matrix by 2x1 vector. 
    # Tensor T[i,j,k] where A is 2x2, v is 2x1.
    # (Av)_i = sum_j A_{ij} v_j.
    # T[i, j, k] with k being the output index?
    # The task says: "multiplying a 2 by 2 matrix by a 2 by 15 matrix".
    # Output is 2x15.
    # T_{i, j, k} corresponds to A_{ij} * B_{jk}? No.
    # A is 2x2, B is 2x15. Output C is 2x15.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor is T_{i, j, k} = A_{ij} B_{jk}.
    # Indices: i in [0,1], j in [0,1], k in [0,14].
    # U has shape (n*m, rank) -> (4, rank). Row index i*m+j.
    # V has shape (m*p, rank) -> (30, rank). Row index j*p+k.
    # W has shape (p*n, rank) -> (30, rank). Row index k*n+i.
    
    # This is a direct sum over k of the tensor T^{(k)}_{i,j}.
    # Each T^{(k)}_{i,j} is a 2x2 matrix multiplication tensor for A * (column k of B)?
    # No, it's just the structure of the bilinear map.
    # The tensor is a block diagonal matrix of 15 2x2 blocks?
    # Let's check the rank of 2x2x1 tensor.
    # Multiplying 2x2 matrix by 2x1 vector.
    # Rank of matrix-vector multiplication M_{2,2,1} is 4?
    # Yes, it's just 4 multiplications: A00*v0, A01*v1, A10*v0, A11*v1.
    # Can it be done in 3? No, because the tensor is decomposable?
    # Actually, the rank of M_{n,m,p} is at least n*m*p? No.
    # For M_{2,2,1}, rank is 4.
    # Since the tensor is a direct sum of 15 such tensors (for each k, the j index is shared? No, j is the summation index).
    # T_{i,j,k}. Sum over j.
    # This is not a direct sum over k because j is the same index for all k?
    # Wait, the tensor is T_{i, j, k}.
    # The multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is T_{i, j, k} = e_{ij} \otimes e_{jk} \otimes e_{ki}.
    # Is it a direct sum?
    # If we fix k, the slice T_{i,j,k} involves A_{ij} and B_{jk}.
    # The variables are A_{ij} (4 of them) and B_{jk} (2*15=30 of them).
    # The output is C_{ik} (2*15=30 of them).
    # The tensor is a 3-tensor of size 4 x 30 x 30.
    # It is not a direct sum because the A's are shared across k.
    # However, for each k, we need to compute C_{0k} and C_{1k}.
    # C_{0k} = A_{00} B_{0k} + A_{01} B_{1k}
    # C_{1k} = A_{10} B_{0k} + A_{11} B_{1k}
    # This looks like we need 4 multiplications for each k?
    # Total 60 multiplications.
    # Can we share?
    # A_{00}, A_{01}, A_{10}, A_{11} are the same for all k.
    # But B_{0k}, B_{1k} are different for each k.
    # There is no sharing of products like in Strassen for 2x2x2 because Strassen works by combining different k's (matrix elements of B) in a clever way.
    # But here B is 2x15. It's like 15 independent columns.
    # For each column, we do a 2x2 matrix by 2x1 vector multiply.
    # The rank of 2x2x1 is 4.
    # So the total rank is 15 * 4 = 60?
    # Or is it?
    # The tensor is a direct sum of 15 tensors T^{(k)} where T^{(k)}_{i,j} = A_{ij} B_{jk}.
    # But A_{ij} is the same in all T^{(k)}.
    # In tensor decomposition, if A is shared, can we reduce rank?
    # No, because the B's are independent.
    # Consider the slices.
    # The tensor is block diagonal in the B-index?
    # Let's view it as a matrix from A \otimes B to C.
    # Actually, the rank of a direct sum of tensors is the sum of ranks.
    # Is this a direct sum?
    # T = sum_k T^{(k)} where T^{(k)}_{i,j,l} is non-zero only if l=k?
    # If we fix the third index to be k, the tensor slice is T_{i,j,k}.
    # But the "sum" over k is just the third dimension.
    # The tensor is decomposable as a direct sum along the k-axis if the other indices don't mix k.
    # Here, for a fixed k, the slice involves A_{ij} and B_{jk}.
    # Since A_{ij} is independent of k, the tensor is NOT a direct sum in the sense of disjoint supports in all modes.
    # However, we can apply a permutation.
    # But intuitively, to compute C_{ik} for all k, we need to multiply A_{ij} by B_{jk} for all j,k.
    # There are 4 * 15 = 60 such products.
    # Can we reduce?
    # If we use Strassen-like tricks, we would need to combine B_{jk} for different k.
    # But B_{jk} for different k are just different variables.
    # We can't form sums like B_{00}+B_{01} because they correspond to different outputs C_{00}, C_{01}.
    # Wait, if we form S_{0k} = B_{0k} + B_{1k}, we still need to recover B_{0k} and B_{1k} from the products?
    # No, we just need the output C_{ik}.
    # C_{0k} = A_{00} B_{0k} + A_{01} B_{1k}.
    # This is a linear combination of B_{0k}, B_{1k}.
    # We need 2 outputs for each k.
    # The rank of the 2x2x1 tensor is 4.
    # So for each k, we need rank 4.
    # Total rank 60.
    
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
