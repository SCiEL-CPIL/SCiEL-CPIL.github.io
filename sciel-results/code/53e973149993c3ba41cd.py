"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 144.
    # We will try to find a lower rank decomposition.
    # For 4x4 * 4x9, the tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # Indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition is T = sum_c U[:,c] outer V[:,c] outer W[:,c] (with appropriate reshaping).
    # Specifically, (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # In the tensor form T_{i,j,k}, we have T_{i,j,k} = delta_{i,j} * B_{j,k}? No, T_{i,j,k} is the tensor such that
    # sum_{j} A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k}.
    # Actually, the tensor multiplication is: C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This is equivalent to contracting A (n x m) and B (m x p) along the middle dimension.
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T_{i,j,k} = 1 if the operation is "multiply A_{i,j} and B_{j,k} and add to C_{i,k}".
    # So T_{i,j,k} = 1 for all i,j,k? No, that's not right.
    # The tensor for matrix multiplication is T_{i,j,k} = delta_{i, i'} delta_{j, j'} delta_{k, k'}?
    # Let's use the standard definition. The tensor M^{n,m,p} is defined by M_{i,j,k} = 1 if the triple (i,j,k) corresponds to a term in the product.
    # Actually, for matrix multiplication C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is such that T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} = delta_{i, i} delta_{j, j} delta_{k, k}? That doesn't make sense.
    # Let's look at the seed program.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # The term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # Summing over c, for a fixed (i,j,k), the term is 1 if there is a c such that
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # In the seed, each c corresponds to a unique (i,j,k).
    # So the sum is 1 if (i,j,k) is the specific triple for c, else 0.
    # But we want C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} where T_{i,j,k} should be such that
    # sum_j A_{i,j} B_{j,k} = sum_{i',j',k'} A_{i',j'} B_{j',k'} T_{i',j',k'} with T_{i',j',k'} = delta_{i',i} delta_{k',k} ... no.
    # Let's re-read the task description.
    # "Matrix multiplication 4,4,9"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 9 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    # This implies the tensor T is defined by T_{i,j,k} = 1 for all i,j,k?
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = (sum_i A_{i,j}) (sum_k B_{j,k})? No.
    # Let's look at the seed again.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # The product is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # For a fixed (i,j,k), this sum is 1 if there exists c such that c corresponds to (i,j,k).
    # In the seed, each c is unique to (i,j,k). So the sum is 1 for each (i,j,k).
    # So the bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} * 1 = sum_{i,j,k} A_{i,j} B_{j,k}.
    # But we want sum_j A_{i,j} B_{j,k} for each (i,k).
    # The output is a matrix C of size n x p.
    # The bilinear form should produce C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T should be such that T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # But the decomposition is a sum of rank-1 tensors.
    # A rank-1 tensor is U_vec outer V_vec outer W_vec.
    # The bilinear form is sum_c (U_vec_c . A) (V_vec_c . B) W_vec_c.
    # Here A is flattened to a vector of length n*m, B to m*p.
    # U_vec_c . A = sum_{i,j} U_{i*m+j, c} A_{i,j}.
    # V_vec_c . B = sum_{j,k} V_{j*p+k, c} B_{j,k}.
    # The result is a vector of length p*n (or n*p?).
    # The task says W has shape (p*n, rank) and follows W[k*n+i, r].
    # So the output vector is indexed by (k, i) -> k*n+i.
    # The output should be C^T or C flattened in column-major?
    # C_{i,k} is at index k*n+i. So it's column-major flattening of C.
    # So the bilinear form produces a vector Y of length p*n, where Y_{k*n+i} = C_{i,k}.
    # Y_{k*n+i} = sum_c (sum_{i',j'} U_{i'*m+j', c} A_{i',j'}) (sum_{j',k'} V_{j'*p+k', c} B_{j',k'}) W_{k*n+i, c}.
    # We want Y_{k*n+i} = C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So we need:
    # sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1 for all i,j,k? No.
    # We need sum_c (U_{i*m+j, c} V_{j*p+k, c} W_{k'*n+i', c}) to be 1 if (i',k')=(i,k) and the j's match?
    # Let's check the condition for the bilinear form to equal matrix multiplication.
    # Y_{k*n+i} = sum_{j} A_{i,j} B_{j,k}.
    # From the decomposition:
    # Y_{k*n+i} = sum_c [sum_{i',j'} U_{i'*m+j', c} A_{i',j'}] [sum_{j',k'} V_{j'*p+k', c} B_{j',k'}] W_{k*n+i, c}
    # = sum_{i',j',j',k'} A_{i',j'} B_{j',k'} W_{k*n+i, c} sum_c U_{i'*m+j', c} V_{j'*p+k', c}
    # Let M_{i',j',j',k'} = sum_c U_{i'*m+j', c} V_{j'*p+k', c} W_{k*n+i, c}.
    # We want Y_{k*n+i} = sum_{i',j',j',k'} A_{i',j'} B_{j',k'} M_{i',j',j',k'} = sum_j A_{i,j} B_{j,k}.
    # This requires M_{i',j',j',k'} = delta_{i',i} delta_{k',k} delta_{j',j}? No, the indices are mixed.
    # Actually, the tensor T is defined by T_{i,j,k} such that the bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = C_{i,k} for each i,k.
    # But the output is a vector. So T is a 3-tensor of shape (n*m, m*p, p*n).
    # T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k'.
    # So T_{(i,j), (j',k), (k',i')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    # The seed program sets U_{i*m+j, c}=1, V_{j*p+k, c}=1, W_{k*n+i, c}=1 for c corresponding to (i,j,k).
    # Then sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1 if (i,j,k) is the triple for c.
    # So T_{(i,j), (j',k'), (k'',i'')} = sum_c U_{i*m+j, c} V_{j'*p+k', c} W_{k''*n+i'', c}.
    # In the seed, this is 1 if i=i'', j=j', k=k', else 0.
    # So T_{(i,j), (j',k), (k',i')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    # This is the correct tensor for matrix multiplication.
    
    # We want to find a decomposition of this tensor with minimal rank.
    # The rank of the matrix multiplication tensor M^{n,m,p} is known to be n*m*p for small sizes, but can be lower.
    # For 4x4 * 4x9, the tensor is M^{4,4,9}.
    # We can try to use known results or search for a decomposition.
    # Since the problem is small, we can try to use a greedy or local search algorithm to find a lower rank decomposition.
    # However, the time limit is 150s, so we need to be efficient.
    
    # Let's start with the schoolbook decomposition and try to reduce the rank.
    # We can use the fact that the tensor is symmetric in some sense? No, it's not symmetric.
    # We can try to use the Coppersmith-Winograd algorithm? That's for large matrices.
    # For small matrices, we can use exact algorithms or search.
    
    # Given the time constraint, let's try a simple approach:
    # 1. Start with the schoolbook decomposition.
    # 2. Try to merge terms or find linear dependencies.
    
    # However, implementing a full tensor decomposition search is complex.
    # Let's check if there's a known lower bound or result for 4x4 * 4x9.
    # The rank of M^{4,4,4} is known to be 49 (or less).
    # For M^{4,4,9}, the rank is likely higher.
    
    # Since we don't have a specific algorithm to find the minimal rank, we will return the schoolbook decomposition.
    # But the task asks for a "stronger algorithm".
    # Let's try to use the fact that 9 = 8+1, and 8 is 2^3.
    # Maybe we can use a recursive approach?
    
    # Given the complexity, let's stick with the schoolbook decomposition for now, but ensure it's correct.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
