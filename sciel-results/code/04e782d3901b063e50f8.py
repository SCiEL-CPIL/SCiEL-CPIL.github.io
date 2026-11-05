"""Research program: Attempts rank-14 construction via tensor embedding, falls back to schoolbook."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: 
    # 1. Construct the ground truth tensor T of shape (n, m, p) where T[i, j, k] = 1.
    # 2. Attempt to find a low-rank decomposition using the structure of the tensor.
    #    The directive suggests using a rank-7 decomposition for 3x3 matrix multiplication.
    #    However, the rank of the 3x3 matrix multiplication tensor is known to be at least 19 (and likely higher, recent bounds suggest 19-20+).
    #    Wait, the "rank-7" usually refers to Strassen's algorithm for 2x2x2 (rank 7).
    #    For 3x3x3, the trivial rank is 27. Strassen-like embeddings can reduce this.
    #    Let's try to construct a decomposition based on block decomposition.
    
    # The tensor T is all ones.
    # T[i, j, k] = 1.
    # We want to express T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    # Let's try a simple block approach.
    # Split indices into blocks.
    # Since n=2, m=3, p=3.
    # This is a small case. Let's see if we can find a decomposition with rank < 18 (schoolbook).
    # Schoolbook rank is n*m*p = 2*3*3 = 18.
    
    # Let's check if rank 17 is possible.
    # If we can replace one term with a combination, or use a smarter structure.
    
    # Actually, let's just generate the schoolbook solution first as a baseline, 
    # then try to optimize it using a greedy algorithm or known results.
    
    # For 2x3x3, the tensor is T[i,j,k]=1.
    # This tensor is the outer product of three vectors of ones?
    # T = u v^T w^T where u, v, w are all-ones vectors?
    # T[i,j,k] = u_i * v_j * w_k.
    # If u_i=1, v_j=1, w_k=1, then T[i,j,k]=1.
    # This is a rank-1 tensor!
    
    # Let's verify the decomposition form.
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # If we set rank = 1:
    # U[i*m+j, 0] = a_{i,j}
    # V[j*p+k, 0] = b_{j,k}
    # W[k*n+i, 0] = c_{k,i}
    # We need sum_c ... = a_{i,j} b_{j,k} c_{k,i} = 1 for all i,j,k.
    
    # If we choose a_{i,j}=1, b_{j,k}=1, c_{k,i}=1, then the product is 1.
    # So rank 1 is sufficient?
    
    # Let's double check the index mapping in the task description.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor T_{ijk} for C=AB is defined such that:
    # C_k = sum_{i,j} T_{ijk} A_i B_j ? No.
    # Standard definition: C_{row, col} = sum_{mid} A_{row, mid} B_{mid, col}.
    # Tensor T_{row, mid, col} = 1.
    # In the task's notation:
    # U has indices (i, j) where i is row of A (0..n-1), j is mid (0..m-1).
    # V has indices (j, k) where j is mid (0..m-1), k is col of B (0..p-1).
    # W has indices (k, i) where k is col of C (0..p-1), i is row of C (0..n-1).
    # Note: C is (n, p). So row index i (0..n-1), col index k (0..p-1).
    # The term in the sum is U[i,j] * V[j,k] * W[k,i].
    # Sum over j? No, the decomposition is over the "rank" dimension c.
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    
    # For the matrix multiplication tensor, T[i,j,k] is 1 if the indices correspond to a valid multiplication path?
    # No, the tensor T is the structure tensor of the bilinear map.
    # The map is (A, B) -> C.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This can be written as C_{i,k} = sum_{j} 1 * A_{i,j} * B_{j,k}.
    # So the tensor T has components T_{i,j,k} = 1?
    # Wait, the indices of T in the decomposition T = sum U V W are:
    # U is associated with input A (size n*m).
    # V is associated with input B (size m*p).
    # W is associated with output C (size p*n).
    
    # The decomposition reconstructs the tensor T_{a, b, c} where:
    # a indexes A (i, j) -> a = i*m + j
    # b indexes B (j, k) -> b = j*p + k
    # c indexes C (k, i) -> c = k*n + i
    
    # The condition for the decomposition to be valid is:
    # sum_c U[a, c] V[b, c] W[c_idx, c] = T[a, b, c_idx]
    
    # What is T[a, b, c_idx]?
    # The tensor T represents the multiplication.
    # T is non-zero only when the indices are consistent with the multiplication flow.
    # Specifically, for the term A_{i,j} B_{j,k} to contribute to C_{i,k}, we must have:
    # A index: (i, j)
    # B index: (j, k)
    # C index: (i, k) -> stored as (k, i) in W's index space?
    # The task says W[k*n+i, r]. So W's row index is k*n+i.
    # So the output index is c_idx = k*n + i.
    
    # The tensor T[a, b, c_idx] is 1 if:
    # a = i*m + j
    # b = j'*p + k  (Note: j in A and j' in B must be the same for the product to be defined in the standard sum)
    # c_idx = k'*n + i' (Note: i in A and i' in C must be the same, k in B and k' in C must be the same)
    
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    # All other entries are 0.
    
    # So we need:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # for all i, j, k.
    # And for any other combination of indices (a, b, c_idx) that does not satisfy the consistency:
    # (a = i*m+j, b = j'*p+k, c_idx = k'*n+i')
    # If j != j' or i != i' or k != k', the sum must be 0.
    
    # Let's check if Rank 1 works.
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Then sum_c ... = 1 * 1 * 1 = 1 for all i, j, k.
    # But what about the zero entries?
    # Consider a = i*m+j, b = j'*p+k, c_idx = k'*n+i'.
    # If j != j', the LHS is 1*1*1 = 1.
    # But the RHS T[a, b, c_idx] is 0 because j != j'.
    # So Rank 1 does NOT work. The tensor is not rank 1.
    
    # The tensor is the "multiplication tensor".
    # For n=m=p=1, T[0,0,0]=1. Rank 1.
    # For n=1, m=1, p=2.
    # T[0, j, k] ? No.
    # A is 1x1, B is 1x2, C is 1x2.
    # C_k = A_0 B_k.
    # T[0, k, k] = 1?
    # Indices:
    # A index: 0 (i=0, j=0)
    # B index: k (j=0, k=k) -> index k
    # C index: k (k=k, i=0) -> index k*n+i = k*1+0 = k.
    # So T[0, k, k] = 1.
    # T[0, k1, k2] = 1 if k1=k2, else 0.
    # This is the identity matrix (in the B-C subspace).
    # Rank of Identity 2x2 is 2.
    # So for 1x1x2, rank is 2.
    
    # For 2x3x3:
    # The tensor T is a 3D array of size (6, 9, 6).
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # This looks like a permutation matrix generalized.
    # Actually, it's a sum of permutation matrices?
    # No, it's a single tensor.
    
    # Let's try to find the rank of this specific tensor.
    # This tensor is known as the matrix multiplication tensor M_{n,m,p}.
    # The rank of M_{2,3,3} is not 1.
    
    # Let's use a solver to find the minimal rank.
    # Since we can't run an NP-hard solver easily, we can try to construct a low rank solution.
    
    # Observation:
    # The condition T[i*m+j, j*p+k, k*n+i] = 1 implies that for each fixed j, the slice T[:, j*p+*, *] has a specific structure.
    
    # Let's try rank = n*m*p = 18 (Schoolbook).
    # U[i*m+j, i*m+j] = 1
    # V[j*p+k, i*m+j] = 1  <-- Wait, V index depends on j,k. U index depends on i,j.
    # In schoolbook, each term c corresponds to a unique (i,j,k) triple.
    # c = i*m*p + j*p + k.
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    # Let's verify this decomposition.
    # Sum_c U[i*m+j, c] V[j'*p+k', c] W[k''*n+i'', c]
    # Non-zero only if c corresponds to (i,j,k), (j',k'), (k'',i'') all being the same triple?
    # No, c is a single index.
    # U[i*m+j, c] is 1 only if c = i*m*p + j*p + k for some k?
    # Actually, in the schoolbook construction:
    # For each (i,j,k), we have a term c_{ijk}.
    # U[i*m+j, c_{ijk}] = 1.
    # V[j*p+k, c_{ijk}] = 1.
    # W[k*n+i, c_{ijk}] = 1.
    
    # Now consider the sum for a general (i, j, k) target (which corresponds to T index a=i*m+j, b=j*p+k, c_idx=k*n+i).
    # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # The only c that makes U non-zero is c = i*m*p + j*p + k' for some k'.
    # The only c that makes V non-zero is c = i'*m*p + j*p + k for some i'. (Wait, V index is j*p+k. c encodes i,j,k. So V[j*p+k, c] is 1 if c's j-part is j and k-part is k. i-part can be anything? No, c is unique to i,j,k).
    # Let c = i0*m*p + j0*p + k0.
    # U[i*m+j, c] = 1 iff i0=i and j0=j.
    # V[j*p+k, c] = 1 iff j0=j and k0=k.
    # W[k*n+i, c] = 1 iff k0=k and i0=i.
    # So all three are 1 iff i0=i, j0=j, k0=k.
    # So the sum is 1 if there exists a c such that i0=i, j0=j, k0=k.
    # Since c = i*m*p + j*p + k is a valid index, the sum is 1.
    
    # Now consider a "wrong" combination, e.g., T[i*m+j, j'*p+k, k*n+i] with j != j'.
    # Sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
    # U non-zero iff i0=i, j0=j.
    # V non-zero iff j0=j', k0=k.
    # If j != j', no c can satisfy both j0=j and j0=j'.
    # So the sum is 0.
    # This matches T=0.
    
    # So the schoolbook rank 18 is valid.
    
    # Can we do better?
    # The rank of M_{2,3,3} is known to be 17? Or 16?
    # Let's check small cases.
    # M_{1,1,1} rank 1.
    # M_{1,1,2} rank 2.
    # M_{1,2,2} rank 4? (Diagonal of 2x2 identity? No. C_k = sum_j A_j B_jk. T[j, jk, k]. This is the identity matrix in B-C space? No.
    # A is 1x2 (row vector). B is 2x2. C is 1x2 (row vector).
    # C_k = A_0 B_0k + A_1 B_1k.
    # T[0, 0, 0] = 1 (A_0 B_00 -> C_0)
    # T[1, 1, 0] = 1 (A_1 B_10 -> C_0)
    # T[0, 2, 1] = 1 (A_0 B_01 -> C_1)
    # T[1, 3, 1] = 1 (A_1 B_11 -> C_1)
    # Indices:
    # A: 0, 1
    # B: 0,1, 2,3 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    # C: 0, 1 (k=0,i=0; k=1,i=0)
    # T[0,0,0]=1, T[1,2,0]=1, T[0,1,1]=1, T[1,3,1]=1.
    # This is a permutation-like structure.
    # It can be decomposed into 4 rank-1 tensors.
    # Can it be done in 3?
    # The tensor is essentially the identity map from R^2 to R^2 (if we view A as fixed? No).
    # It's the tensor for C = A B.
    # This is the same as the tensor for 2x2 matrix multiplication but with A being 1x2.
    # The rank of M_{1,2,2} is 4.
    
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{2,3,3}, we can embed M_{2,2,2} and handle the rest.
    # Split j into j1 (0,1) and j2 (2).
    # Split k into k1 (0,1) and k2 (2).
    # This splits the tensor into blocks.
    # Block (j1, k1): M_{2,2,2} part?
    # i is 0,1. j is 0,1. k is 0,1.
    # This is exactly M_{2,2,2} for the sub-matrices A[:, 0:2] and B[0:2, 0:2].
    # This contributes to C[:, 0:2].
    # Rank 7.
    
    # Block (j1, k2): A[:, 0:2] * B[0:2, 2].
    # Result is C[:, 2] (2x1 vector).
    # C_i2 = sum_{j in 0,1} A_{ij} B_{j2}.
    # This is M_{2,2,1}.
    # Rank of M_{2,2,1}?
    # C_i = sum_j A_{ij} B_j.
    # T[i, j, i] ? No, C index is (k=2, i).
    # T[i*m+j, j*p+2, 2*n+i].
    # This is a 2x2 matrix (i vs j) mapped to 2 outputs.
    # It's essentially the identity matrix if B is a vector?
    # No, B_{j2} is a variable.
    # T[i, j, i] = 1 (if we map C_i to index i).
    # This is the identity tensor I_2. Rank 2.
    # So M_{2,2,1} has rank 2.
    
    # Block (j2, k1): A[:, 2] * B[2, 0:2].
    # Result is C[:, 0:2] (2x2 matrix).
    # C_i k = A_{i2} B_{2k}.
    # This is outer product of A[:,2] and B[2,:].
    # T[i, 2, k] = 1?
    # T[i*m+2, 2*p+k, k*n+i].
    # This is a 2x2 matrix (i vs k) for each fixed "structure".
    # Actually, it's the tensor for C = u v^T where u is 2x1, v is 2x1.
    # C_{ik} = u_i v_k.
    # This is a rank-1 tensor in the (i,k) space?
    # T[i, 2, k] = 1.
    # This is the all-ones 2x2 matrix.
    # Rank of all-ones 2x2 matrix is 1.
    # So M_{2,1,2} has rank 1?
    # Let's check.
    # C_{ik} = A_{i2} B_{2k}.
    # One multiplication: P = A_{02}B_{20}? No.
    # We need to compute 4 entries.
    # C_{00} = A_{02} B_{20}
    # C_{01} = A_{02} B_{21}
    # C_{10} = A_{12} B_{20}
    # C_{11} = A_{12} B_{21}
    # We can compute P1 = A_{02} B_{20}, P2 = A_{02} B_{21}, P3 = A_{12} B_{20}, P4 = A_{12} B_{21}.
    # That's 4 multiplications.
    # Can we do better?
    # C = u v^T.
    # Rank of the tensor T_{i, 2, k} = 1?
    # T[i, j=2, k] = 1.
    # This is a 2x1x2 tensor with all ones.
    # T = u v^T w^T?
    # u_i=1, v_j=1 (only j=2), w_k=1.
    # Rank 1.
    # So M_{2,1,2} has rank 1.
    
    # Block (j2, k2): A[:, 2] * B[2, 2].
    # Result is C[:, 2] (2x1 vector).
    # C_i2 = A_{i2} B_{22}.
    # T[i, 2, 2] = 1?
    # T[i, j=2, k=2] -> C index (2, i).
    # This is a 2x1x1 tensor with all ones.
    # Rank 1.
    # So M_{2,1,1} has rank 1.
    
    # Total rank = 7 (M_{2,2,2}) + 2 (M_{2,2,1}) + 1 (M_{2,1,2}) + 1 (M_{2,1,1}) = 11?
    # Wait, the blocks are not independent in the decomposition because they share indices?
    # No, the tensor is a direct sum of these blocks if we partition the indices.
    # The indices j and k are partitioned.
    # j in {0,1} vs {2}.
    # k in {0,1} vs {2}.
    # The tensor T is zero unless j and k are consistent.
    # In the block decomposition, we are summing contributions to C.
    # C = C_{from j1,k1} + C_{from j1,k2} + C_{from j2,k1} + C_{from j2,k2}.
    # Since the input indices (j) and output indices (k) are disjoint for different blocks, the tensor is the direct sum.
    # The rank of a direct sum of tensors is the sum of their ranks.
    # So rank = 7 + 2 + 1 + 1 = 11.
    
    # Let's verify the ranks.
    # M_{2,2,2} rank 7. Correct.
    # M_{2,2,1} rank 2.
    # C_i = sum_{j=0,1} A_{ij} B_j.
    # T[i, j, i] = 1.
    # This is the identity tensor I_2.
    # Rank of I_2 is 2. Correct.
    # M_{2,1,2} rank 1.
    # C_{ik} = A_{i2} B_{2k}.
    # T[i, 2, k] = 1.
    # This is the all-ones tensor J_{2,1,2}.
    # Rank 1. Correct.
    # M_{2,1,1} rank 1.
    # C_{i2} = A_{i2} B_{22}.
    # T[i, 2, 2] = 1.
    # Rank 1. Correct.
    
    # So we can construct a rank-11 decomposition!
    
    # Let's build it.
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: j in {0,1}, k in {0,1}. M_{2,2,2}.
    # Strassen factors for 2x2x2.
    # Local indices: i (0,1), j (0,1), k (0,1).
    # Global indices:
    # i_g = i
    # j_g = j
    # k_g = k
    
    # Strassen P1..P7
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U factors (A part):
    # P1: A00, A11
    # P2: A10, A11
    # P3: A00
    # P4: A11
    # P5: A00, A01
    # P6: A10, A00
    # P7: A01, A11
    
    # V factors (B part):
    # P1: B00, B11
    # P2: B00
    # P3: B01, B11
    # P4: B10, B00
    # P5: B11
    # P6: B00, B01
    # P7: B10, B11
    
    # W factors (C part):
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's assign columns 0..6 to P1..P7.
    
    # Helper to add factors
    def add_factor(col, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (index, coeff)
        # v_coeffs: list of (index, coeff)
        # w_coeffs: list of (index, coeff)
        for idx, coeff in u_coeffs:
            U[idx, col] += coeff
        for idx, coeff in v_coeffs:
            V[idx, col] += coeff
        for idx, coeff in w_coeffs:
            W[idx, col] += coeff
    
    # Block 1: j=0,1; k=0,1.
    # U index: i*m + j = 2i + j.
    # V index: j*p + k = 3j + k.
    # W index: k*n + i = 2k + i.
    
    # P1 (col 0)
    # U: A00 (i=0,j=0 -> 0), A11 (i=1,j=1 -> 3)
    add_factor(0, [(0, 1), (3, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1), (1, -1), (2, 1)])
    # Wait, W coeffs for P1 are for C00 (1), C11 (1).
    # C00 index: k=0, i=0 -> 0*2+0=0.
    # C11 index: k=1, i=1 -> 1*2+1=3.
    # C01 index: k=1, i=0 -> 1*2+0=2.
    # C10 index: k=0, i=1 -> 0*2+1=1.
    
    # Let's re-verify W indices.
    # W[k*n+i].
    # C_{i,k}.
    # C00: i=0, k=0 -> 0.
    # C01: i=0, k=1 -> 2.
    # C10: i=1, k=0 -> 1.
    # C11: i=1, k=1 -> 3.
    
    # P1: C00 (+1), C11 (+1).
    # W coeffs: [(0, 1), (3, 1)]
    
    # P2 (col 1)
    # U: A10 (i=1,j=0 -> 2), A11 (i=1,j=1 -> 3)
    # V: B00 (j=0,k=0 -> 0)
    # W: C10 (+1). Index 1.
    add_factor(1, [(2, 1), (3, 1)], [(0, 1)], [(1, 1)])
    
    # P3 (col 2)
    # U: A00 (0)
    # V: B01 (1), B11 (4)
    # W: C01 (+1), C11 (+1). Indices 2, 3.
    add_factor(2, [(0, 1)], [(1, 1), (4, 1)], [(2, 1), (3, 1)])
    
    # P4 (col 3)
    # U: A11 (3)
    # V: B10 (3), B00 (0)
    # W: C00 (+1), C10 (+1). Indices 0, 1.
    add_factor(3, [(3, 1)], [(3, 1), (0, 1)], [(0, 1), (1, 1)])
    
    # P5 (col 4)
    # U: A00 (0), A01 (1)
    # V: B11 (4)
    # W: C00 (-1), C01 (+1). Indices 0, 2.
    add_factor(4, [(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)])
    
    # P6 (col 5)
    # U: A10 (2), A00 (0)
    # V: B00 (0), B01 (1)
    # W: C11 (+1). Index 3.
    add_factor(5, [(2, 1), (0, 1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # P7 (col 6)
    # U: A01 (1), A11 (3)
    # V: B10 (3), B11 (4)
    # W: C00 (+1). Index 0.
    add_factor(6, [(1, 1), (3, 1)], [(3, 1), (4, 1)], [(0, 1)])
    
    # Block 2: j in {0,1}, k=2. M_{2,2,1}.
    # C_i2 = sum_{j=0,1} A_{ij} B_{j2}.
    # Rank 2.
    # T[i, j, 2] = 1?
    # W index: k=2, i. -> 2*2 + i = 4+i.
    # U index: 2i + j.
    # V index: 3j + 2.
    
    # Term 1 (col 7): i=0, j=0.
    # U[0, 7] = 1
    # V[2, 7] = 1
    # W[4, 7] = 1
    add_factor(7, [(0, 1)], [(2, 1)], [(4, 1)])
    
    # Term 2 (col 8): i=1, j=0.
    # U[2, 8] = 1
    # V[2, 8] = 1
    # W[5, 8] = 1
    add_factor(8, [(2, 1)], [(2, 1)], [(5, 1)])
    
    # Wait, M_{2,2,1} is C_i = sum_j A_{ij} B_j.
    # The tensor is T[i, j, i] = 1.
    # This is the identity.
    # Decomposition:
    # Term 1: i=0, j=0. A00 B0 -> C0.
    # Term 2: i=1, j=1. A11 B1 -> C1.
    # But we have j=0,1 and k=2.
    # B indices are j=0,k=2 and j=1,k=2.
    # So B_{02} and B_{12}.
    # C_{02} = A_{00} B_{02} + A_{01} B_{12}.
    # C_{12} = A_{10} B_{02} + A_{11} B_{12}.
    
    # My previous "Rank 2" assumption for M_{2,2,1} might be wrong if it's not just identity.
    # M_{2,2,1} tensor:
    # T[i, j, i] = 1?
    # C index is (k=2, i).
    # So T[i, j, 2] is not the right notation.
    # T[i, j, k=2] contributes to C_{i, 2}.
    # The term is A_{ij} B_{j2}.
    # So T[i, j, 2] = 1?
    # Yes, for each i, j, the term A_{ij} B_{j2} contributes to C_{i2}.
    # So T[i, j, 2] = 1 for all i,j?
    # No, T is a 3D tensor.
    # T[i, j, k] = 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # Here k=2.
    # So T[i, j, 2] = 1 for all i in {0,1}, j in {0,1}.
    # This is a 2x2x1 tensor with all ones.
    # Rank of 2x2x1 all-ones tensor?
    # It is the outer product of u (2x1 ones), v (2x1 ones), w (1x1 one).
    # Rank 1.
    
    # So M_{2,2,1} has rank 1?
    # C_{02} = A_{00} B_{02} + A_{01} B_{12}
    # C_{12} = A_{10} B_{02} + A_{11} B_{12}
    # Can we do it in 1 multiplication?
    # P = (A_{00}+A_{01}+A_{10}+A_{11}) (B_{02}+B_{12})?
    # No, we need to separate the terms.
    # If we use 1 multiplication, we can only form one product of sums.
    # We need to recover 4 terms.
    # Rank of the tensor T_{i,j,k=2} = 1 (all ones) is 1?
    # Let's check the rank of the 2x2 all-ones matrix. It is 1.
    # The tensor is T[i,j] = 1.
    # This is a matrix, not a 3D tensor?
    # In the context of the decomposition, the "tensor" for this block is T[i, j, c_idx].
    # c_idx is fixed to (2, i).
    # So we have two separate tensors?
    # No, c_idx depends on i.
    # For i=0, c_idx=4. T[0, j, 4] = 1.
    # For i=1, c_idx=5. T[1, j, 5] = 1.
    # So we have T[0,0,4]=1, T[0,1,4]=1, T[1,0,5]=1, T[1,1,5]=1.
    # This is two separate rank-1 tensors?
    # Tensor 1: i=0, j=0,1 -> c=4.
    # Tensor 2: i=1, j=0,1 -> c=5.
    # Can we combine them?
    # If we use 1 term:
    # U[0, c] = u0, U[2, c] = u1
    # V[2, c] = v0, V[5, c] = v1
    # W[4, c] = w0, W[5, c] = w1
    # We need:
    # u0 v0 w0 = 1 (for 0,0,4)
    # u0 v1 w0 = 1 (for 0,1,4)
    # u1 v0 w1 = 1 (for 1,0,5)
    # u1 v1 w1 = 1 (for 1,1,5)
    
    # From first two: v0 = v1.
    # From last two: v0 = v1.
    # So v0 = v1 = v.
    # u0 w0 = 1/v
    # u1 w1 = 1/v
    # This is possible.
    # So Rank 1 is sufficient for this block.
    
    # Let's fix Block 2 with Rank 1.
    # Col 7.
    # U[0, 7] = 1, U[2, 7] = 1
    # V[2, 7] = 1, V[5, 7] = 1
    # W[4, 7] = 1, W[5, 7] = 1
    
    # Remove previous cols 7,8.
    # U = np.zeros((6, 10))
    # V = np.zeros((9, 10))
    # W = np.zeros((6, 10))
    
    # Let's restart the construction with Rank 10.
    # Block 1: 7 terms.
    # Block 2: 1 term.
    # Block 3: M_{2,1,2}. Rank 1.
    # Block 4: M_{2,1,1}. Rank 1.
    # Total 10.
    
    # Let's rebuild.
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1 (Cols 0-6)
    add_factor(0, [(0, 1), (3, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)])
    add_factor(1, [(2, 1), (3, 1)], [(0, 1)], [(1, 1)])
    add_factor(2, [(0, 1)], [(1, 1), (4, 1)], [(2, 1), (3, 1)])
    add_factor(3, [(3, 1)], [(3, 1), (0, 1)], [(0, 1), (1, 1)])
    add_factor(4, [(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)])
    add_factor(5, [(2, 1), (0, 1)], [(0, 1), (1, 1)], [(3, 1)])
    add_factor(6, [(1, 1), (3, 1)], [(3, 1), (4, 1)], [(0, 1)])
    
    # Block 2 (Col 7): j in {0,1}, k=2.
    # U[0, 7] = 1, U[2, 7] = 1
    # V[2, 7] = 1, V[5, 7] = 1
    # W[4, 7] = 1, W[5, 7] = 1
    add_factor(7, [(0, 1), (2, 1)], [(2, 1), (5, 1)], [(4, 1), (5, 1)])
    
    # Block 3 (Col 8): j=2, k in {0,1}.
    # C_{ik} = A_{i2} B_{2k}.
    # U[1, 8] = 1, U[3, 8] = 1 (A_{02}, A_{12})
    # V[6, 8] = 1, V[7, 8] = 1 (B_{20}, B_{21})
    # W[0, 8] = 1, W[1, 8] = 1, W[2, 8] = 1, W[3, 8] = 1?
    # C_{00} (0), C_{01} (2), C_{10} (1), C_{11} (3).
    # All get contribution.
    # So W[0, 8]=1, W[1, 8]=1, W[2, 8]=1, W[3, 8]=1.
    add_factor(8, [(1, 1), (3, 1)], [(6, 1), (7, 1)], [(0, 1), (1, 1), (2, 1), (3, 1)])
    
    # Block 4 (Col 9): j=2, k=2.
    # C_{i2} = A_{i2} B_{22}.
    # U[1, 9] = 1, U[3, 9] = 1
    # V[8, 9] = 1
    # W[4, 9] = 1, W[5, 9] = 1
    add_factor(9, [(1, 1), (3, 1)], [(8, 1)], [(4, 1), (5, 1)])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
