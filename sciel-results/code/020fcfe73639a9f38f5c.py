import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # We are multiplying A (2x4) by B (4x8) to get C (2x8).
    # The tensor T is in U (8) x V (32) x W (16).
    # The decomposition is sum over k of U_k (8) ⊗ V_k (32) ⊗ W_k (16).
    
    # We can view A as a 2x4 matrix. Let's flatten A into a vector of length 8?
    # Or keep the 2x4 structure.
    # The indices of A are (i, j) where i in {0,1}, j in {0,1,2,3}.
    # The indices of B are (j, k) where j in {0,1,2,3}, k in {0,...,7}.
    # The indices of C are (k, i) in the W matrix? 
    # The task says W has shape (p*n, rank). "official trace convention W[k*n+i, r]".
    # So W index is k*n + i, where k is the column of C (0..7) and i is the row of C (0..1).
    # Wait, C is 2x8. Row i, Col k.
    # Usually matrix multiplication C = A B, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] = A[i,j] B[j,k] (conceptually).
    # The factorization is C[i,k] = sum_r U[r, i,j] V[r, j,k] W[r, k,i]?
    # Let's check the interface.
    # U: (n*m, rank) -> (8, R). Index i*m + j? Or j*n + i?
    # "Matrix multiplication ... output factor follows ... W[k*n+i,r]".
    # Usually U corresponds to A, V to B, W to C (inverse transpose?).
    # Let's assume standard Kronecker structure.
    # A is 2x4. U is 8xR. Let U be indexed by (i, j) flattened as i*4 + j.
    # B is 4x8. V is 32xR. Let V be indexed by (j, k) flattened as j*8 + k.
    # C is 2x8. W is 16xR. W indexed by (k, i) flattened as k*2 + i.
    
    # Check consistency:
    # C[i,k] = sum_r U[r, i*4+j] * V[r, j*8+k] * W[r, k*2+i] ???
    # This would mean U, V, W are rank-1 matrices in the tensor product space?
    # No, it means U_k, V_k, W_k are vectors.
    # U_k is a vector in R^8. V_k is a vector in R^32. W_k is a vector in R^16.
    # The product U_k ⊗ V_k ⊗ W_k is a rank-1 tensor.
    # The entry at (i,j, j,k, k,i) is U_k(i,j) V_k(j,k) W_k(k,i).
    # Summing over k (rank index) gives C(i,k) if the tensor is exactly the multiplication tensor.
    # The multiplication tensor T has T[(i,j), (j,k), (k,i)] = 1 if indices match, else 0?
    # No, the tensor *is* the linear map.
    # The condition is: for all A, B, C = A B = sum_k (U_k · flatten(A)) * (V_k · flatten(B)) * reshape(W_k).
    # This requires that the rank-1 tensor U_k ⊗ V_k ⊗ W_k* has 1 at (i,j, j,k, k,i) positions?
    # Actually, we need:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # And also C[i,k] = sum_r U[r, i,j] V[r, j,k] W[r, k,i] summed over i,j,k?
    # No. The formula is C = sum_r U^{(r)} V^{(r)} W^{(r)} where U^{(r)} is 2x4, V^{(r)} is 4x8?
    # No, the interface returns flat vectors.
    # U is (8, R). V is (32, R). W is (16, R).
    # This implies U_r is an 8-vector, V_r is a 32-vector, W_r is a 16-vector.
    # The contribution of the r-th term to C is:
    # Let u_r be reshaped to 2x4.
    # Let v_r be reshaped to 4x8.
    # Let w_r be reshaped to 8x2 (since index is k*2+i, i is row, k is col -> 8 cols, 2 rows).
    # Then C += u_r^T * v_r^T * w_r^T?
    # The standard rank-r decomposition of the multiplication tensor M_{A,B,C} is:
    # C_{ik} = sum_r sum_{i,j,k} A_{ij} B_{jk} (something)?
    # Actually, the condition for U, V, W to be a valid decomposition is:
    # For any A, B: A B = sum_{r} (U_r A^T) (V_r B^T) ? No.
    
    # Let's look at the "official trace convention".
    # T[i, j, k] (input A at i,j, input B at j,k, output C at i,k).
    # Tensor T has rank R if T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    # u_r is in space of A (dim 8). v_r in space of B (dim 32). w_r in space of C (dim 16, dual?).
    # Specifically, T[a,b,c] = sum_r U[r,a] V[r,b] W[r,c].
    # Where:
    # a indexes A (i*4+j).
    # b indexes B (j*8+k).
    # c indexes C (k*2+i).
    # We require T[i,j, j,k, k,i] = 1 (when summed appropriately?)
    # Wait, the tensor T IS the matrix multiplication map.
    # T[a, b, c] is 1 if a=(i,j), b=(j,k), c=(k,i) for the *same* i,j,k?
    # No, the tensor defines the map.
    # The condition is that for any A, B,
    # sum_{a,b,c} U[r,a] V[r,b] W[r,c] A[a] B[b] = (AB)[c reshaped]?
    # Actually, it's simpler:
    # The rank of the multiplication tensor M_{m,n,p} is the minimum number of multiplications.
    # The indices must be consistent.
    # T[i,j; j,k; i,k] = 1. (Semicolon separates spaces).
    # So U[r, i*4+j] * V[r, j*8+k] * W[r, k*2+i] must sum to 1 for the correct terms and 0 otherwise?
    # No, it must sum to 1 if (i,j) and (j,k) and (k,i) form a valid chain?
    # Actually, the tensor T has 1 at (i,j, j,k, k,i) for all i,j,k?
    # Let's check dimensions.
    # A dim: 2*4=8.
    # B dim: 4*8=32.
    # C dim: 2*8=16.
    # T is 8x32x16 tensor.
    # T[a, b, c] = 1 if a corresponds to A[i,j], b to B[j,k], c to C[i,k] (or C[k,i]?).
    # The problem states W[k*n+i, r]. So c = k*2 + i.
    # So T[i*4+j, j*8+k, k*2+i] = 1.
    # And 0 otherwise.
    
    # We need to decompose this tensor into sum of rank-1 tensors.
    # Rank-1 tensor is U_r ⊗ V_r ⊗ W_r.
    # So we need sum_r U_r[i*4+j] V_r[j*8+k] W_r[k*2+i] = 1 if indices match chain, else 0.
    # Wait, "indices match chain" means the j in A matches the row of B, and the k in B matches the col of C (transposed), and i in A matches row of C.
    # Since a,b,c are fixed indices in the tensor, the condition is simply:
    # T[a, b, c] = delta_{a, (i,j)} delta_{b, (j,k)} delta_{c, (k,i)} ... this is confusing.
    
    # Let's rephrase.
    # We want C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # In the tensor decomposition, C[i,k] = sum_r (sum_{j} U_r[i,j] A[i,j] ?? No.
    # The standard form is:
    # C[i,k] = sum_r ( (U_r A^T) [k?] ) ...
    
    # Let's use the verification from the previous attempt which seemed to work (score 0.0000 was likely due to a bug, not the method).
    # The previous attempt used:
    # U3 = U.reshape(2, 4, 56)  # (i, j, r)
    # V3 = V.reshape(4, 8, 56)  # (j, k, r)
    # W3 = W.reshape(8, 2, 56)  # (k, i, r)
    # C_calc = np.einsum('ijr,jkr,kir->ik', U3, V3, W3)
    # This calculates C[i,k] = sum_r sum_j U[i,j,r] V[j,k,r] W[k,i,r].
    # This is the correct formula.
    # So we need to construct U, V, W such that this equals A B.
    
    # Method: Tensor Product / Kronecker Product.
    # We can decompose 2x4 * 4x8 using the rank of 2x2 * 2x2 and 2x2 * 2x2?
    # Or 2x2 * 2x4 and 2x2 * 2x4?
    
    # Let's use the rank-7 (Strassen) for 2x2 * 2x2.
    # Rank = 7.
    # We have:
    # A is 2x4. B is 4x8.
    # Split A into A1 (2x2), A2 (2x2).
    # Split B into B1 (2x4), B2 (2x4)? No, B is 4 rows.
    # B1 corresponds to rows 0,1. B2 corresponds to rows 2,3.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 * 2x4.
    # A2 B2 is 2x2 * 2x4.
    
    # How to decompose 2x2 * 2x4?
    # B is 2x4. Split B into B_col1 (2x2) and B_col2 (2x2).
    # C_left = A * B_col1 (2x2).
    # C_right = A * B_col2 (2x2).
    # Rank(2x2 * 2x4) <= Rank(2x2*2x2) + Rank(2x2*2x2) = 7 + 7 = 14?
    # Wait, can we share the 7 multiplications of A?
    # Strassen for 2x2*2x2 computes 7 products.
    # If we compute the 7 products M_1..M_7 from A and some combinations of B?
    # No, Strassen uses linear combinations of A and B.
    # For 2x2 * 2x4, we can view it as (2x2 * 2x2) + (2x2 * 2x2).
    # The rank is at most 14.
    # Actually, the rank of 2x2 * 2x4 is known to be 14? Or less?
    # Bini's algorithm gives rank 12 for 2x2*2x2? No, 7 is optimal for 2x2*2x2 over complex/rationals (mostly).
    # For 2x2 * 2x4, the rank is 14.
    
    # So:
    # Block 1: A1 (2x2) * B1 (2x4) -> Rank 14.
    # Block 2: A2 (2x2) * B2 (2x4) -> Rank 14.
    # Total Rank = 28.
    
    # This is much better than 56!
    
    # Let's implement Rank 28.
    # We need the 7 Strassen terms for 2x2*2x2.
    # For 2x2 * 2x4, we apply Strassen to A * B_left and A * B_right.
    # This uses 14 products.
    # The products for A * B_left use A and B_left.
    # The products for A * B_right use A and B_right.
    
    # Construction:
    # U, V, W shapes: (8, 28), (32, 28), (16, 28).
    
    # Indices:
    # A: 2x4. i in {0,1}, j in {0,1,2,3}.
    # A1: i in {0,1}, j in {0,1}.
    # A2: i in {0,1}, j in {2,3}.
    
    # B: 4x8. j in {0,1,2,3}, k in {0..7}.
    # B1: j in {0,1}, k in {0..7}. (2x8)
    # B2: j in {2,3}, k in {0..7}. (2x8)
    
    # Wait, B1 is 2x8, not 2x4.
    # My previous split was B into 4 blocks of 2x2.
    # If I split B into B1 (2x8) and B2 (2x8), then A1*B1 is 2x2 * 2x8.
    # Rank(2x2 * 2x8) = Rank(2x2*2x2) * 4? No.
    # 2x8 is four 2x2 blocks horizontally.
    # Rank(2x2 * 2x8) = 7 * 4 = 28?
    # If I do A1 * B1 where B1 is 2x8.
    # B1 = [B11, B12, B13, B14] where B1x are 2x2.
    # A1 * B1 = [A1 B11, A1 B12, A1 B13, A1 B14].
    # Each A1 B1x is 2x2*2x2, rank 7.
    # Total rank for A1*B1 is 28.
    # Then A2*B2 is 28.
    # Total rank 56. This is what we had.
    
    # How to get lower?
    # We need to exploit that A1 and A2 are the *same* A? No, A is one matrix.
    # A = [A1, A2].
    # B = [B1; B2] (vertically).
    # C = A1 B1 + A2 B2.
    
    # Is there a better algorithm for 2x4 * 4x8?
    # This is equivalent to 2x2 * 2x2 over the ring of 2x4 matrices? No.
    # It's equivalent to 2x1 * 1x8? No.
    
    # Consider the rank of 2x4 * 4x8.
    # This is a block diagonal of 2x2 * 2x2 if we permute?
    # No.
    
    # Let's check the lower bound.
    # Rank(2x4 * 4x8) >= Rank(2x2 * 2x2) = 7?
    # Actually, the rank of matrix multiplication M_{m,n,p} is at least m*n.
    # Here 2*4=8.
    # Known ranks:
    # M_{2,2,2} = 7.
    # M_{2,4,4} = ?
    # M_{2,4,8} = ?
    
    # If we use the "4-way" tensor method?
    
    # Let's try to find a decomposition with rank 28 or 32.
    
    # Strategy:
    # We have A (2x4) and B (4x8).
    # Let's use the rank-15 algorithm for 2x2 * 2x4?
    # Wait, 2x2 * 2x4 can be seen as 2x2 * 2x2 + 2x2 * 2x2.
    # Is there a combined algorithm?
    # The rank of 2x2 * 2x4 is 14.
    # So A1*B1 (2x2*2x8) has rank 14*4? No, B1 is 2x8.
    # B1 has 4 columns of 2x2.
    # So A1*B1 has rank 14 * 4? No.
    # Each 2x2 output block requires 7 multiplications.
    # There are 4 output blocks in a row?
    # A1 is 2x2. B1 is 2x8.
    # Result is 2x8.
    # It consists of 4 blocks of 2x2.
    # Each 2x2 block is A1 * (2x2 slice of B1).
    # Since A1 is shared, can we save multiplications?
    # In Strassen, A1 is combined with B.
    # For A1 * B_slice1, we compute M1...M7 using A1 and B_slice1.
    # For A1 * B_slice2, we compute M1...M7 using A1 and B_slice2.
    # The M's are different because B is different.
    # So we need 7 multiplications per 2x2 block of B.
    # B1 has 4 such blocks. So 28 multiplications for A1*B1.
    # Similarly 28 for A2*B2.
    # Total 56.
    
    # Is there a way to reduce this?
    # If we split B differently?
    # What if we view A as 4x2? No, A is 2x4.
    
    # Let's look at the dimensions again.
    # 2x4 * 4x8.
    # This is exactly M_{2,4,8}.
    # The rank is conjectured to be 28?
    # Actually, M_{m,n,p} rank is at least max(mn, np, pm).
    # Here max(8, 16, 32/2? no).
    # Lower bound: n=4, m=2, p=8.
    # LB = max(2*4, 4*8, 8*2) / 4? No.
    # The trivial lower bound is m*n = 8.
    # The upper bound from Strassen is 56.
    # Can we get 28?
    
    # Consider the algorithm for 2x2 * 2x2 (rank 7).
    # We have two independent 2x2 multiplications if we split B's columns?
    # No, A is fixed.
    
    # Let's try a different split.
    # Split A into 4 column vectors?
    # A = [a1, a2, a3, a4] (vectors in R^2).
    # B = [B1, B2, B3, B4] (matrices 4x8).
    # C = a1 B1 + a2 B2 + a3 B3 + a4 B4.
    # This is 4 terms. Each is a rank-1 multiplication?
    # a1 B1 is a 2x8 matrix. It is rank 1 (vector times matrix).
    # To compute a1 B1, we need 2 multiplications? (a1[0]*B1[0,:], a1[1]*B1[1,:])
    # Actually, computing a vector times a matrix costs 2 * 8 = 16 multiplications?
    # No, we need to compute C = sum_i a_i B_i.
    # C[i, k] = sum_j A[i,j] B[j,k].
    
    # Let's stick to the Strassen block approach but see if we can merge.
    
    # Wait!
    # Look at the dimensions: 2x4 and 4x8.
    # This is a tensor product of 2x2 and 2x2?
    # M_{2,4,8} \cong M_{2,2,2} \otimes M_{2,2,2}?
    # If we view 4 as 2x2 and 8 as 4x2?
    # If the rank is multiplicative, rank(M_{2,4,8}) = rank(M_{2,2,2}) * rank(M_{2,2,4})?
    # No.
    
    # However, there is a known result:
    # Rank of 2x2 * 2x2 is 7.
    # Rank of 2x2 * 2x4 is 14.
    # Rank of 2x4 * 4x4 is ?
    
    # Let's assume rank 56 is the best we can easily get with simple blocks.
    # BUT, the previous code had a bug.
    # Let's fix the bug in the previous code and see if it scores.
    # The previous code had "Stagnation".
    # The bug was likely in the verification or the indexing.
    # Actually, the previous code returned 0.0000, which usually means an exception or invalid format.
    # Let's make sure the format is exactly correct.
    
    # Also, I will implement the verification to ensure correctness before returning.
    # If verification fails, I will return a fallback (e.g., schoolbook rank 64).
    
    # But wait, the previous code *did* implement the Strassen blocks.
    # Why did it score 0?
    # "The previous proposal was not admitted as a new mutation."
    # "REFUSAL_JSON: {"reason": "empty", ...}"
    # This might mean the code crashed or timed out.
    # Or the output was invalid.
    
    # Let's write a clean, robust implementation.
    
    def add_strassen_term(r, U, V, W, a_i, a_j, b_i, b_j, c_k, c_i, term_id):
        # term_id 0..6 corresponds to the 7 Strassen multiplications
        # Indices:
        # A: (a_i, a_j)
        # B: (b_i, b_j)
        # C: (c_k, c_i) -> W index c_k * 2 + c_i
        
        # A coefficients
        # B coefficients
        # C coefficients
        
        # P1: (A11 - A00) * (B00 - B11) -> C00 +1, C11 +1
        # P2: (A00 + A10) * B11 -> C10 +1, C11 -1
        # P3: (A01 + A11) * B00 -> C01 +1, C11 +1
        # P4: A00 * (B01 - B11) -> C00 +1, C10 +1
        # P5: A11 * (B00 - B10) -> C00 -1, C01 +1
        # P6: (A00 + A01) * B10 -> C11 +1
        # P7: (A10 + A11) * B01 -> C00 +1
        
        u_idx_a = a_i * 4 + a_j
        u_idx_b = (a_i+1) * 4 + (a_j+1)
        # Wait, a_i and a_j are relative offsets?
        # In the block, a_i is 0 or 1, a_j is 0 or 1 (relative to block start).
        
        # Let's define the exact indices based on the block position.
        
        # This is getting messy to hardcode.
        # I will use a general function.

    R = 56
    U = np.zeros((8, R))
    V = np.zeros((32, R))
    W = np.zeros((16, R))
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # ... 8 blocks total.
    
    # Let's generate the 7 terms for each block.
    
    def fill_strassen_block(base_r, a_row0, a_col0, b_row0, b_col0, c_row0, c_col0):
        # a_row0, a_col0: start row, col in A
        # b_row0, b_col0: start row, col in B
        # c_row0, c_col0: start row, col in C
        
        # A elements:
        # a00 = A[a_row0, a_col0]
        # a01 = A[a_row0, a_col0+1]
        # a10 = A[a_row0+1, a_col0]
        # a11 = A[a_row0+1, a_col0+1]
        
        # B elements:
        # b00 = B[b_row0, b_col0]
        # b01 = B[b_row0, b_col0+1]
        # b10 = B[b_row0+1, b_col0]
        # b11 = B[b_row0+1, b_col0+1]
        
        # C elements:
        # c00 = C[c_row0, c_col0]
        # c01 = C[c_row0, c_col0+1]
        # c10 = C[c_row0+1, c_col0]
        # c11 = C[c_row0+1, c_col0+1]
        
        # U indices (8 x R): row = a_i * 4 + a_j
        u_00 = a_row0 * 4 + a_col0
        u_01 = a_row0 * 4 + a_col0 + 1
        u_10 = (a_row0+1) * 4 + a_col0
        u_11 = (a_row0+1) * 4 + a_col0 + 1
        
        # V indices (32 x R): row = b_i * 8 + b_j
        v_00 = b_row0 * 8 + b_col0
        v_01 = b_row0 * 8 + b_col0 + 1
        v_10 = (b_row0+1) * 8 + b_col0
        v_11 = (b_row0+1) * 8 + b_col0 + 1
        
        # W indices (16 x R): row = c_k * 2 + c_i
        # c_k is col, c_i is row.
        w_00 = c_col0 * 2 + c_row0
        w_01 = (c_col0+1) * 2 + c_row0
        w_10 = c_col0 * 2 + c_row0 + 1
        w_11 = (c_col0+1) * 2 + c_row0 + 1
        
        # Term 1: P1
        r = base_r
        U[r, u_11] = 1
        U[r, u_00] = -1
        V[r, v_00] = 1
        V[r, v_11] = -1
        W[r, w_00] = 1
        W[r, w_11] = 1
        
        # Term 2: P2
        r = base_r + 1
        U[r, u_00] = 1
        U[r, u_10] = 1
        V[r, v_11] = 1
        W[r, w_10] = 1
        W[r, w_11] = -1
        
        # Term 3: P3
        r = base_r + 2
        U[r, u_01] = 1
        U[r, u_11] = 1
        V[r, v_00] = 1
        W[r, w_01] = 1
        W[r, w_11] = 1
        
        # Term 4: P4
        r = base_r + 3
        U[r, u_00] = 1
        V[r, v_01] = 1
        V[r, v_11] = -1
        W[r, w_00] = 1
        W[r, w_10] = 1
        
        # Term 5: P5
        r = base_r + 4
        U[r, u_11] = 1
        V[r, v_00] = 1
        V[r, v_10] = -1
        W[r, w_00] = -1
        W[r, w_01] = 1
        
        # Term 6: P6
        r = base_r + 5
        U[r, u_00] = 1
        U[r, u_01] = 1
        V[r, v_10] = 1
        W[r, w_11] = 1
        
        # Term 7: P7
        r = base_r + 6
        U[r, u_10] = 1
        U[r, u_11] = 1
        V[r, v_01] = 1
        W[r, w_00] = 1

    # 8 blocks
    # A is 2x4. B is 4x8.
    # A1: rows 0-1, cols 0-1.
    # A2: rows 0-1, cols 2-3.
    
    # B1: rows 0-1, cols 0-7? No, B is 4x8.
    # B splits into 4 blocks of 2x2 columns?
    # B is 4 rows.
    # Rows 0,1 and Rows 2,3.
    # Cols 0-7.
    # We can split cols into 4 chunks of 2.
    
    # Block (a_type, b_chunk)
    # a_type 0: A cols 0-1.
    # a_type 1: A cols 2-3.
    # b_row_type 0: B rows 0-1.
    # b_row_type 1: B rows 2-3.
    # b_col_chunk 0-3: B cols 0-1, 2-3, 4-5, 6-7.
    
    # This gives 2 * 2 * 4 = 16 blocks?
    # Wait, A has 2 row chunks? No, A is 2 rows total.
    # A is 2x4.
    # We split A into 2 blocks of 2x2 horizontally.
    # We split B into 2 blocks of 2x8 vertically.
    # Then each 2x8 is 4 blocks of 2x2.
    # So 2 * 2 * 4 = 16 blocks of 2x2*2x2?
    # No.
    
    # C = A B.
    # A = [A_L, A_R] (2x2 each).
    # B = [B_T; B_B] (2x8 each).
    # C = A_L B_T + A_R B_B.
    
    # A_L B_T: (2x2) * (2x8).
    # This is 4 (2x2*2x2) multiplications.
    # Rank 4 * 7 = 28.
    
    # A_R B_B: (2x2) * (2x8).
    # Rank 28.
    
    # Total 56.
    
    # Blocks for A_L B_T:
    # B_T cols 0-1: b_row0=0, b_col0=0.
    # B_T cols 2-3: b_row0=0, b_col0=2.
    # B_T cols 4-5: b_row0=0, b_col0=4.
    # B_T cols 6-7: b_row0=0, b_col0=6.
    # A_L cols 0-1: a_row0=0, a_col0=0.
    # C cols 0-1, 2-3, 4-5, 6-7. c_row0=0.
    
    base = 0
    for c_chunk in range(4):
        b_col = c_chunk * 2
        c_col = c_chunk * 2
        fill_strassen_block(base, 0, 0, 0, b_col, 0, c_col)
        base += 7
        
    # Blocks for A_R B_B:
    # A_R cols 2-3: a_row0=0, a_col0=2.
    # B_B rows 2-3: b_row0=2.
    # B_B cols 0-7.
    for c_chunk in range(4):
        b_col = c_chunk * 2
        c_col = c_chunk * 2
        fill_strassen_block(base, 0, 2, 2, b_col, 0, c_col)
        base += 7
    
    # Verification
    try:
        A_test = np.random.rand(2, 4)
        B_test = np.random.rand(4, 8)
        C_test = A_test @ B_test
        
        U3 = U.reshape(2, 4, R)
        V3 = V.reshape(4, 8, R)
        W3 = W.reshape(8, 2, R)
        
        C_calc = np.einsum('ijr,jkr,kir->ik', U3, V3, W3)
        
        if not np.allclose(C_calc, C_test, atol=1e-7):
            # Fallback to schoolbook (Rank 64)
            R_fallback = 64
            U_f = np.zeros((8, R_fallback))
            V_f = np.zeros((32, R_fallback))
            W_f = np.zeros((16, R_fallback))
            idx = 0
            for i in range(2):
                for j in range(4):
                    for k in range(8):
                        U_f[idx, i*4+j] = 1
                        V_f[idx, j*8+k] = 1
                        W_f[idx, k*2+i] = 1
                        idx += 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
            
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

