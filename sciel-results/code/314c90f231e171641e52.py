import numpy as np
import time

# Pre-computed Strassen factors for 2x2 matrix multiplication.
# Shapes: (2, 2, 7)
# U2[i, j, r] corresponds to the coefficient of A_ij in the r-th product M_r
# V2[j, k, r] corresponds to the coefficient of B_jk in the r-th product M_r
# W2[k, i, r] corresponds to the coefficient of M_r in C_ki (Note: Trace convention W[k*n+i, r])
# 
# Standard Strassen Products:
# M0 = (A00 + A11)(B00 + B11)
# M1 = (A10 + A11)B00
# M2 = A00(B01 - B11)
# M3 = A11(B10 - B00)
# M4 = (A00 + A01)B11
# M5 = (A10 - A00)(B00 + B01)
# M6 = (A01 - A11)(B10 + B11)

def _get_strassen_2x2():
    """
    Returns U2, V2, W2 as numpy arrays of shape (2, 2, 7).
    Indices: [dim1, dim2, rank]
    """
    U2 = np.zeros((2, 2, 7), dtype=int)
    V2 = np.zeros((2, 2, 7), dtype=int)
    W2 = np.zeros((2, 2, 7), dtype=int)
    
    # r=0: M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1: M1 = (A10+A11)B00 -> C00 += M1, C11 -= M1
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 0, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2: M2 = A00(B01-B11) -> C01 += M2, C11 += M2
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[0, 1, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3: M3 = A11(B10-B00) -> C10 += M3, C11 += M3
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[1, 0, 3] = 1; W2[1, 1, 3] = 1
    
    # r=4: M4 = (A00+A01)B11 -> C00 -= M4, C10 += M4
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5: M5 = (A10-A00)(B00+B01) -> C01 -= M5
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[0, 1, 5] = -1
    
    # r=6: M6 = (A01-A11)(B10+B11) -> C10 -= M6
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[1, 0, 6] = -1
    
    return U2, V2, W2

def construct():
    # Deadline guard
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # The 4x4x4 tensor is the tensor product (Kronecker product) of two 2x2x2 tensors.
    # Rank = 7 * 7 = 49.
    # We need to map the flat indices of the rank-49 factors to the 4x4 sub-indices.
    # For U4 (shape 4,4,49): U4[i, j, r] where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
    # r = 7*s + t. U4[i,j,r] = U2[i_hi, j_hi, s] * U2[i_lo, j_lo, t].
    
    # We will construct the 4x4 factors efficiently using broadcasting and reshaping.
    # Instead of nested loops, we can use tensor operations.
    
    # Let's construct U4, V4, W4 with shapes (4, 4, 49)
    
    # Indices for high and low bits
    # i_hi in {0,1}, i_lo in {0,1}
    # We can view the 4x4 matrix as a 2x2 block matrix where each block is 2x2.
    # Actually, standard Kronecker product logic:
    # A = A_hi ⊗ A_lo (conceptually).
    # But the indices are mixed. i = 2*i_hi + i_lo.
    
    # Let's use numpy tensordot or einsum-like logic.
    # U4[i,j,r] = U2[i//2, j//2, r//7] * U2[i%2, j%2, r%7]
    
    # Create index arrays
    i_idx = np.arange(4)[:, np.newaxis]
    j_idx = np.arange(4)[np.newaxis, :]
    r_idx = np.arange(49)
    
    i_hi = (i_idx // 2)
    i_lo = (i_idx % 2)
    j_hi = (j_idx // 2)
    j_lo = (j_idx % 2)
    r_hi = (r_idx // 7)
    r_lo = (r_idx % 7)
    
    # Broadcast to shapes (4, 1, 1), (1, 4, 1), (1, 1, 49)
    # U2 is (2, 2, 7)
    
    # Get U2 components
    # U2_hi[i_hi, j_hi, r_hi] -> shape (4, 4, 49) after broadcasting?
    # U2[i_hi, j_hi, r_hi] works if we expand dims correctly.
    
    # Let's extract slices
    # U2_hi_vals[i, j, r] = U2[i_hi[i], j_hi[j], r_hi[r]]
    # This is effectively a Kronecker product of the matrices.
    
    # Simpler approach: Use the known structure.
    # U4 is (4,4,49). 
    # We can build it by iterating over the rank factor s (0..6) and filling the 7x7 blocks?
    # No, r = 7*s + t.
    
    # Let's just use the efficient vectorized way to generate the 4x4 factors.
    
    # U4[i, j, r]
    # i -> i_hi, i_lo
    # j -> j_hi, j_lo
    # r -> s, t
    
    # U4 = U2[i_hi, j_hi, s] * U2[i_lo, j_lo, t]
    
    # Construct U4
    # Shape of U2 is (2,2,7).
    # We need to map 4x4x49 indices to 2x2x7 indices.
    
    # Let's reshape U2 to separate high/low? No, U2 is fixed 2x2.
    
    # Let's use numpy indexing.
    # Create indices for U2 lookups
    # idx_i_hi = [[0,1],[0,1]] flattened? No.
    
    # Let's do this:
    # U4 = np.einsum or manual broadcasting.
    
    # Flatten 4x4 to 16.
    # Let's stick to the loop but make it fast or just use the direct mapping which is robust.
    # Given the small size, even Python loops are fast. 
    # But to be "distinct" and efficient, I will use the vectorized approach.
    
    # Map global indices to local indices
    # i in 0..3 -> i_hi = i//2, i_lo = i%2
    
    # Create 4x4x49 array
    # We can view U2 as having 7 slices (one for each rank component).
    # Let's construct the 49 rank slices. Each slice is 4x4.
    # Slice r (of 49) corresponds to s = r//7, t = r%7.
    # The slice is U2[:,:,s] (2x2) tensor U2[:,:,t] (2x2).
    # The Kronecker product A ⊗ B has indices (2*a1 + a2, 2*b1 + b2).
    
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Kronecker product of U2[:,:,s] and U2[:,:,t]
            # np.kron(A, B) where A is 2x2, B is 2x2 -> 4x4
            # Indices of kron: [2*iA + iB, 2*jA + jB] = A[iA, jA] * B[iB, jB]
            # This matches i = 2*i_hi + i_lo if we map i_hi->iA, i_lo->iB.
            
            U4[:, :, r] = np.kron(U2[:, :, s], U2[:, :, t])
            V4[:, :, r] = np.kron(V2[:, :, s], V2[:, :, t])
            W4[:, :, r] = np.kron(W2[:, :, s], W2[:, :, t])
            
            if time.time() - start_time > 140:
                break
        if time.time() - start_time > 140:
            break

    # Now construct the final 4x4x8 factors.
    # U: (16, 98) -> shape (n*m, rank)
    # V: (32, 98) -> shape (m*p, rank)
    # W: (32, 98) -> shape (p*n, rank)
    
    # The tensor is A(4x4) * B(4x8).
    # This can be seen as two 4x4 multiplications for the two 4x4 blocks of B.
    # B = [B_left (4x4), B_right (4x4)]
    # C = [C_left, C_right]
    # C_left = A * B_left
    # C_right = A * B_right
    
    # The rank is 49 + 49 = 98.
    # The U factors for the left and right blocks are identical (same A).
    # The V factors for the left block correspond to B_left, right block to B_right.
    # The W factors for the left block correspond to C_left, right to C_right.
    
    # Indexing conventions:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Total rank R = 98.
    # r in 0..48: Left block.
    # r in 49..97: Right block.
    
    U = np.zeros((16, 98), dtype=float)
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # Left Block (k in 0..3)
    # U part: Same for all k in 0..3? No, U depends on i,j.
    # U[i*4 + j, r]
    for r in range(49):
        # U is independent of the B block choice, so it's the same for left and right?
        # Yes, A is the same.
        # So U[:, 0:49] = U_flat
        # U[:, 49:98] = U_flat
        pass
        
    U_flat = U4.reshape(16, 49) # Row-major: i*4+j
    U[:, 0:49] = U_flat
    U[:, 49:98] = U_flat
    
    # V part:
    # V[j*8 + k, r]
    # For left block (r < 49), k ranges 0..3. j ranges 0..3.
    # V_left[j*8 + k, r] = V4[j, k, r]
    # Note: V4 is 4x4x49.
    # For right block (r >= 49), let r' = r - 49. k' = 4..7. k = 4 + k'.
    # V_right[j*8 + k, r] = V4[j, k-4, r']
    
    # Let's fill V
    for j in range(4):
        # Left half (k=0..3)
        V[j*8 + 0:4, 0:49] = V4[j, :, :] # Wait, V4 is (j,k,r). 
        # V4[j, k, r] needs to be placed at V[j*8+k, r].
        # So for a fixed j, V[j*8 + 0:4, 0:49] is V4[j, 0:4, :].
        
        # Right half (k=4..7)
        V[j*8 + 4:8, 49:98] = V4[j, :, :] # Same values, different r block
        
    # W part:
    # W[k*n + i, r]
    # For left block (r < 49), k ranges 0..3.
    # W_left[k*4 + i, r] = W4[k, i, r]
    # For right block (r >= 49), k ranges 4..7.
    # W_right[k*4 + i, r] = W4[k-4, i, r']
    
    for k in range(4):
        # Left half (k=0..3)
        W[k*4 + 0:4, 0:49] = W4[k, :, :]
        # Right half (k=4..3) -> k' in 0..3 maps to index 4+k'
        W[(k+4)*4 + 0:4, 49:98] = W4[k, :, :]
        
    # Optimization: Vectorized filling
    # U is already filled.
    # V:
    # V is 32x98.
    # Left: V[0:32, 0:49]. Indexing j*8+k.
    # V4 is 4x4x49. Reshape to (16, 49) -> (j*4+k).
    # But V index is j*8+k.
    # We need to scatter.
    
    # Let's just use the loops for V and W to be safe and clear, as 32*49 is small.
    # But the loops above are slightly inefficient. Let's do it properly.
    
    # Clear V and W to be sure
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # Fill V
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j, k, :]
            V[j*8 + (4+k), 49:98] = V4[j, k, :]
            
    # Fill W
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k, i, :]
            W[(4+k)*4 + i, 49:98] = W4[k, i, :]
    
    # Time check
    if time.time() - start_time > 140:
         pass # Still return the current result which is likely correct
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
