"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _strassen_2x2_2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen with rank 7.
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # The 7 Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B21)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express each M as a rank-1 tensor in the 2x2x2 space.
    # For each M, we define vectors u, v, w in R^4 such that
    # U[i,j] = u_{i,j}, V[j,k] = v_{j,k}, W[k,i] = w_{k,i}
    # and the sum over r gives the correct C.
    
    # Let's define the 7 rank-1 tensors explicitly.
    # Each tensor is defined by (u_vec, v_vec, w_vec) where each is length 4.
    # u_vec index: i*2+j
    # v_vec index: j*2+k
    # w_vec index: k*2+i
    
    tensors = []
    
    # M1 = (A11+A22)(B11+B22)
    # This product contributes to C11 and C22.
    # C11 gets +1, C22 gets +1.
    # u = A11+A22 = [1,0,0,1]
    # v = B11+B22 = [1,0,0,1]
    # w = C11+C22 = [1,0,0,1] (since C11 is at (i=0,k=0) -> w[0*2+0]=1, C22 at (i=1,k=1) -> w[1*2+1]=1)
    tensors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A21+A22)B11
    # Contributes to C21 (+1) and C22 (-1)
    # u = A21+A22 = [0,0,1,1]
    # v = B11 = [1,0,0,0]
    # w = C21 - C22 = [0,0,1,-1] (C21 at i=1,k=0 -> w[0*2+1]=1, C22 at i=1,k=1 -> w[1*2+1]=-1)
    tensors.append(([0,0,1,1], [1,0,0,0], [0,1,-1,0])) # Wait, w index is k*2+i.
    # C21: i=1, k=0 -> index 0*2+1 = 1
    # C22: i=1, k=1 -> index 1*2+1 = 3
    # So w = [0, 1, 0, -1]
    tensors[-1] = ([0,0,1,1], [1,0,0,0], [0,1,0,-1])
    
    # M3 = A11(B21-B22)
    # Contributes to C12 (+1) and C22 (+1)
    # u = A11 = [1,0,0,0]
    # v = B21-B22 = [0,1,-1,0]
    # w = C12 + C22 = [0,1,0,1]
    # C12: i=0, k=1 -> index 1*2+0 = 2
    # C22: i=1, k=1 -> index 1*2+1 = 3
    # So w = [0, 0, 1, 1]
    tensors.append(([1,0,0,0], [0,1,-1,0], [0,0,1,1]))
    
    # M4 = A22(B11-B21)
    # Contributes to C11 (+1) and C21 (+1)
    # u = A22 = [0,0,0,1]
    # v = B11-B21 = [1,0,-1,0]
    # w = C11 + C21 = [1,1,0,0]
    # C11: i=0, k=0 -> index 0*2+0 = 0
    # C21: i=1, k=0 -> index 0*2+1 = 1
    # So w = [1, 1, 0, 0]
    tensors.append(([0,0,0,1], [1,0,-1,0], [1,1,0,0]))
    
    # M5 = (A11+A12)B22
    # Contributes to C11 (-1) and C12 (+1)
    # u = A11+A12 = [1,1,0,0]
    # v = B22 = [0,0,0,1]
    # w = -C11 + C12 = [-1,0,1,0]
    # C11: i=0, k=0 -> index 0
    # C12: i=0, k=1 -> index 2
    # So w = [-1, 0, 1, 0]
    tensors.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6 = (A21-A11)(B11+B12)
    # Contributes to C22 (+1)
    # u = A21-A11 = [-1,0,1,0]
    # v = B11+B12 = [1,1,0,0]
    # w = C22 = [0,0,0,1]
    # C22: i=1, k=1 -> index 3
    # So w = [0, 0, 0, 1]
    tensors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A12-A22)(B21+B22)
    # Contributes to C11 (+1)
    # u = A12-A22 = [0,1,0,-1]
    # v = B21+B22 = [0,1,1,0]
    # w = C11 = [1,0,0,0]
    # C11: i=0, k=0 -> index 0
    # So w = [1, 0, 0, 0]
    tensors.append(([0,1,0,-1], [0,1,1,0], [1,0,0,0]))
    
    # Build U, V, W of size (4, 7)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for r in range(7):
        u, v, w = tensors[r]
        for idx in range(4):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
    
    return U, V, W

def _strassen_4x4_4x4():
    """
    Construct 4x4x4 Strassen decomposition with rank 49.
    This treats the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    """
    U_local, V_local, W_local = _strassen_2x2_2x2()
    
    # The 4x4 matrices are indexed by (I, i) and (J, j) where I,J in {0,1} are block indices
    # and i,j in {0,1} are local indices.
    # Global index for A: (I*2+i)*4 + (J*2+j) = (I*2+i)*4 + J*2 + j
    # Global index for B: (J*2+j)*4 + (K*2+k) = (J*2+j)*4 + K*2 + k
    # Global index for C: (K*2+k)*4 + (I*2+i) = (K*2+k)*4 + I*2 + i
    
    # We have 7 block-level products. For each block product r_block in 0..6,
    # we have a 2x2x2 Strassen decomposition.
    # The total rank is 7 * 7 = 49.
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for rb in range(7):
        # For each block product rb, we use the local Strassen factors U_local, V_local, W_local
        # The global rank index is rb * 7 + rl, where rl is the local rank index.
        for rl in range(7):
            r = rb * 7 + rl
            
            # For each local block index (I,J,K) in the 2x2 block grid:
            # The local Strassen factors are U_local[i*2+j, rl], etc.
            # But we need to map this to the global 4x4 matrices.
            
            # Actually, the 4x4x4 Strassen is a block decomposition where each block is 2x2.
            # The block-level Strassen gives 7 products of 2x2 blocks.
            # Each 2x2 block product is itself decomposed using the 2x2x2 Strassen (rank 7).
            
            # So for each rb in 0..6, we have a 2x2x2 tensor (from block Strassen)
            # which is decomposed into 7 rank-1 tensors (from local Strassen).
            
            # The block Strassen factors for rb are:
            # u_block[I*2+J], v_block[J*2+K], w_block[K*2+I]
            # These are the same as U_local[:, rb], V_local[:, rb], W_local[:, rb]
            # Wait, no. The block Strassen is a 2x2x2 tensor with rank 7.
            # Its factors are U_block, V_block, W_block of size (4, 7).
            # U_block[I*2+J, rb] = U_local[I*2+J, rb]
            
            # Now, for each rb, we decompose the block product using local Strassen.
            # The local Strassen has rank 7.
            # So for each rl in 0..6, we have local factors U_local[:, rl], V_local[:, rl], W_local[:, rl].
            
            # The combined factor for (rb, rl) is:
            # U_global[(I*2+i)*4 + (J*2+j), r] = U_block[I*2+J, rb] * U_local[i*2+j, rl]
            # V_global[(J*2+j)*4 + (K*2+k), r] = V_block[J*2+K, rb] * V_local[j*2+k, rl]
            # W_global[(K*2+k)*4 + (I*2+i), r] = W_block[K*2+I, rb] * W_local[k*2+i, rl]
            
            for I in range(2):
                for J in range(2):
                    u_block_val = U_local[I*2+J, rb]
                    if u_block_val == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            u_local_val = U_local[i*2+j, rl]
                            u_val = u_block_val * u_local_val
                            if u_val != 0:
                                U[(I*2+i)*4 + (J*2+j)][r] = u_val
                            
            for J in range(2):
                for K in range(2):
                    v_block_val = V_local[J*2+K, rb]
                    if v_block_val == 0:
                        continue
                    for j in range(2):
                        for k in range(2):
                            v_local_val = V_local[j*2+k, rl]
                            v_val = v_block_val * v_local_val
                            if v_val != 0:
                                V[(J*2+j)*4 + (K*2+k)][r] = v_val
                            
            for K in range(2):
                for I in range(2):
                    w_block_val = W_local[K*2+I, rb]
                    if w_block_val == 0:
                        continue
                    for k in range(2):
                        for i in range(2):
                            w_local_val = W_local[k*2+i, rl]
                            w_val = w_block_val * w_local_val
                            if w_val != 0:
                                W[(K*2+k)*4 + (I*2+i)][r] = w_val
                                
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Uses Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x5 part (rank 80).
    Total rank: 129.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Split the 4x9 matrix into a 4x4 part and a 4x5 part.
    p1 = 4
    p2 = 5
    
    # 4x4x4 Strassen decomposition
    U1, V1, W1 = _strassen_4x4_4x4()
    rank1 = 49
    
    # 4x4x5 schoolbook decomposition
    U2, V2, W2 = _schoolbook(4, 4, 5)
    rank2 = 80
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy U1, V1, W1 into the first rank1 columns
    for i in range(n * m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    
    # Copy V1, W1 with appropriate index mapping
    # V1 has size (4*4, 49) = (16, 49)
    # Global V has size (4*9, 129) = (36, 129)
    # V1[j*4+k, r] corresponds to V[j*9+k, r] for k in 0..3
    for j in range(4):
        for k in range(4):
            for r in range(rank1):
                V[j*9+k][r] = V1[j*4+k][r]
    
    # W1 has size (4*4, 49) = (16, 49)
    # Global W has size (9*4, 129) = (36, 129)
    # W1[k*4+i, r] corresponds to W[k*4+i, r] for k in 0..3
    for k in range(4):
        for i in range(4):
            for r in range(rank1):
                W[k*4+i][r] = W1[k*4+i][r]
    
    # Copy U2, V2, W2 into the next rank2 columns
    for i in range(n * m):
        for r in range(rank2):
            U[i][rank1 + r] = U2[i][r]
    
    # V2 has size (4*5, 80) = (20, 80)
    # Global V has size (4*9, 129) = (36, 129)
    # V2[j*5+k, r] corresponds to V[j*9+(k+4), r] for k in 0..4
    for j in range(4):
        for k in range(5):
            for r in range(rank2):
                V[j*9+(k+4)][rank1 + r] = V2[j*5+k][r]
    
    # W2 has size (5*4, 80) = (20, 80)
    # Global W has size (9*4, 129) = (36, 129)
    # W2[k*4+i, r] corresponds to W[(k+4)*4+i, r] for k in 0..4
    for k in range(5):
        for i in range(4):
            for r in range(rank2):
                W[(k+4)*4+i][rank1 + r] = W2[k*4+i][r]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
