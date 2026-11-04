"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 115.0  # 150s wall-clock limit, keep 35s safety margin


def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)


# EVOLVE-BLOCK-START
def construct():
    """
    Build the best known exact tensor decomposition for M_{4,4,9}.
    
    Strategy: Partition the 9 columns into three blocks: 4 + 4 + 1.
    - Two M_{4,4,4} blocks each use Strassen's rank-49 decomposition.
    - One M_{4,4,1} block uses the trivial rank-16 decomposition.
    Total rank = 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9

    # --- Strassen decomposition for M_{2,2,2} (rank 7) ---
    # Indices: U index = i*2+j (i in {0,1}, j in {0,1}), V index = j*2+k, W index = k*2+i
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We need to embed Strassen into M_{4,4,4} by applying it to a 2x2 block partition.
    # Partition indices {0,1,2,3} into {0,1} and {2,3}.
    # M_{4,4,4} = sum over all i,j,k in {0,1,2,3} of A_{ij} B_{jk} C_{ki}
    # = sum over block-pairs of A_{Ij'} B_{j'K} C_{K'I}
    # where I, K in {{0,1},{2,3}} and j' in {0,1} within each block.
    
    # Actually, the standard way: M_{4,4,4} can be decomposed as:
    # M_{4,4,4} = M_{2,2,2}(top-left) + M_{2,2,2}(top-right) + M_{2,2,2}(bottom-left) + M_{2,2,2}(bottom-right)
    # Each M_{2,2,2} has rank 7, but they are not independent because the A, B, C indices overlap.
    
    # Better approach: Use the known rank-49 decomposition of M_{4,4,4}.
    # This can be constructed by taking the Kronecker product of two M_{2,2,2} tensors,
    # but that gives rank 49 = 7*7.
    
    # Actually, M_{4,4,4} = M_{2,2,2} \otimes M_{2,2,2} (Kronecker product), so rank = 7*7 = 49.
    # The Kronecker product decomposition is exact.
    
    # Let's build M_{4,4,4} via Kronecker product of two M_{2,2,2} Strassen decompositions.
    
    # First, define the Strassen M_{2,2,2} rank-1 terms:
    # For each of the 7 Strassen products, we have (u_r, v_r, w_r) where:
    # u_r is a vector of length 4 (indices i*2+j), v_r length 4 (indices j*2+k), w_r length 4 (indices k*2+i)
    
    # Let me define the 7 terms explicitly.
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # M1 = (A00+A11)(B00+B11): u has 1 at A00 and A11, v has 1 at B00 and B11, w has 1 at C00 and C11
    # Actually no. The Strassen formula is:
    # C00 = M1 + M4 - M5 + M7 where M1 = (A00+A11)(B00+B11)
    # This means M1 contributes to both C00 and C11 with coefficient +1.
    # So w for M1 has 1 at C00 and C11.
    
    # Let me be more careful. Each Strassen product M_r is a scalar multiplication.
    # The contribution to C is:
    # M1: C00 += M1, C11 += M1
    # M2: C10 += M2, C11 -= M2
    # M3: C01 += M3, C10 += M3
    # M4: C00 += M4, C11 += M4
    # M5: C00 -= M5, C01 += M5
    # M6: C11 += M6
    # M7: C00 += M7
    
    # For the rank-1 tensor decomposition, each term r has:
    # u_r: vector over A indices (length 4)
    # v_r: vector over B indices (length 4)  
    # w_r: vector over C indices (length 4)
    # such that sum_r u_r[i] * v_r[j] * w_r[k] = T[i,j,k]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # u1: A00=1, A11=1 (indices 0 and 3)
    # v1: B00=1, B11=1 (indices 0 and 3)
    # w1: C00=1, C11=1 (indices 0 and 3)
    
    # Term 2: M2 = (A10+A11)(B00)
    # u2: A10=1, A11=1 (indices 2 and 3)
    # v2: B00=1 (index 0)
    # w2: C10=1, C11=-1 (indices 2 and 3)
    
    # Term 3: M3 = (A00)(B10-B11)
    # u3: A00=1 (index 0)
    # v3: B10=1, B11=-1 (indices 2 and 3)
    # w3: C01=1, C10=1 (indices 1 and 2)
    
    # Term 4: M4 = (A11)(B01-B00)
    # u4: A11=1 (index 3)
    # v4: B01=1, B00=-1 (indices 1 and 0)
    # w4: C00=1, C11=1 (indices 0 and 3)
    
    # Term 5: M5 = (A00+A01)(B11)
    # u5: A00=1, A01=1 (indices 0 and 1)
    # v5: B11=1 (index 3)
    # w5: C00=-1, C01=1 (indices 0 and 1)
    
    # Term 6: M6 = (A10-A00)(B00+B10)
    # u6: A10=1, A00=-1 (indices 2 and 0)
    # v6: B00=1, B10=1 (indices 0 and 2)
    # w6: C11=1 (index 3)
    
    # Term 7: M7 = (A01-A11)(B01+B11)
    # u7: A01=1, A11=-1 (indices 1 and 3)
    # v7: B01=1, B11=1 (indices 1 and 3)
    # w7: C00=1 (index 0)
    
    strassen_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    strassen_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 1, 0],  # M6
        [0, 1, 0, 1],  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 1, 0],  # M3
        [1, 0, 0, 1],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]
    
    # Now build M_{4,4,4} via Kronecker product.
    # M_{4,4,4}(i,j,k) where i,j,k in {0,1,2,3}
    # Write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # Then M_{4,4,4}(i,j,k) = M_{2,2,2}(i1,j1,k1) * M_{2,2,2}(i0,j0,k0)
    # Wait, that's not quite right. The Kronecker product of two matrix multiplication tensors:
    # (M_{2,2,2} \otimes M_{2,2,2})_{(i1,i0),(j1,j0),(k1,k0)} = M_{2,2,2}(i1,j1,k1) * M_{2,2,2}(i0,j0,k0)
    # But this gives the tensor for multiplying (A1 \otimes A0) by (B1 \otimes B0),
    # which is not the same as M_{4,4,4}.
    
    # Actually, M_{n*m, n*m, n*m} = M_{n,n,n} \otimes M_{m,m,m} is NOT correct in general.
    # The correct identity is: M_{n,m,p} \otimes M_{n',m',p'} = M_{n*n', m*m', p*p'}
    # So M_{2,2,2} \otimes M_{2,2,2} = M_{4,4,4}. Yes!
    # The rank of the tensor product is rank1 * rank2 = 7 * 7 = 49.
    
    # To construct the rank-49 decomposition:
    # For each pair (r1, r2) where r1, r2 in {0,...,6}:
    # u_{r1,r2}[i] where i = 2*i1 + i0: u_{r1,r2}[2*i1+i0] = strassen_u[r1][i1] * strassen_u[r2][i0]
    # v_{r1,r2}[j] where j = 2*j1 + j0: v_{r1,r2}[2*j1+j0] = strassen_v[r1][j1] * strassen_v[r2][j0]
    # w_{r1,r2}[k] where k = 2*k1 + k0: w_{r1,r2}[2*k1+k0] = strassen_w[r1][k1] * strassen_w[r2][k0]
    
    def build_m444():
        """Build rank-49 decomposition of M_{4,4,4}."""
        U444 = []
        V444 = []
        W444 = []
        for r1 in range(7):
            for r2 in range(7):
                u = [0] * 16
                v = [0] * 16
                w = [0] * 16
                for i1 in range(2):
                    for i0 in range(2):
                        idx = i1 * 4 + i0  # i = i1*2 + i0, but A has n*m = 4*4 = 16 entries
                        # Wait, A is 4x4, so A index is i*4 + j? No.
                        # The interface says U has shape (n*m, rank) = (16, rank)
                        # U[i*m + j, c] where i in 0..n-1, j in 0..m-1
                        # For M_{4,4,4}, n=m=p=4, so U index = i*4 + j
                        # In the Kronecker product, the "outer" index is (i1,j1) and "inner" is (i0,j0)
                        # So A_{(i1,j1),(i0,j0)} corresponds to A_{2*i1+i0, 2*j1+j0}
                        # U index = (2*i1+i0)*4 + (2*j1+j0)
                        pass
                
                # Let me redo this. For M_{4,4,4}:
                # A is 4x4, indexed by (i,j) with i,j in 0..3
                # B is 4x4, indexed by (j,k)
                # C is 4x4, indexed by (k,i)
                # U index = i*4 + j (16 entries)
                # V index = j*4 + k (16 entries)
                # W index = k*4 + i (16 entries)
                
                # Kronecker: i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
                # U[2*i1+i0, 2*j1+j0] contribution from term (r1,r2):
                # strassen_u[r1][i1] * strassen_u[r2][i0] * ... wait no.
                
                # The Kronecker product tensor:
                # T444[i,j,k] = T222[i1,j1,k1] * T222[i0,j0,k0]
                # where i=2*i1+i0, etc.
                # The rank-1 decomposition of T222: T222 = sum_r u_r \otimes v_r \otimes w_r
                # So T444 = sum_{r1,r2} (u_{r1} \otimes u_{r2}) \otimes (v_{r1} \otimes v_{r2}) \otimes (w_{r1} \otimes w_{r2})
                # where (u_{r1} \otimes u_{r2})[2*i1+i0] = u_{r1}[i1] * u_{r2}[i0]
                
                for i1 in range(2):
                    for i0 in range(2):
                        for j1 in range(2):
                            for j0 in range(2):
                                u_idx = (2*i1 + i0) * 4 + (2*j1 + j0)
                                u[u_idx] = strassen_u[r1][i1] * strassen_u[r2][i0] * strassen_v[r1][j1] * strassen_v[r2][j0]
                                # Wait, that's wrong. u should only depend on A indices.
                                # u_{r1,r2} is a vector of length 16 (for A_{i,j}).
                                # (u_{r1} \otimes u_{r2}) is not right because u and v are different.
                                pass
                
                # I'm confusing myself. Let me restart the Kronecker construction.
                # T222 has three legs: A-leg (length 4), B-leg (length 4), C-leg (length 4)
                # T444 = T222 \otimes T222 has:
                # A-leg: length 16, indexed by (a1,a0) where a1,a0 in 0..3
                # B-leg: length 16, indexed by (b1,b0)
                # C-leg: length 16, indexed by (c1,c0)
                # T444[a1,b1,c1,a0,b0,c0] = T222[a1,b1,c1] * T222[a0,b0,c0]
                
                # Rank-1 term for pair (r1,r2):
                # A-vector: length 16, entry (a1,a0) = u222[r1][a1] * u222[r2][a0]
                # B-vector: length 16, entry (b1,b0) = v222[r1][b1] * v222[r2][b0]
                # C-vector: length 16, entry (c1,c0) = w222[r1][c1] * w222[r2][c0]
                
                # But we need to map these to the interface convention:
                # For M_{4,4,4}: U index = i*4+j, V index = j*4+k, W index = k*4+i
                # The Kronecker A-leg index (a1,a0) corresponds to (i,j) where:
                # a1 = i1 (row block), a0 = i0 (row within block)? No.
                # Actually, the A-leg of T_{n,m,p} is indexed by (i,j) pairs.
                # For T_{4,4,4}, A-leg index = (i,j) with i,j in 0..3.
                # For T_{2,2,2} \otimes T_{2,2,2}, A-leg index = ((i1,j1),(i0,j0))
                # This should map to i = 2*i1 + i0, j = 2*j1 + j0 for M_{4,4,4}.
                # So A-leg index (a1,a0) where a1=(i1,j1), a0=(i0,j0) maps to (i,j) = (2*i1+i0, 2*j1+j0)
                # And U index = i*4 + j = (2*i1+i0)*4 + (2*j1+j0)
                
                u_vec = [0] * 16
                v_vec = [0] * 16
                w_vec = [0] * 16
                for i1 in range(2):
                    for j1 in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                # A index
                                i = 2*i1 + i0
                                j = 2*j1 + j0
                                u_idx = i * 4 + j
                                u_val = strassen_u[r1][i1] * strassen_u[r2][i0]
                                # But wait, u222[r1] is indexed by the A-leg of T222, which is (i,j) pairs for 2x2.
                                # For T222, A-leg index = i*2 + j (length 4).
                                # So strassen_u[r1][i1*2 + j1] is the value.
                                u_val = strassen_u[r1][i1 * 2 + j1] * strassen_u[r2][i0 * 2 + j0]
                                u_vec[u_idx] = u_val
                                
                                # B index: V index = j*4 + k
                                # B-leg of T444: ((j1,k1),(j0,k0)) maps to j=2*j1+j0, k=2*k1+k0
                                # V index = j*4 + k
                                # We'll fill this in the k loop
                                
                for j1 in range(2):
                    for k1 in range(2):
                        for j0 in range(2):
                            for k0 in range(2):
                                j = 2*j1 + j0
                                k = 2*k1 + k0
                                v_idx = j * 4 + k
                                v_val = strassen_v[r1][j1 * 2 + k1] * strassen_v[r2][j0 * 2 + k0]
                                v_vec[v_idx] = v_val
                                
                for k1 in range(2):
                    for i1 in range(2):
                        for k0 in range(2):
                            for i0 in range(2):
                                k = 2*k1 + k0
                                i = 2*i1 + i0
                                w_idx = k * 4 + i
                                w_val = strassen_w[r1][k1 * 2 + i1] * strassen_w[r2][k0 * 2 + i0]
                                w_vec[w_idx] = w_val
                                
                U444.append(u_vec)
                V444.append(v_vec)
                W444.append(w_vec)
        
        return U444, V444, W444
    
    U444, V444, W444 = build_m444()
    
    # Now build M_{4,4,1} (rank 16):
    # A is 4x4, B is 4x1, C is 1x4
    # U shape: (4*4, 16) = (16, 16)
    # V shape: (4*1, 16) = (4, 16)
    # W shape: (1*4, 16) = (4, 16)
    # U[i*4+j, c], V[j*1+k, c] = V[j, c], W[k*1+i, c] = W[i, c]
    # c = i*4 + j
    # U[i*4+j, i*4+j] = 1
    # V[j, i*4+j] = 1 for all i? No.
    # T[i,j,k] = 1 if k=i (for M_{4,4,1}, p=1 so k=0 always, and C is 1x4)
    # Wait, M_{4,4,1}: A is 4x4, B is 4x1, C is 1x4.
    # C[0,i] = sum_j A[i,j] * B[j,0]
    # So T[i,j,0] = delta(i, i) = 1 for all i,j.
    # Actually T[i,j,k] where k is the B index (only 0), and C index is (0, i).
    # The tensor: T[A_idx, B_idx, C_idx] where A_idx=(i,j), B_idx=(j,0), C_idx=(0,i)
    # T[i,j,0] = 1 for all i,j (since C[0,i] = sum_j A[i,j]*B[j,0])
    # So T = sum_{i,j} e_{(i,j)} \otimes e_{(j,0)} \otimes e_{(0,i)}
    # This has rank 16 (one term per (i,j) pair).
    
    U441 = [[0] * 16 for _ in range(16)]
    V441 = [[0] * 16 for _ in range(4)]
    W441 = [[0] * 16 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U441[i * 4 + j][c] = 1
            V441[j][c] = 1
            W441[i][c] = 1
    
    # Now assemble M_{4,4,9}:
    # A is 4x4, B is 4x9, C is 9x4
    # U shape: (4*4, rank) = (16, rank)
    # V shape: (4*9, rank) = (36, rank)
    # W shape: (9*4, rank) = (36, rank)
    # 
    # B is partitioned as B[:, 0:4], B[:, 4:8], B[:, 8:9]
    # C is partitioned as C[0:4, :], C[4:8, :], C[8:9, :]
    
    # For block 1 (columns 0-3 of B, rows 0-3 of C): M_{4,4,4}
    # U444 has shape (16, 49). Map to U.
    # V444 has shape (16, 49). Map to V[:, 0:4].
    # W444 has shape (16, 49). Map to W[0:4, :].
    
    # For block 2 (columns 4-7 of B, rows 4-7 of C): M_{4,4,4}
    # U444 same U.
    # V444 maps to V[:, 4:8].
    # W444 maps to W[4:8, :].
    
    # For block 3 (column 8 of B, row 8 of C): M_{4,4,1}
    # U441 maps to U.
    # V441 maps to V[:, 8].
    # W441 maps to W[8, :].
    
    total_rank = 49 + 49 + 16  # 114
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Block 1
    for c in range(49):
        for u_idx in range(16):
            U[u_idx][c] = U444[u_idx][c]
        for v_idx in range(16):
            V[v_idx][c] = V444[v_idx][c]
        for w_idx in range(16):
            W[w_idx][c] = W444[w_idx][c]
    
    # Block 2
    offset = 49
    for c in range(49):
        for u_idx in range(16):
            U[u_idx][offset + c] = U444[u_idx][c]
        for v_idx in range(16):
            V[v_idx + 16][offset + c] = V444[v_idx][c]  # V index for B[:, 4:8] is j*9 + (4+k)
            # Wait, V index = j*p + k = j*9 + k. For block 2, k in 4..7.
            # V444 index is j*4 + k_local where k_local in 0..3.
            # So V[j*9 + 4 + k_local][offset + c] = V444[j*4 + k_local][c]
        for w_idx in range(16):
            W[w_idx + 16][offset + c] = W444[w_idx][c]  # W index = k*n + i = (4+k_local)*4 + i
    
    # Block 3
    offset2 = 98
    for c in range(16):
        for u_idx in range(16):
            U[u_idx][offset2 + c] = U441[u_idx][c]
        for v_idx in range(4):
            V[v_idx * 9 + 8][offset2 + c] = V441[v_idx][c]  # V[j*9 + 8]
        for w_idx in range(4):
            W[8 * 4 + w_idx][offset2 + c] = W441[w_idx][c]  # W[8*4 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
