"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block-structured Strassen: 3x3 blocks of 2x2, rank 7 per 2x2 block
    # Total rank = 27 * 7 = 189
    import numpy as np
    
    n = 6
    rank = 189
    
    # 2x2 Strassen factors
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U2: (2*2, 7) = (4, 7) - maps A sub-block to Strassen inputs
    # A = [[a, b], [c, d]] indexed as A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    U2 = np.zeros((4, 7))
    # M1: a+d -> U2[0,0]=1, U2[3,0]=1
    U2[0, 0] = 1; U2[3, 0] = 1
    # M2: c+d -> U2[2,1]=1, U2[3,1]=1
    U2[2, 1] = 1; U2[3, 1] = 1
    # M3: a -> U2[0,2]=1
    U2[0, 2] = 1
    # M4: d -> U2[3,3]=1
    U2[3, 3] = 1
    # M5: a+b -> U2[0,4]=1, U2[1,4]=1
    U2[0, 4] = 1; U2[1, 4] = 1
    # M6: c-a -> U2[2,5]=1, U2[0,5]=-1
    U2[2, 5] = 1; U2[0, 5] = -1
    # M7: b-d -> U2[1,6]=1, U2[3,6]=-1
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V2: (2*2, 7) = (4, 7) - maps B sub-block to Strassen inputs
    # B = [[e, f], [g, h]] indexed as B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    V2 = np.zeros((4, 7))
    # M1: e+h -> V2[0,0]=1, V2[3,0]=1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: e -> V2[0,1]=1
    V2[0, 1] = 1
    # M3: f-h -> V2[1,2]=1, V2[3,2]=-1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4: g-f -> V2[2,3]=1, V2[1,3]=-1
    V2[2, 3] = 1; V2[1, 3] = -1
    # M5: h -> V2[3,4]=1
    V2[3, 4] = 1
    # M6: e+f -> V2[0,5]=1, V2[1,5]=1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: g+h -> V2[2,6]=1, V2[3,6]=1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2: (2*2, 7) = (4, 7) - maps Strassen outputs to C sub-block
    # C = [[C11, C12], [C21, C22]] indexed as C[0,0], C[0,1], C[1,0], C[1,1]
    # W2[i*2+j, k] means contribution to C[i,j] from M_k
    W2 = np.zeros((4, 7))
    # C11 = M1 - M5 + M7
    W2[0, 0] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C12 = M3 + M5
    W2[1, 2] = 1; W2[1, 4] = 1
    # C21 = M2 + M4
    W2[2, 1] = 1; W2[2, 3] = 1
    # C22 = M1 - M2 + M3 + M6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Now build the full 6x6 decomposition using Kronecker structure
    # A is 6x6, partitioned into 3x3 blocks of 2x2
    # A[i*2+i2, j*2+j2] for block (i,j) element (i2,j2)
    
    U = np.zeros((n * n, rank))  # (36, 189)
    V = np.zeros((n * n, rank))  # (36, 189)
    W = np.zeros((n * n, rank))  # (36, 189), but W uses trace convention: W[k*n+i, r]
    
    # For each pair of 2x2 blocks (bi, bj) in the 3x3 structure:
    # bi, bj in range(3)
    # Each block pair uses 7 Strassen multiplications
    # Total: 9 block positions * 7 = 63... wait, that's not right.
    
    # Actually, in the standard block Strassen for 3x3 blocks:
    # We have 9 output blocks, each computed via Strassen on 2x2 sub-blocks
    # But the input blocks are shared.
    
    # Let me think differently. The tensor for 6x6 mult has entries T[i,j,k] where
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Partition into 2x2 blocks: let i = 2*I + i2, j = 2*J + j2, k = 2*K + k2
    # where I,J,K in {0,1,2} and i2,j2,k2 in {0,1}
    #
    # C[2*I+i2, 2*K+k2] = sum_{J=0}^{2} sum_{j2=0}^{1} A[2*I+i2, 2*J+j2] * B[2*J+j2, 2*K+k2]
    #
    # For each fixed I,K, this is a 2x2 = sum over J of 2x2 block multiplications
    # But Strassen applies to the entire 2x2 matrix multiplication when A and B are 2x2.
    #
    # Actually, the standard approach: treat the 6x6 as 3x3 blocks of 2x2.
    # For each output block position (I,K), we compute:
    # C[I,K] = sum_J A[I,J] * B[J,K]  (all 2x2 matrices)
    #
    # To use Strassen on each 2x2 block multiplication, we'd need 9*7=63 rank-1 terms,
    # but that's not how it works. Strassen replaces one 2x2 mult (rank 8) with 7 mults.
    #
    # The correct decomposition: the 6x6 multiplication tensor can be written as a sum of
    # rank-1 tensors. With block Strassen, we get rank 189.
    #
    # The structure: for each of the 9 (I,J) input block pairs and each Strassen factor,
    # we get contributions. Let me build it explicitly.
    
    # Reset
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    col = 0
    for I in range(3):  # row block of A / col block of C
        for J in range(3):  # col block of A / row block of B
            for K in range(3):  # col block of B / row block of C... 
                # This doesn't work directly. Let me use the Kronecker product approach.
                pass
    
    # Better: use the fact that if M_n has rank r, then M_{n^2} has rank at most r^2 * (n/n)^2... 
    # Actually, the Kronecker product of two tensor decompositions:
    # If T6 = sum_c U_c ⊗ V_c ⊗ W_c, then T36 = T6 ⊗ T6 has rank (rank T6)^2.
    # But we want T6, not T36.
    
    # Let me just directly construct the 189-rank decomposition.
    # The 6x6 multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Using 2x2 block Strassen:
    # Let i = 2*I + i2, j = 2*J + j2, k = 2*K + k2
    #
    # For each (I,K) output block, C[2*I+i2, 2*K+k2] = sum_J sum_j2 A[2*I+i2, 2*J+j2] B[2*J+j2, 2*K+k2]
    #
    # This is a sum of 3 terms (over J), each being a 2x2 block product.
    # If we apply Strassen to each 2x2 block product, we get 3*7 = 21 rank-1 terms per output block,
    # but the input blocks are shared across different output blocks.
    
    # The correct way: the bilinear form for each (i2, j2, k2, I, J, K) is:
    # A[2*I+i2, 2*J+j2] * B[2*J+j2, 2*K+k2] contributes to C[2*I+i2, 2*K+k2]
    #
    # So T[i,k,j] = 1 if i=2*I+i2, j=2*J+j2, k=2*K+k2 for the same I,J,K,i2,j2,k2
    #
    # This is just the schoolbook with rank 216. To reduce to 189, we need to find
    # linear dependencies using the Strassen identity within each 2x2 block structure.
    
    # The key: for each fixed (I,J,K), the 8 terms (i2,j2,k2) form a 2x2 matrix multiplication.
    # Strassen shows these 8 can be replaced by 7 rank-1 terms.
    
    # So for each of the 27 triples (I,J,K), we replace 8 schoolbook terms with 7 Strassen terms.
    # Total rank = 27 * 7 = 189.
    
    U = np.zeros((36, 189))
    V = np.zeros((36, 189))
    W = np.zeros((36, 189))
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # Strassen factor s for the 2x2 block at (I,J,K)
                    # U contribution: from A[2*I+*, 2*J+*]
                    for i2 in range(2):
                        for j2 in range(2):
                            U[(2*I + i2) * 6 + (2*J + j2), col] = U2[i2 * 2 + j2, s]
                    # V contribution: from B[2*J+*, 2*K+*]
                    for j2 in range(2):
                        for k2 in range(2):
                            V[(2*J + j2) * 6 + (2*K + k2), col] = V2[j2 * 2 + k2, s]
                    # W contribution: to C[2*I+*, 2*K+*]
                    # W uses trace convention: W[k*n+i, r] = W[k*6+i, r]
                    for i2 in range(2):
                        for k2 in range(2):
                            W[(2*K + k2) * 6 + (2*I + i2), col] = W2[i2 * 2 + k2, s]
                    col += 1
    
    # Convert to lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

