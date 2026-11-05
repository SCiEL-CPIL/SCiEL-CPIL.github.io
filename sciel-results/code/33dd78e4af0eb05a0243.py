import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    The tensor T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p) represents
    the matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k].
    
    We use a known efficient decomposition approach. For 2x4x8, we can exploit the
    structure to get a rank significantly below the schoolbook 64.
    
    The approach: Use the fact that for small matrices, we can find decompositions
    based on known results. For 2xm x mxp, the rank is known to be less than 2mp.
    
    For 2x4x8, we'll construct a decomposition of rank 56 (using Strassen-like
    sharing on 2x2 blocks as noted in the knowledge) or try to find an even better one.
    
    Actually, let me think more carefully. The matrix multiplication tensor M_{n,m,p}
    has rank at least max(n*m, m*p, p*n) = max(8, 32, 16) = 32, and at most n*m*p = 64.
    
    Known results: For M_{2,2,2}, rank is 7 (Strassen). For M_{2,4,8}, we can use
    block structure.
    
    Let's use a direct construction. For 2x4x8, we can view this as two independent
    1x4x8 multiplications combined, but that gives rank 2*4*8=64.
    
    Better: Use the fact that we're multiplying 2x4 by 4x8. We can decompose the
    4x8 matrix into 2x2 blocks, giving a 2x2 grid of 2x4 blocks times 4x8 blocks...
    
    Let me use a simpler but effective approach: exploit that for 2xm x mxp, we can
    get rank close to (3/4)*m*p for large m,p using Strassen, but for small cases
    we need explicit constructions.
    
    For 2x4x8, a known upper bound comes from viewing it as 2 copies of 1x4x8 with
    some sharing. The rank of M_{1,m,p} is m*p (no savings possible for rank-1).
    So we need to share between the two rows.
    
    Let me try a different approach: use the fact that C[0,k] = sum_j A[0,j]*B[j,k]
    and C[1,k] = sum_j A[1,j]*B[j,k]. We can compute all 8 outputs for each row,
    but we need to do both rows.
    
    One approach: compute S_j = A[0,j]*B[j,:] + A[1,j]*B[j,:] and T_j = A[0,j]*B[j,:] - A[1,j]*B[j,:]
    for j=0..3. Then C[0,:] = (S_0 + S_1 + S_2 + S_3 + T_0 + T_1 + T_2 + T_3)/2 ... 
    no, that doesn't work because S_j and T_j are vectors.
    
    Let me think differently. For each k, C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k].
    We have 16 output values (2 rows x 8 columns) and 32 input values (2x4 + 4x8).
    
    A rank-r decomposition means we can compute the product using r scalar multiplications.
    
    Let's try to construct a rank-48 decomposition or better. 
    
    Actually, let me use a known result: for M_{2,m,p}, the rank is at most 
    ceil(3/4 * m * p) + something for large m,p, but for small m,p we need explicit constructions.
    
    For M_{2,4,8}, let me try to use a block Strassen approach. Split the 4x8 matrix into
    two 4x4 blocks: B = [B1 | B2] where B1, B2 are 4x4. Then:
    C[:, 0:4] = A * B1 (2x4 times 4x4 = 2x4)
    C[:, 4:8] = A * B2 (2x4 times 4x4 = 2x4)
    
    Each 2x4x4 multiplication: rank at most 2*4*4=32, but can we do better?
    M_{2,4,4}: rank is at least max(8,16,8)=16, at most 32.
    
    For M_{2,4,4}, we can use: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k] for k=0..3.
    That's 8 outputs, 16 inputs. Rank at least 8.
    
    Using Strassen on 2x2 blocks: split A into [A11 A12; A21 A22] (2x2 blocks of 1x2 each? No, A is 2x4).
    Let A = [a00 a01 a02 a03; a10 a11 a12 a13] and B is 4x4.
    
    This is getting complex. Let me just use a known efficient decomposition.
    
    For the specific case of 2x4x8, I'll use a rank-56 decomposition based on 
    sharing multiplications within 2x2 block structures (as noted in the knowledge).
    
    The idea: group the 4 intermediate indices j into 2 pairs, and for each pair, 
    use Strassen-like sharing to reduce from 2*2*8=32 to 7*... hmm.
    
    Let me just implement a direct search or use a known construction.
    
    Given time constraints, let me use a rank-56 construction based on the following:
    For each pair of rows of A (there's only 1 pair since n=2) and for each 2x2 block 
    of the 4x8 matrix (there are 2*4=8 such blocks if we split j into pairs and k into pairs),
    we can use Strassen to reduce each 2x2x2 block from 8 to 7 multiplications.
    
    Split j={0,1,2,3} into pairs (0,1) and (2,3), and k={0,...,7} into pairs.
    For block (j-pair, k-pair), we compute a 2x2 product using 7 multiplications instead of 8.
    
    Number of such blocks: 2 (j-pairs) * 4 (k-pairs) = 8 blocks.
    Each block: 7 multiplications. Total: 8*7 = 56.
    
    This gives rank 56.
    """
    n, m, p = DIMENSIONS
    # rank = 56 using block Strassen
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Block Strassen for 2x4x8
    # Split j into 2 pairs: (0,1) and (2,3)
    # Split k into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # For each (j-pair, k-pair), use Strassen's 7-multiplication algorithm
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for jp in range(2):
        j0, j1 = j_pairs[jp]
        for kp in range(4):
            k0, k1 = k_pairs[kp]
            # Strassen for 2x2 block:
            # A_block = [[A[0,j0], A[0,j1]], [A[1,j0], A[1,j1]]]
            # B_block = [[B[j0,k0], B[j0,k1]], [B[j1,k0], B[j1,k1]]]
            # C_block = A_block * B_block
            # 
            # Strassen computes:
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B01-B11)
            # M4 = A11*(B10-B00)
            # M5 = (A00+A01)*B11
            # M6 = (A10-A00)*(B00+B01)
            # M7 = (A01-A11)*(B10+B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0 * m + j0][col] = 1.0
            U[1 * m + j1][col] = 1.0
            V[j0 * p + k0][col] = 1.0
            V[j1 * p + k1][col] = 1.0
            W[k0 * n + 0][col] = 1.0
            W[k0 * n + 1][col] = 1.0
            W[k1 * n + 0][col] = -1.0
            W[k1 * n + 1][col] = 1.0
            col += 1
            
            # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1 * m + j0][col] = 1.0
            U[1 * m + j1][col] = 1.0
            V[j0 * p + k0][col] = 1.0
            W[k0 * n + 1][col] = 1.0
            col += 1
            
            # M3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0 * m + j0][col] = 1.0
            V[j0 * p + k1][col] = 1.0
            V[j1 * p + k1][col] = -1.0
            W[k1 * n + 0][col] = 1.0
            W[k1 * n + 1][col] = 1.0
            col += 1
            
            # M4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1 * m + j1][col] = 1.0
            V[j1 * p + k0][col] = 1.0
            V[j0 * p + k0][col] = -1.0
            W[k0 * n + 0][col] = 1.0
            W[k0 * n + 1][col] = 1.0
            col += 1
            
            # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0 * m + j0][col] = 1.0
            U[0 * m + j1][col] = 1.0
            V[j1 * p + k1][col] = 1.0
            W[k1 * n + 0][col] = -1.0
            W[k1 * n + 1][col] = -1.0
            col += 1
            
            # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1 * m + j0][col] = 1.0
            U[0 * m + j0][col] = -1.0
            V[j0 * p + k0][col] = 1.0
            V[j0 * p + k1][col] = 1.0
            W[k0 * n + 1][col] = -1.0
            W[k1 * n + 1][col] = -1.0
            col += 1
            
            # M7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0 * m + j1][col] = 1.0
            U[1 * m + j1][col] = -1.0
            V[j1 * p + k0][col] = 1.0
            V[j1 * p + k1][col] = 1.0
            W[k0 * n + 0][col] = 1.0
            W[k1 * n + 0][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
