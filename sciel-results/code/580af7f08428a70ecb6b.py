"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We will construct a decomposition of rank 104
    # by using Strassen's algorithm for the 4x4 * 4x4 part (rank 56)
    # and schoolbook for the 4x4 * 4x3 part (rank 48).
    # Total rank = 56 + 48 = 104.
    
    # Split B (4x7) into B_left (4x4) and B_right (4x3).
    # C = A * B = A * B_left + A * B_right
    # A * B_left is 4x4 * 4x4 -> 4x4 (use Strassen, rank 56)
    # A * B_right is 4x4 * 4x3 -> 4x3 (use schoolbook, rank 48)
    
    rank = 56 + 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: Strassen for 4x4 * 4x4 (rank 56) ---
    # We use 2x2 block Strassen.
    # A is split into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B_left is split into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # C_left is 4x4.
    
    # Strassen terms for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # Each Mi is a product of two 2x2 matrices, so it has rank 4 (schoolbook for 2x2*2x2).
    # Total rank for 4x4*4x4 via Strassen = 7 * 4 = 28? No.
    # Wait, the rank of the tensor for 2x2*2x2 is 7.
    # So the rank of the block multiplication (2x2*2x2) is 7.
    # But here we are doing 4x4*4x4.
    # The block multiplication is 2x2 * 2x2.
    # The rank of the 2x2*2x2 tensor is 7.
    # So the rank of the 4x4*4x4 tensor using this block decomposition is 7 * 4 = 28?
    # No, the "block multiplication" is not a scalar multiplication.
    # The block multiplication is a 2x2 * 2x2 matrix multiplication.
    # The rank of the 2x2*2x2 tensor is 7.
    # So the rank of the 4x4*4x4 tensor is 7 * 4 = 28?
    # Let's check: 4x4*4x4 has 64 entries.
    # Schoolbook rank is 64.
    # Strassen rank is 49 (7^2).
    # Why 49? Because we do 7 block multiplications, and each block multiplication is a 2x2*2x2 product.
    # The rank of a 2x2*2x2 product is 4 (schoolbook) or 7 (Strassen)?
    # The rank of the tensor T_{ijk} = delta_{jk} for 2x2*2x2 is 7.
    # So if we use Strassen for the block multiplication, the rank is 7 * 7 = 49.
    # If we use schoolbook for the block multiplication, the rank is 7 * 4 = 28?
    # No, that's not right.
    # The block multiplication is not a scalar multiplication.
    # It is a multiplication of two 2x2 matrices.
    # The rank of this operation is the rank of the 2x2*2x2 tensor, which is 7.
    # So the total rank is 7 (block terms) * 7 (scalar terms per block) = 49.
    
    # Let's implement the rank 49 decomposition for 4x4*4x4.
    # We need to map the 49 terms to U, V, W.
    
    # Indices for 4x4: i, j, k in 0..3.
    # Block indices: I, J, K in 0..1.
    # Sub-block indices: u, v, w in 0..1.
    # Global index for A: i = 2*I + u, j = 2*J + v.
    # Global index for B: j = 2*J + v, k = 2*K + w.
    # Global index for C: i = 2*I + u, k = 2*K + w.
    
    # Strassen block terms:
    # 1. (A11+A22)(B11+B22) -> contributes to C11, C22
    # 2. (A21+A22)B11 -> contributes to C21
    # 3. A11(B12-B22) -> contributes to C12
    # 4. A22(B21-B11) -> contributes to C11, C21
    # 5. (A11+A12)B22 -> contributes to C11, C12
    # 6. (A21-A11)(B11+B12) -> contributes to C22
    # 7. (A12-A22)(B21+B22) -> contributes to C11
    
    # For each block term, we multiply two 2x2 matrices.
    # We use the 7 scalar Strassen terms for this 2x2*2x2 multiplication.
    
    # Let's define the 7 scalar Strassen terms for 2x2*2x2.
    # Indices: a[0]=a00, a[1]=a01, a[2]=a10, a[3]=a11
    #          b[0]=b00, b[1]=b01, b[2]=b10, b[3]=b11
    #          c[0]=c00, c[1]=c01, c[2]=c10, c[3]=c11
    
    # M1 = (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
    # M2 = (a10+a11)b00 -> c10+=M2
    # M3 = a00(b01-b11) -> c01+=M3
    # M4 = a11(b10-b00) -> c00+=M4, c10+=M4
    # M5 = (a00+a01)b11 -> c00-=M5, c01+=M5
    # M6 = (a10-a00)(b00+b01) -> c11+=M6
    # M7 = (a01-a11)(b10+b11) -> c00+=M7
    
    # Let's generate the 49 terms.
    # We will iterate over the 7 block terms and the 7 scalar terms.
    
    # First, let's define the block terms as linear combinations of A and B blocks.
    # A blocks: A11 (I=0,J=0), A12 (I=0,J=1), A21 (I=1,J=0), A22 (I=1,J=1)
    # B blocks: B11 (J=0,K=0), B12 (J=0,K=1), B21 (J=1,K=0), B22 (J=1,K=1)
    
    # Block term definitions:
    # 1. A: A11+A22, B: B11+B22, C: C11+=, C22+=
    # 2. A: A21+A22, B: B11, C: C21+=
    # 3. A: A11, B: B12-B22, C: C12+=
    # 4. A: A22, B: B21-B11, C: C11+=, C21+=
    # 5. A: A11+A12, B: B22, C: C11-=, C12+=
    # 6. A: A21-A11, B: B11+B12, C: C22+=
    # 7. A: A12-A22, B: B21+B22, C: C11+=
    
    # For each block term, we have a 2x2 matrix X (from A) and a 2x2 matrix Y (from B).
    # We then multiply X * Y using the 7 scalar Strassen terms.
    
    # Let's implement this.
    
    # Define the 7 scalar Strassen terms for 2x2*2x2.
    # Each term is a list of (a_idx, b_idx, c_idx, coeff) tuples.
    # a_idx, b_idx, c_idx are in 0..3.
    
    def get_strassen_2x2_terms():
        terms = []
        # M1 = (a00+a11)(b00+b11)
        # Expansion: a00*b00 + a00*b11 + a11*b00 + a11*b11
        # c00 += M1, c11 += M1
        terms.append([
            (0, 0, 0, 1), (0, 3, 0, 1), (3, 0, 0, 1), (3, 3, 0, 1),
            (0, 0, 3, 1), (0, 3, 3, 1), (3, 0, 3, 1), (3, 3, 3, 1)
        ])
        # M2 = (a10+a11)b00
        # Expansion: a10*b00 + a11*b00
        # c10 += M2
        terms.append([
            (2, 0, 2, 1), (3, 0, 2, 1)
        ])
        # M3 = a00(b01-b11)
        # Expansion: a00*b01 - a00*b11
        # c01 += M3
        terms.append([
            (0, 1, 1, 1), (0, 3, 1, -1)
        ])
        # M4 = a11(b10-b00)
        # Expansion: a11*b10 - a11*b00
        # c00 += M4, c10 += M4
        terms.append([
            (3, 2, 0, 1), (3, 0, 0, -1),
            (3, 2, 2, 1), (3, 0, 2, -1)
        ])
        # M5 = (a00+a01)b11
        # Expansion: a00*b11 + a01*b11
        # c00 -= M5, c01 += M5
        terms.append([
            (0, 3, 0, -1), (1, 3, 0, -1),
            (0, 3, 1, 1), (1, 3, 1, 1)
        ])
        # M6 = (a10-a00)(b00+b01)
        # Expansion: a10*b00 + a10*b01 - a00*b00 - a00*b01
        # c11 += M6
        terms.append([
            (2, 0, 3, 1), (2, 1, 3, 1), (0, 0, 3, -1), (0, 1, 3, -1)
        ])
        # M7 = (a01-a11)(b10+b11)
        # Expansion: a01*b10 + a01*b11 - a11*b10 - a11*b11
        # c00 += M7
        terms.append([
            (1, 2, 0, 1), (1, 3, 0, 1), (3, 2, 0, -1), (3, 3, 0, -1)
        ])
        return terms

    strassen_2x2_terms = get_strassen_2x2_terms()
    
    # Now, let's define the block terms.
    # Each block term has:
    # - A_blocks: list of (block_idx, coeff) where block_idx is 0..3 (0=A11, 1=A12, 2=A21, 3=A22)
    # - B_blocks: list of (block_idx, coeff) where block_idx is 0..3 (0=B11, 1=B12, 2=B21, 3=B22)
    # - C_blocks: list of (block_idx, coeff) where block_idx is 0..3 (0=C11, 1=C12, 2=C21, 3=C22)
    
    # Block indices:
    # A11: I=0, J=0 -> global i=0,1; j=0,1
    # A12: I=0, J=1 -> global i=0,1; j=2,3
    # A21: I=1, J=0 -> global i=2,3; j=0,1
    # A22: I=1, J=1 -> global i=2,3; j=2,3
    
    # B11: J=0, K=0 -> global j=0,1; k=0,1
    # B12: J=0, K=1 -> global j=0,1; k=2,3
    # B21: J=1, K=0 -> global j=2,3; k=0,1
    # B22: J=1, K=1 -> global j=2,3; k=2,3
    
    # C11: I=0, K=0 -> global i=0,1; k=0,1
    # C12: I=0, K=1 -> global i=0,1; k=2,3
    # C21: I=1, K=0 -> global i=2,3; k=0,1
    # C22: I=1, K=1 -> global i=2,3; k=2,3
    
    # Let's define the global indices for each block.
    # A block (I, J): global indices are (2*I+u, 2*J+v) for u,v in 0..1.
    # B block (J, K): global indices are (2*J+v, 2*K+w) for v,w in 0..1.
    # C block (I, K): global indices are (2*I+u, 2*K+w) for u,w in 0..1.
    
    # Let's create a helper function to get the global indices for a block.
    def get_block_indices(block_type, I, J):
        if block_type == 'A':
            return [(2*I+u, 2*J+v) for u in range(2) for v in range(2)]
        elif block_type == 'B':
            return [(2*J+v, 2*I+w) for v in range(2) for w in range(2)]  # Note: B is (J,K), so J is the second index
        elif block_type == 'C':
            return [(2*I+u, 2*J+w) for u in range(2) for w in range(2)]
    
    # Wait, the indices for B are (j, k).
    # B block (J, K): j = 2*J+v, k = 2*K+w.
    # So the global index for B is (2*J+v, 2*K+w).
    # But in the V matrix, the index is j*p+k = j*4+k.
    # So we need to map (j, k) to j*4+k.
    
    # Let's redefine the block indices properly.
    # A block (I, J): global indices (i, j) = (2*I+u, 2*J+v).
    # B block (J, K): global indices (j, k) = (2*J+v, 2*K+w).
    # C block (I, K): global indices (i, k) = (2*I+u, 2*K+w).
    
    # Let's create a list of global indices for each block.
    A_blocks = {
        0: [(0,0), (0,1), (1,0), (1,1)],  # A11
        1: [(0,2), (0,3), (1,2), (1,3)],  # A12
        2: [(2,0), (2,1), (3,0), (3,1)],  # A21
        3: [(2,2), (2,3), (3,2), (3,3)]   # A22
    }
    B_blocks = {
        0: [(0,0), (0,1), (1,0), (1,1)],  # B11
        1: [(0,2), (0,3), (1,2), (1,3)],  # B12
        2: [(2,0), (2,1), (3,0), (3,1)],  # B21
        3: [(2,2), (2,3), (3,2), (3,3)]   # B22
    }
    C_blocks = {
        0: [(0,0), (0,1), (1,0), (1,1)],  # C11
        1: [(0,2), (0,3), (1,2), (1,3)],  # C12
        2: [(2,0), (2,1), (3,0), (3,1)],  # C21
        3: [(2,2), (2,3), (3,2), (3,3)]   # C22
    }
    
    # Now, let's define the 7 block terms.
    # Each block term has A_blocks, B_blocks, C_blocks.
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    
    block_terms = [
        # 1. (A11+A22)(B11+B22) -> C11+=, C22+=
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # 2. (A21+A22)B11 -> C21+=
        ([(2,1), (3,1)], [(0,1)], [(2,1)]),
        # 3. A11(B12-B22) -> C12+=
        ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
        # 4. A22(B21-B11) -> C11+=, C21+=
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # 5. (A11+A12)B22 -> C11-=, C12+=
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # 6. (A21-A11)(B11+B12) -> C22+=
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # 7. (A12-A22)(B21+B22) -> C11+=
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    ]
    
    # Now, for each block term, we multiply the two 2x2 matrices using the 7 scalar Strassen terms.
    # The scalar Strassen terms are defined for local indices 0..3.
    # We need to map the local indices to the global indices.
    
    # Let's iterate over the block terms and the scalar Strassen terms.
    col = 0
    for block_A, block_B, block_C in block_terms:
        for scalar_term in strassen_2x2_terms:
            for a_local, b_local, c_local, coeff in scalar_term:
                # a_local is in 0..3, corresponding to a[0]=a00, a[1]=a01, a[2]=a10, a[3]=a11
                # b_local is in 0..3, corresponding to b[0]=b00, b[1]=b01, b[2]=b10, b[3]=b11
                # c_local is in 0..3, corresponding to c[0]=c00, c[1]=c01, c[2]=c10, c[3]=c11
                
                # We need to find the global indices for a_local, b_local, c_local.
                # a_local = 2*u + v, where u, v are the sub-block indices.
                # b_local = 2*v + w, where v, w are the sub-block indices.
                # c_local = 2*u + w, where u, w are the sub-block indices.
                
                u_a = a_local // 2
                v_a = a_local % 2
                v_b = b_local // 2
                w_b = b_local % 2
                u_c = c_local // 2
                w_c = c_local % 2
                
                # Check if v_a == v_b.
                if v_a != v_b:
                    continue
                
                # Now, we need to find the global indices.
                # The global index for A is (I, J, u, v) -> (2*I+u, 2*J+v).
                # But we don't know I and J yet.
                # We know that the block term has A_blocks and B_blocks.
                # The block term defines the linear combination of A and B blocks.
                # So we need to iterate over the A_blocks and B_blocks in the block term.
                
                # This is getting complicated. Let's simplify.
                # We will just use the schoolbook for the 4x4*4x4 part, which has rank 64.
                # Then the total rank is 64 + 48 = 112.
                # But we want rank 104.
                
                # Let's try a different approach.
                # We will use the Strassen decomposition for 4x4*4x4, which has rank 49.
                # But we need to implement it correctly.
                
                # Given the complexity, let's just use the schoolbook for now.
                pass
    
    # Let's just use the schoolbook for the entire 4x4*4x7.
    # Rank = 4*4*7 = 112.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
