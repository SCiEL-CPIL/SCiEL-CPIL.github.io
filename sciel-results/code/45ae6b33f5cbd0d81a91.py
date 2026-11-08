import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    Strategy: Use the tensor product structure M_{4,4,8} = M_{4,4,4} ⊗ M_{2,1,1}
    and apply rank-reduction via Strassen's algorithm at multiple levels.
    
    The 4x4x4 multiplication can be decomposed using Strassen recursively:
    - 2x2x2 has rank 7 (Strassen)
    - 4x4x4 can be done with rank <= 49 (Strassen applied to 2x2 blocks)
    
    For 4x4x8, we view it as two 4x4x4 multiplications (columns 0-3 and 4-7).
    Naively this gives rank 98, but we can potentially share structure.
    
    Actually, a better approach: view 4x4x8 as 4x4x4 ⊗ 2x1x1.
    The 2x1x1 tensor has rank 1 (it's just scalar multiplication).
    So M_{4,4,8} = M_{4,4,4} ⊗ M_{2,1,1} has rank <= rank(M_{4,4,4}) * 1 = rank(M_{4,4,4}).
    
    Wait, that's not right. Let me think more carefully.
    
    Actually, for A(4x4) * B(4x8) = C(4x8), we can write B = [B1 | B2] where B1, B2 are 4x4.
    Then C = [A*B1 | A*B2].
    
    This is equivalent to two independent 4x4*4x4 multiplications, so rank <= 2 * rank(4x4x4).
    
    With Strassen, rank(4x4x4) <= 49, so rank(4x4x8) <= 98.
    
    But we can do better by using the structure more cleverly. Let me implement the
    Strassen-based decomposition and verify it works.
    """
    n, m, p = 4, 4, 8
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strassen 2x2x2 rank-7 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 
    # P0 = (a00+a11)(b00+b11)
    # P1 = (a10+a11)b00
    # P2 = a00(b10-b11)
    # P3 = a11(b00-b10)
    # P4 = (a00+a01)b11
    # P5 = (a10-a00)(b00+b01)
    # P6 = (a01-a11)(b10+b11)
    #
    # c00 = P0 + P3 - P4 + P6
    # c01 = P2 + P4
    # c10 = P1 + P3
    # c11 = P0 - P1 + P2 + P5
    
    def get_strassen_2x2():
        """Returns U, V, W for 2x2*2x2 -> 2x2, rank 7."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # P0: (a00+a11)(b00+b11)
        U[0, 0] = 1; U[2, 0] = 1
        V[0, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P1: (a10+a11)b00
        U[1, 1] = 1; U[2, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # P2: a00(b10-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # P3: a11(b00-b10)
        U[2, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # P4: (a00+a01)b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # P5: (a10-a00)(b00+b01)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P6: (a01-a11)(b10+b11)
        U[1, 6] = 1; U[2, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Build 4x4x4 using Strassen on 2x2 blocks
    def get_strassen_4x4():
        """
        Returns U, V, W for 4x4*4x4 -> 4x4, rank <= 49.
        Uses Strassen on the 2x2 block structure.
        """
        U2, V2, W2 = get_strassen_2x2()
        
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        # Block indices for 4x4 matrix split into 2x2 blocks:
        # Block (bi, bj) contains elements at positions:
        # (bi*2+i)*4 + (bj*2+j) for i,j in {0,1}
        
        def block_elems(bi, bj):
            """Returns the 4 flat indices for block (bi, bj) in a 4x4 matrix."""
            return [(bi*2+i)*4 + (bj*2+j) for i in range(2) for j in range(2)]
        
        # The 7 Strassen products at the 4x4 level:
        # Each product involves sums/differences of 2x2 blocks
        products = [
            # (A_blocks, B_blocks) where each is list of (block_id, coeff)
            # Block IDs: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P0: (A00+A11)(B00+B11)
            ([(2, 1), (3, 1)], [(0, 1)]),            # P1: (A10+A11)B00
            ([(0, 1)], [(2, 1), (3, -1)]),           # P2: A00(B10-B11)
            ([(3, 1)], [(0, 1), (2, -1)]),           # P3: A11(B00-B10)
            ([(0, 1), (1, 1)], [(3, 1)]),            # P4: (A00+A01)B11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # P5: (A10-A00)(B00+B01)
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # P6: (A01-A11)(B10+B11)
        ]
        
        # Result combinations for C blocks:
        # C00 = P0 + P3 - P4 + P6
        # C01 = P2 + P4
        # C10 = P1 + P3
        # C11 = P0 - P1 + P2 + P5
        c_combos = [
            [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
            [(2, 1), (4, 1)],                     # C01
            [(1, 1), (3, 1)],                     # C10
            [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
        ]
        
        for p_idx in range(7):
            A_blocks = products[p_idx][0]
            B_blocks = products[p_idx][1]
            
            for s in range(7):
                col = p_idx * 7 + s
                
                # Add to U from A blocks
                for block_id, coeff in A_blocks:
                    bi, bj = block_id // 2, block_id % 2
                    elems = block_elems(bi, bj)
                    for local_idx, global_idx in enumerate(elems):
                        U[global_idx, col] += coeff * U2[local_idx, s]
                
                # Add to V from B blocks
                for block_id, coeff in B_blocks:
                    bi, bj = block_id // 2, block_id % 2
                    elems = block_elems(bi, bj)
                    for local_idx, global_idx in enumerate(elems):
                        V[global_idx, col] += coeff * V2[local_idx, s]
                
                # Add to W from C combinations
                for c_block_id, coeff in c_combos[0]:  # This is wrong, need to handle properly
                    pass
                
                # Let me redo the W part correctly
                # For each C block, check if this product contributes to it
                for c_block_idx in range(4):
                    for prod_idx, coeff in c_combos[c_block_idx]:
                        if prod_idx == p_idx:
                            bi, bj = c_block_idx // 2, c_block_idx % 2
                            elems = block_elems(bi, bj)
                            for local_idx, global_idx in enumerate(elems):
                                W[global_idx, col] += coeff * W2[local_idx, s]
        
        return U, V, W
    
    # Build 4x4x8 by using two 4x4x4 decompositions
    def get_4x4x8_strassen():
        """
        Returns U, V, W for 4x4*4x8 -> 4x8, rank <= 98.
        Splits B into two 4x4 halves.
        """
        U44, V44, W44 = get_strassen_4x4()
        
        rank = 98
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((32, rank), dtype=float)
        W = np.zeros((32, rank), dtype=float)
        
        # First half: B[:, 0:4] -> V indices 0..15, C[:, 0:4] -> W indices 0..15
        U[:, :49] = U44
        V[:16, :49] = V44
        W[:16, :49] = W44
        
        # Second half: B[:, 4:8] -> V indices 16..31, C[:, 4:8] -> W indices 16..31
        U[:, 49:] = U44
        V[16:, 49:] = V44
        W[16:, 49:] = W44
        
        return U, V, W
    
    # Verify a decomposition
    def verify_decomposition(U, V, W, n=4, m=4, p=8, seed=42):
        """Verify that the decomposition correctly computes matrix multiplication."""
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_comp = np.zeros((n, p))
        rank = U.shape[1]
        
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_comp[i, k] = val
        
        return np.allclose(C_expected, C_comp, atol=1e-9)
    
    # Try the Strassen-based decomposition
    best_rank = 128
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook fallback
    U_school = np.zeros((16, 128))
    V_school = np.zeros((32, 128))
    W_school = np.zeros((32, 128))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U_school[i*4+j, col] = 1
                V_school[j*8+k, col] = 1
                W_school[k*4+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 128
    
    if time.time() < deadline - 30:
        try:
            U, V, W = get_4x4x8_strassen()
            if verify_decomposition(U, V, W):
                best_rank = 98
                best_U, best_V, best_W = U, V, W
        except Exception as e:
            pass
    
    # Return the best result found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
