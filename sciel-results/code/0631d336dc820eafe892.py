import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    # Deadline guard setup
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The rank of 2x4x8 matrix multiplication is 56.
    # This is derived from decomposing the operation into 8 independent 2x2x2 multiplications.
    # Each 2x2x2 multiplication has a minimal rank of 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    total_rank = 56
    num_factors = 8 # 8 blocks of 2x2x2
    strassen_rank = 7

    # Initialize U, V, W as lists of lists to match expected interface if strict typing is needed,
    # but numpy is faster for manipulation. We'll build with numpy and convert.
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Strassen coefficients for 2x2x2 multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # U extracts linear combos of A
    # V extracts linear combos of B
    # W combines products to get C
    
    # Indexing for A (2x2): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Indexing for B (2x2): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Indexing for C (2x2): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Strassen products:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)(B11)
    # P3 = (A11)(B12 - B22)
    # P4 = (A22)(B21 - B11)
    # P5 = (A11 + A12)(B22)
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # U vectors (rows correspond to A elements, columns to strassen terms)
    # A elements: A00, A01, A10, A11
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    
    # V vectors (rows correspond to B elements, columns to strassen terms)
    # B elements: B00, B01, B10, B11
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ]
    
    # W vectors (rows correspond to C elements, columns to strassen terms)
    # C elements: C00, C01, C10, C11
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0] # C11
    ]
    
    # Mapping of the 8 blocks to indices
    # We split M=4 into 2 blocks of size 2: M0, M1
    # We split P=8 into 4 blocks of size 2: P0, P1, P2, P3
    # For each combination (m_block, p_block), we perform a 2x2x2 mult.
    # A is 2x4. A[:, m_block*2 : m_block*2+2] is the 2x2 submatrix.
    # B is 4x8. B[m_block*2 : m_block*2+2, p_block*2 : p_block*2+2] is the 2x2 submatrix.
    # C is 2x8. C[:, p_block*2 : p_block*2+2] receives the result.
    
    block_idx = 0
    for m_block in range(2):
        for p_block in range(4):
            # A submatrix indices:
            # A is 2x4. Row i (0,1), Col j (m_block*2, m_block*2+1)
            a_row_offsets = [0, 1] # Since n=2, rows are 0 and 1
            
            # Map local A indices (0,1,2,3) to global A indices (n*m = 8)
            # Local A[r][c] -> Global A[r * m + (m_block*2 + c)]
            global_a_indices = []
            for lr in range(2):
                for lc in range(2):
                    global_a_indices.append(lr * m + (m_block * 2 + lc))
            
            # Map local B indices (0,1,2,3) to global B indices (m*p = 32)
            # Local B[r][c] -> Global B[(m_block*2 + r) * p + (p_block*2 + c)]
            global_b_indices = []
            for lr in range(2):
                for lc in range(2):
                    global_b_indices.append((m_block * 2 + lr) * p + (p_block * 2 + lc))
            
            # Map local C indices (0,1,2,3) to global C indices (p*n = 16)
            # Local C[r][c] -> Global C[(p_block*2 + c) * n + r]
            global_c_indices = []
            for lr in range(2):
                for lc in range(2):
                    global_c_indices.append((p_block * 2 + lc) * n + lr)
            
            # Add this block's Strassen factors to U, V, W
            for s in range(7):
                rank_idx = block_idx * 7 + s
                
                # Update U
                for local_a_idx, global_a_idx in enumerate(global_a_indices):
                    U[global_a_idx, rank_idx] = U_strassen[local_a_idx][s]
                
                # Update V
                for local_b_idx, global_b_idx in enumerate(global_b_indices):
                    V[global_b_idx, rank_idx] = V_strassen[local_b_idx][s]
                
                # Update W
                for local_c_idx, global_c_idx in enumerate(global_c_indices):
                    W[global_c_idx, rank_idx] = W_strassen[local_c_idx][s]
            
            block_idx += 1

    # Convert to lists if necessary, though numpy arrays are usually accepted
    # The prompt implies standard python structures might be safer if "integer/half-integer" is strict
    # But "NumPy ... are available" suggests arrays are fine.
    # To be safe and distinct from previous list-based implementation, I'll return numpy arrays.
    
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
