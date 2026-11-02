import numpy as np
import time

def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    This uses Strassen's algorithm for the 4x4 * 4x4 blocks.
    """
    # Dimensions
    n, m, p = 4, 4, 8
    
    # Strassen 2x2 * 2x2 decomposition (rank 7)
    # X, Y, Z are 2x2 matrices flattened to length 4: [00, 01, 10, 11]
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]),   # M2
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),  # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [1, 0, 0, 0]),  # M4
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),  # M5
        ([1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, 1]),  # M6
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),  # M7
    ]
    
    # Top-level Strassen for 4x4 * 4x4 (rank 49)
    # Blocks: 11, 12, 21, 22
    # Flat indices for 4x4 matrix:
    # 11: 0, 1, 4, 5
    # 12: 2, 3, 6, 7
    # 21: 8, 9, 12, 13
    # 22: 10, 11, 14, 15
    
    block_indices = {
        11: [0, 1, 4, 5],
        12: [2, 3, 6, 7],
        21: [8, 9, 12, 13],
        22: [10, 11, 14, 15]
    }
    
    # Top-level terms: (A_blocks, B_blocks, C_blocks)
    # Coefficients are applied to the 2x2 sub-terms
    top_level = [
        ({11: 1, 22: 1}, {11: 1, 22: 1}, {11: 1, 22: 1}),
        ({22: 1}, {11: 1, 12: 1}, {21: 1, 22: 1}),
        ({11: 1, 12: 1}, {22: 1}, {12: 1, 22: -1}),
        ({11: 1}, {21: 1, 22: -1}, {11: 1}),
        ({21: 1, 22: 1}, {11: 1}, {11: -1, 21: 1}),
        ({11: 1, 21: -1}, {12: 1, 22: 1}, {12: 1, 22: 1}),
        ({12: 1, 22: -1}, {21: 1, 11: 1}, {11: 1})
    ]
    
    rank_4x4 = 49
    U_4x4 = np.zeros((16, rank_4x4), dtype=int)
    V_4x4 = np.zeros((16, rank_4x4), dtype=int)
    W_4x4 = np.zeros((16, rank_4x4), dtype=int)
    
    col = 0
    for t in range(7):
        A_blocks, B_blocks, C_blocks = top_level[t]
        for s in range(7):
            X_vec, Y_vec, Z_vec = strassen_2x2[s]
            
            for b_id, coeff in A_blocks.items():
                idxs = block_indices[b_id]
                for k in range(4):
                    U_4x4[idxs[k], col] += coeff * X_vec[k]
            
            for b_id, coeff in B_blocks.items():
                idxs = block_indices[b_id]
                for k in range(4):
                    V_4x4[idxs[k], col] += coeff * Y_vec[k]
            
            for b_id, coeff in C_blocks.items():
                idxs = block_indices[b_id]
                for k in range(4):
                    W_4x4[idxs[k], col] += coeff * Z_vec[k]
            
            col += 1
            
    # Full 4x4 * 4x8 decomposition
    # Split B into B1 (cols 0-3) and B2 (cols 4-7)
    # Split C into C1 (cols 0-3) and C2 (cols 4-7)
    # U is 16x98, V is 32x98, W is 32x98
    
    rank_total = 98
    U = np.zeros((16, rank_total), dtype=int)
    V = np.zeros((32, rank_total), dtype=int)
    W = np.zeros((32, rank_total), dtype=int)
    
    # Block 1
    U[:, 0:49] = U_4x4
    V[0:16, 0:49] = V_4x4
    W[0:16, 0:49] = W_4x4
    
    # Block 2
    U[:, 49:98] = U_4x4
    V[16:32, 49:98] = V_4x4
    W[16:32, 49:98] = W_4x4
    
    # Convert to lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    np.random.seed(0)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 8)
    C_expected = A @ B
    
    C_computed = np.zeros((4, 8))
    for c in range(98):
        u = U[:, c].reshape(4, 4)
        v = V[:, c].reshape(4, 8)
        w = W[:, c].reshape(4, 8)
        C_computed += np.sum(u * A) * np.sum(v * B) * w
        
    print("Max Error:", np.max(np.abs(C_computed - C_expected)))
