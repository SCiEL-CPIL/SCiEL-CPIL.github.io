import numpy as np

def construct():
    """
    Returns a rank-14 exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    U: (n*m, rank) = (4, 14)
    V: (m*p, rank) = (8, 14)
    W: (p*n, rank) = (8, 14)
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Indices for A (flattened): 00->0, 01->1, 10->2, 11->3
    # Indices for B (flattened): 00->0, 01->1, 10->2, 11->3 (local to block)
    # Indices for C (flattened): 00->0, 01->1, 10->2, 11->3 (local to block)
    
    # Strassen terms: (u_indices, v_indices, w_indices)
    # Each term is a list of (index, coefficient)
    strassen_terms = [
        # M1 = (a00 + a11) * (b00 + b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10 + a11) * b00
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3 = a00 * (b01 - b11)
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4 = a11 * (b10 - b00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5 = (a00 + a01) * b11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6 = (a10 - a00) * (b00 + b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01 - a11) * (b10 + b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Apply to first block (B columns 0,1 -> C columns 0,1)
    # B global index: j * p + k. Here j in {0,1}, k in {0,1}.
    # C global index: k * n + i. Here k in {0,1}, i in {0,1}.
    # Wait, the trace convention is W[k*n+i, r].
    # C is p x n. So C[k, i] where k is column of C (0..p-1), i is row of C (0..n-1).
    # In standard multiplication C = A B, C[i, k] = sum A[i,j] B[j,k].
    # The problem statement says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the output matrix C is indexed by [k, i] where k is the column index of B (0..p-1) and i is the row index of A (0..n-1).
    # So C is stored in column-major order relative to the standard [i, k] matrix, or simply flattened as k*n+i.
    # Let's verify the indices in the current program.
    # Current program:
    # global_w_idx = k * 2 + i.
    # This matches k*n + i with n=2.
    
    for r_local in range(7):
        r_global = r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx, r_global] += val
            
        for idx, val in v_terms:
            j = idx // 2
            k = idx % 2
            # B is 2x4. Flattened as j*p + k? 
            # B is m x p = 2 x 4.
            # Standard flattening: row-major? 
            # The problem doesn't specify B flattening, but V is (m*p, rank).
            # Usually V[j, k] corresponds to B[j, k].
            # If V is flattened row-major: index = j * p + k.
            # Let's check the current program's V indexing.
            # global_v_idx = j * 4 + k. Yes, row-major for B (2x4).
            global_v_idx = j * p + k
            V[global_v_idx, r_global] += val
            
        for idx, val in w_terms:
            k = idx // 2
            i = idx % 2
            # W is p x n = 4 x 2.
            # Flattened as k * n + i.
            global_w_idx = k * n + i
            W[global_w_idx, r_global] += val

    # Apply to second block (B columns 2,3 -> C columns 2,3)
    # B local indices j in {0,1}, k in {0,1} map to global k in {2,3}.
    # C local indices k in {0,1}, i in {0,1} map to global k in {2,3}.
    
    for r_local in range(7):
        r_global = 7 + r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx, r_global] += val
            
        for idx, val in v_terms:
            j = idx // 2
            k_local = idx % 2
            k_global = k_local + 2
            global_v_idx = j * p + k_global
            V[global_v_idx, r_global] += val
            
        for idx, val in w_terms:
            k_local = idx // 2
            i = idx % 2
            k_global = k_local + 2
            global_w_idx = k_global * n + i
            W[global_w_idx, r_global] += val

    # Convert to lists of lists as required by the interface?
    # The interface says: "factor matrices of shapes ... with integer/half-integer entries"
    # The current program returns lists of lists.
    # I should return lists of lists to be safe, or numpy arrays if accepted.
    # The seed program returned lists. I will return lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
