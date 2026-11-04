import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    block_size = 3
    num_blocks = 2

    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps defined as (A_terms, B_terms, C_terms)
    # Each term is (I, J, coeff)
    steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = block_size
    col_idx = 0
    
    # Pre-compute indices for speed and distinct structure
    # We iterate over Strassen steps, then over the 3x3 sub-blocks
    for step in steps:
        a_terms, b_terms, c_terms = step
        for r, s, t in itertools.product(range(bs), repeat=3):
            # Process U
            for I, J, ca in a_terms:
                u_idx = (I * bs + r) * dim + (J * bs + s)
                U[u_idx, col_idx] += ca

            # Process V
            for J, K, cb in b_terms:
                v_idx = (J * bs + s) * dim + (K * bs + t)
                V[v_idx, col_idx] += cb

            # Process W
            for I, K, cc in c_terms:
                w_idx = (K * bs + t) * dim + (I * bs + r)
                W[w_idx, col_idx] += cc

            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Check for linear dependencies and reduce rank if possible
    if time.time() < deadline - 10:
        # Stack U, V, W into a single matrix for SVD analysis
        # Each column k represents a rank-1 term: U[:,k] ⊗ V[:,k] ⊗ W[:,k]
        # Check if any columns are linearly dependent
        U_svd = np.linalg.svd(U, full_matrices=False)
        V_svd = np.linalg.svd(V, full_matrices=False)
        W_svd = np.linalg.svd(W, full_matrices=False)
        
        # The rank of each factor matrix tells us about dependencies
        # But we need to check if the tensor decomposition has redundant terms
        # A simpler check: see if any pair of columns produces the same outer product pattern
        
        # For now, just return the current decomposition
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
