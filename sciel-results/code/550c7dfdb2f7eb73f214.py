import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    The tensor <2,2,6> is the direct sum of three <2,2,2> tensors (one for each column of the 2x6 matrix B).
    The rank of a direct sum is the sum of the ranks, so rank(<2,2,6>) = 3 * rank(<2,2,2>) = 3 * 7 = 21.
    We use the Strassen decomposition for each block, giving a total rank of 21.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    rank_total = 21
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))

    # Strassen coefficients for 2x2 multiplication
    # Each step: u (coeffs for A entries), v (coeffs for B entries), w (coeffs for C entries)
    # A is 2x2, indexed as [i*2+j] for i,j in {0,1}
    # B is 2x2 (local), indexed as [i*2+j] for i,j in {0,1}
    # C is 2x2 (local), indexed as [i*2+j] for i,j in {0,1}
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    # For each of the 3 blocks (corresponding to the 3 column-pairs of the 2x6 matrix B):
    # Block g handles B columns 2g and 2g+1, and C rows 2g and 2g+1.
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # V indices: B is 2x6. Local B for block g is rows 0,1 and cols k0, k1.
        # Global B index = row * 6 + col
        v_idx = [0 * 6 + k0, 0 * 6 + k1, 1 * 6 + k0, 1 * 6 + k1]
        
        # W indices: C is 6x2 (p x n). Local C for block g is rows k0, k1 and cols 0, 1.
        # Global C index = row * 2 + col
        w_idx = [k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1]
        
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            for local_idx in range(4):
                if step['u'][local_idx] != 0:
                    U[local_idx, r] = step['u'][local_idx]
                if step['v'][local_idx] != 0:
                    V[v_idx[local_idx], r] = step['v'][local_idx]
                if step['w'][local_idx] != 0:
                    W[w_idx[local_idx], r] = step['w'][local_idx]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank {len(result['U'][0])} decomposition generated.")
