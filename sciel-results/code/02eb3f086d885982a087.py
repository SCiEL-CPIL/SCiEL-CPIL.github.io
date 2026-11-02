import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Rank 21 is achieved by decomposing the problem into three 2x2x2 blocks and using Strassen's algorithm.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    rank_total = 21

    U = np.zeros((nm, rank_total), dtype=int)
    V = np.zeros((mp, rank_total), dtype=int)
    W = np.zeros((pn, rank_total), dtype=int)

    # Strassen coefficients for one 2x2x2 block
    # Indices for A, B, C are 00, 01, 10, 11
    strassen_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    strassen_w = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]

    # Mapping local indices (00, 01, 10, 11) to global indices for each block g
    # Block g corresponds to B columns 2g, 2g+1 and C rows 2g, 2g+1
    
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # V indices: B is 2x6. Local B for block g is rows 0,1 and cols k0, k1.
        # Global B index = row * 6 + col
        # Local 00 -> (0, k0) -> 0*6 + k0
        # Local 01 -> (0, k1) -> 0*6 + k1
        # Local 10 -> (1, k0) -> 1*6 + k0
        # Local 11 -> (1, k1) -> 1*6 + k1
        v_idx = [k0, k1, 6 + k0, 6 + k1]
        
        # W indices: C is 6x2. Local C for block g is rows k0, k1 and cols 0, 1.
        # Global C index = row * 2 + col
        # Local 00 -> (k0, 0) -> k0*2 + 0
        # Local 01 -> (k1, 0) -> k1*2 + 0
        # Local 10 -> (k0, 1) -> k0*2 + 1
        # Local 11 -> (k1, 1) -> k1*2 + 1
        w_idx = [2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1]
        
        # U indices are always 0, 1, 2, 3 for A (2x2)
        u_idx = [0, 1, 2, 3]
        
        for s in range(7):
            r = 7 * g + s
            
            for local_idx in range(4):
                u_val = strassen_u[s][local_idx]
                if u_val != 0:
                    U[u_idx[local_idx], r] = u_val
                    
                v_val = strassen_v[s][local_idx]
                if v_val != 0:
                    V[v_idx[local_idx], r] = v_val
                    
                w_val = strassen_w[s][local_idx]
                if w_val != 0:
                    W[w_idx[local_idx], r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank {len(result['U'][0])} decomposition generated.")
