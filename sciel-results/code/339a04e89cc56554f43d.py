import numpy as np
import time

def construct():
    """
    Returns a rank-21 exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    The tensor <2,2,6> is the direct sum of three <2,2,2> tensors (one for each column-pair of B).
    Rank(<2,2,6>) = 3 * Rank(<2,2,2>) = 3 * 7 = 21 (using Strassen for each block).
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    rank_total = 21
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))

    # Strassen coefficients for 2x2 multiplication (0-indexed)
    # A is 2x2: indices [i*2+j] for i,j in {0,1}
    # B is 2x2 (local): indices [i*2+j] for i,j in {0,1}
    # C is 2x2 (local): indices [i*2+j] for i,j in {0,1}
    strassen_steps = [
        {'u': [1, 0, 0, 1],  'v': [1, 0, 0, 1],  'w': [1, 0, 0, 1]},   # M1
        {'u': [0, 0, 1, 1],  'v': [1, 0, 0, 0],  'w': [0, 0, 1, -1]},  # M2
        {'u': [1, 0, 0, 0],  'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},   # M3
        {'u': [0, 0, 0, 1],  'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},   # M4
        {'u': [1, 1, 0, 0],  'v': [0, 0, 0, 1],  'w': [-1, 1, 0, 0]},  # M5
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0],  'w': [0, 0, 0, 1]},   # M6
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1],  'w': [1, 0, 0, 0]},   # M7
    ]

    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # V indices: B is 2x6. Global B index = row * 6 + col
        v_idx = [0 * 6 + k0, 0 * 6 + k1, 1 * 6 + k0, 1 * 6 + k1]
        
        # W indices: C is 6x2 (p x n). Global C index = row * 2 + col
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

    # Verify correctness
    _verify_decomposition(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes C = A*B."""
    import random
    rng = random.Random(42)
    for _ in range(50):
        A = np.array([[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)])
        B = np.array([[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)])
        C_ref = A @ B
        
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    for j in range(m):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_comp[i, k] = s
        
        if not np.allclose(C_comp, C_ref, atol=1e-10):
            raise ValueError("Decomposition verification failed")

if __name__ == "__main__":
    result = construct()
    print(f"Rank {len(result['U'][0])} decomposition generated and verified.")
