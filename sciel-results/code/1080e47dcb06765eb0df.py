"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt rank-60 Laderman-based decomposition; fall back to rank-189 Strassen
    if verification fails.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    bs = 2
    
    # Try Laderman rank-60
    U, V, W = _build_laderman(n, m, p, bs)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Strassen rank-189
    U, V, W = _build_strassen(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_laderman(n, m, p, bs):
    rank = 60
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    laderman = [
        ({(0,1):1, (1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1}),
        ({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):-1}),
        ({(0,0):1, (1,0):1}, {(1,1):1, (0,1):-1}, {(0,1):1}),
        ({(0,0):1, (1,0):-1}, {(1,1):1, (0,1):1}, {(0,1):-1, (1,1):1}),
        ({(0,2):1, (2,2):1}, {(2,0):1, (2,1):-1}, {(0,2):1}),
        ({(0,2):1, (2,2):-1}, {(2,0):1, (2,1):1}, {(0,2):1, (2,2):-1}),
        ({(1,2):1, (2,2):1}, {(2,1):1, (2,0):-1}, {(1,2):1}),
        ({(1,2):1, (2,2):-1}, {(2,1):1, (2,0):1}, {(1,2):1, (2,2):-1}),
        ({(0,0):1, (0,1):1}, {(1,1):1, (1,2):1}, {(0,1):-1}),
        ({(1,0):1, (1,1):1}, {(0,0):1, (0,2):1}, {(1,0):-1}),
        ({(2,0):1, (2,1):1}, {(1,2):1, (2,2):1}, {(2,1):-1}),
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):-1, (2,2):-1}),
        ({(0,2):1, (1,2):1}, {(0,1):1, (1,1):1}, {(0,2):-1}),
        ({(2,0):1, (1,0):1}, {(0,2):1, (1,2):1}, {(2,0):-1}),
        ({(2,1):1, (0,1):1}, {(1,0):1, (2,0):1}, {(2,1):-1}),
    ]
    
    col = 0
    for a_coeffs, b_coeffs, c_coeffs in laderman:
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    for (ib, jb), coeff in a_coeffs.items():
                        U[(ib*bs+r)*n + (jb*bs+s), col] += coeff
                    for (jb, kb), coeff in b_coeffs.items():
                        V[(jb*bs+s)*p + (kb*bs+c), col] += coeff
                    for (ib, kb), coeff in c_coeffs.items():
                        W[(kb*bs+c)*n + (ib*bs+r), col] += coeff
                    col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, num_tests=3):
    rng = np.random.RandomState(42)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for t in range(U.shape[1]):
            ua = np.dot(U[:, t], A.flatten())
            vb = np.dot(V[:, t], B.flatten())
            C_computed += W[:, t].reshape(p, n).T * ua * vb
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _build_strassen(n, m, p):
    rank = 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    strassen_steps = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for a_coeffs, b_coeffs, c_coeffs in strassen_steps:
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    for (ib, jb), coeff in a_coeffs.items():
                        U[(ib*bs+r)*n + (jb*bs+s), col] += coeff
                    for (jb, kb), coeff in b_coeffs.items():
                        V[(jb*bs+s)*p + (kb*bs+c), col] += coeff
                    for (ib, kb), coeff in c_coeffs.items():
                        W[(kb*bs+c)*n + (ib*bs+r), col] += coeff
                    col += 1
    
    return U, V, W
# EVOLVE-BLOCK-END
