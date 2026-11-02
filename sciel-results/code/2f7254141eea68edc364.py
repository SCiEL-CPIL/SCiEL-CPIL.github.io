"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_rank56_strassen():
    """
    Build a rank-56 Strassen decomposition for 2x4 by 4x8 matrix multiplication.
    
    C[k,i] = sum_j A[i,j] * B[j,k]  (k in 0..7, i in 0..1, j in 0..3)
    
    Split into 8 independent 2x2x2 blocks:
      For each B-column pair d in {0,1,2,3} (columns 2d, 2d+1):
        For each A-column half h in {0,1} (A columns 2h, 2h+1):
          2x2 Strassen product (rank 7)
    
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    for d in range(4):       # B column pair: columns 2d, 2d+1
        for h in range(2):   # A column half: columns 2h, 2h+1
            base = (d * 2 + h) * 7
            
            if h == 0:
                def a_idx(r, c): return r * m + c
                def b_idx(r, c): return r * p + (2 * d + c)
            else:
                def a_idx(r, c): return r * m + (2 + c)
                def b_idx(r, c): return (2 + r) * p + (2 * d + c)
            def c_idx(r, c): return (2 * d + c) * n + r
            
            # M1 = (A00 + A11) * (B00 + B11)
            col = base + 0
            U[a_idx(0,0), col] += 1; U[a_idx(1,1), col] += 1
            V[b_idx(0,0), col] += 1; V[b_idx(1,1), col] += 1
            W[c_idx(0,0), col] += 1; W[c_idx(1,1), col] += 1
            
            # M2 = (A10 + A11) * B00
            col = base + 1
            U[a_idx(1,0), col] += 1; U[a_idx(1,1), col] += 1
            V[b_idx(0,0), col] += 1
            W[c_idx(1,0), col] += 1; W[c_idx(1,1), col] -= 1
            
            # M3 = A00 * (B01 - B11)
            col = base + 2
            U[a_idx(0,0), col] += 1
            V[b_idx(0,1), col] += 1; V[b_idx(1,1), col] -= 1
            W[c_idx(0,1), col] += 1; W[c_idx(1,1), col] += 1
            
            # M4 = A11 * (B10 - B00)
            col = base + 3
            U[a_idx(1,1), col] += 1
            V[b_idx(1,0), col] += 1; V[b_idx(0,0), col] -= 1
            W[c_idx(0,0), col] += 1; W[c_idx(1,0), col] += 1
            
            # M5 = (A00 + A01) * B11
            col = base + 4
            U[a_idx(0,0), col] += 1; U[a_idx(0,1), col] += 1
            V[b_idx(1,1), col] += 1
            W[c_idx(0,0), col] -= 1; W[c_idx(0,1), col] += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            col = base + 5
            U[a_idx(1,0), col] += 1; U[a_idx(0,0), col] -= 1
            V[b_idx(0,0), col] += 1; V[b_idx(0,1), col] += 1
            W[c_idx(1,1), col] += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            col = base + 6
            U[a_idx(0,1), col] += 1; U[a_idx(1,1), col] -= 1
            V[b_idx(1,0), col] += 1; V[b_idx(1,1), col] += 1
            W[c_idx(0,0), col] += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that non-matching triples give zero
    import random
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False
    return True


def construct():
    """
    Construct a rank-minimal decomposition for 2x4 by 4x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank56_strassen()
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    valid = _verify_decomposition(U, V, W, n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    ok = _verify_decomposition(U, V, W, n, m, p)
    rank = U.shape[1]
    elapsed = time.time() - t0
    print(f"Rank: {rank}, Valid: {ok}, Time: {elapsed:.2f}s")
