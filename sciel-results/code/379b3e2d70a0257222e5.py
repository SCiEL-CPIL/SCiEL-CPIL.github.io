import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p, num_trials=20, seed=42):
    """Verify decomposition with random integer inputs using vectorized ops."""
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    for _ in range(num_trials):
        A = rng.randint(-10, 11, size=(n, m)).astype(float)
        B = rng.randint(-10, 11, size=(m, p)).astype(float)
        C_direct = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()
        u_sums = U.T @ A_flat
        v_sums = V.T @ B_flat
        scalars = u_sums * v_sums
        C_flat = W @ scalars
        C_decomp = C_flat.reshape(p, n).T

        err = np.max(np.abs(C_decomp - C_direct))
        if err > 1e-9:
            return False, err
    return True, 0.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Rank calculation:
    # Part 1: 5 blocks of 2x2x2 using Strassen (rank 7 each) -> 35
    # Part 2: 1 block of 2x1x10 using Schoolbook (rank 20) -> 20
    # Total Rank = 55
    rank = 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Strassen decomposition for 2x2 x 2x2
    # Indices for A: (0,0), (0,1), (1,0), (1,1) -> flat indices 0, 1, 3, 4 (if row major i*m+k? No, i*m+k is 0,1,3,4 for m=2? 
    # Wait, A is 2x3. 
    # In the block 2x2, we use A[0,0], A[0,1], A[1,0], A[1,1].
    # Flat index in U (size n*m = 6):
    # A[i, k] is at index i*m + k.
    # For k in {0,1}:
    # A[0,0] -> 0*3+0 = 0
    # A[0,1] -> 0*3+1 = 1
    # A[1,0] -> 1*3+0 = 3
    # A[1,1] -> 1*3+1 = 4
    
    # For B (size m*p = 30):
    # B[k, j] is at index k*p + j.
    # For block b (j in 2b, 2b+1):
    # B[0, 2b] -> 0*10 + 2b
    # B[0, 2b+1] -> 0*10 + 2b+1
    # B[1, 2b] -> 1*10 + 2b
    # B[1, 2b+1] -> 1*10 + 2b+1
    
    # For C (size p*n = 20):
    # C[j, i] is at index j*n + i.
    # C[2b, 0] -> 2b*2 + 0
    # C[2b, 1] -> 2b*2 + 1
    # C[2b+1, 0] -> (2b+1)*2 + 0
    # C[2b+1, 1] -> (2b+1)*2 + 1

    for b in range(5):
        j0 = 2 * b
        j1 = 2 * b + 1
        
        # A indices
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        # B indices
        b00 = 0 * 10 + j0
        b01 = 0 * 10 + j1
        b10 = 1 * 10 + j0
        b11 = 1 * 10 + j1
        
        # C indices
        c00 = j0 * 2 + 0
        c01 = j0 * 2 + 1
        c10 = j1 * 2 + 0
        c11 = j1 * 2 + 1

        # M1 = (A00 + A11) * (B00 + B11)
        U[a00, idx] += 1.0
        U[a11, idx] += 1.0
        V[b00, idx] += 1.0
        V[b11, idx] += 1.0
        W[c00, idx] += 1.0
        W[c11, idx] += 1.0
        idx += 1

        # M2 = (A10 + A11) * B00
        U[a10, idx] += 1.0
        U[a11, idx] += 1.0
        V[b00, idx] += 1.0
        W[c10, idx] += 1.0
        W[c11, idx] += -1.0
        idx += 1

        # M3 = A00 * (B01 - B11)
        U[a00, idx] += 1.0
        V[b01, idx] += 1.0
        V[b11, idx] += -1.0
        W[c01, idx] += 1.0
        W[c11, idx] += 1.0
        idx += 1

        # M4 = A11 * (B10 - B00)
        U[a11, idx] += 1.0
        V[b10, idx] += 1.0
        V[b00, idx] += -1.0
        W[c00, idx] += 1.0
        W[c10, idx] += 1.0
        idx += 1

        # M5 = (A00 + A01) * B11
        U[a00, idx] += 1.0
        U[a01, idx] += 1.0
        V[b11, idx] += 1.0
        W[c00, idx] += -1.0
        W[c01, idx] += 1.0
        idx += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[a10, idx] += 1.0
        U[a00, idx] += -1.0
        V[b00, idx] += 1.0
        V[b01, idx] += 1.0
        W[c11, idx] += 1.0
        idx += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[a01, idx] += 1.0
        U[a11, idx] += -1.0
        V[b10, idx] += 1.0
        V[b11, idx] += 1.0
        W[c10, idx] += 1.0
        idx += 1

    # Part 2: Schoolbook for k=2
    # A[0,2], A[1,2]
    # B[2, j] for j in 0..9
    a02 = 0 * 3 + 2 # 2
    a12 = 1 * 3 + 2 # 5
    
    for j in range(10):
        b2j = 2 * 10 + j
        c_j0 = j * 2 + 0
        c_j1 = j * 2 + 1
        
        # C[j, 0] += A[0, 2] * B[2, j]
        U[a02, idx] += 1.0
        V[b2j, idx] += 1.0
        W[c_j0, idx] += 1.0
        idx += 1
        
        # C[j, 1] += A[1, 2] * B[2, j]
        U[a12, idx] += 1.0
        V[b2j, idx] += 1.0
        W[c_j1, idx] += 1.0
        idx += 1

    # Verify correctness
    ok, err = _verify_decomposition(U, V, W, n, m, p, num_trials=20, seed=42)
    if not ok:
        raise RuntimeError(f"Decomposition verification failed: err={err}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
