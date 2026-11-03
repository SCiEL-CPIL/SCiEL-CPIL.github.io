import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    try:
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False, "Shape mismatch"
        
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False, "Rank mismatch"
            
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val - 1.0) > tol:
                        return False, (a, b, c, val)
        
        # Check invalid entries (sample)
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j != j2 or k != k2:
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val) > tol:
                    return False, (a, b, c, val)
        return True, None
    except Exception as e:
        return False, str(e)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: Full Schoolbook (Rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt Rank 27 (9 + 18)
    try:
        # Part 1: 3x3x3 (first 3 columns of B)
        rank_1 = n * m * 3
        # Part 2: 3x3x2 (last 2 columns of B)
        rank_2 = n * m * 2
        
        total_rank = rank_1 + rank_2
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        # Fill Part 1 (k=0,1,2)
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Fill Part 2 (k=3,4)
        for i in range(n):
            for j in range(m):
                for k in range(3, 5):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Verify
        valid, err = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            print(f"Verification failed for rank {total_rank}: {err}")
            
    except Exception as e:
        print(f"Error in rank 27 construction: {e}")

    # Return the valid Rank 45 decomposition
    return get_schoolbook()
