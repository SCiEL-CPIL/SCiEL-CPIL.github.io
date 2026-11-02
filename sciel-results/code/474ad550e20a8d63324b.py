import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Block decomposition:
    - j=0: 2x1x5 schoolbook, rank 10
    - j in {1,2}, k in {0,1}: 2x2x2 Strassen, rank 7
    - j in {1,2}, k=2: 2x2x1 schoolbook, rank 4
    - j in {1,2}, k in {3,4}: 2x2x2 Strassen, rank 7
    Total rank: 28
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit with margin
    
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4} — schoolbook 2x1x5, rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    # col = 10
    
    # Helper: add Strassen 2x2x2 block at j offsets [base_j, base_j+1], k offsets [base_k, base_k+1]
    def add_strassen_2x2x2(base_j, base_k):
        nonlocal col
        # M1: (A[0,base_j]+A[1,base_j+1])*(B[base_j,base_k]+B[base_j+1,base_k+1])
        U[0 * m + base_j, col] = 1.0
        U[1 * m + (base_j + 1), col] = 1.0
        V[base_j * p + base_k, col] = 1.0
        V[(base_j + 1) * p + (base_k + 1), col] = 1.0
        W[base_k * n + 0, col] = 1.0
        W[(base_k + 1) * n + 1, col] = 1.0
        col += 1
        
        # M2: (A[1,base_j]+A[1,base_j+1])*B[base_j,base_k]
        U[1 * m + base_j, col] = 1.0
        U[1 * m + (base_j + 1), col] = 1.0
        V[base_j * p + base_k, col] = 1.0
        W[base_k * n + 1, col] = 1.0
        W[(base_k + 1) * n + 1, col] = -1.0
        col += 1
        
        # M3: A[0,base_j]*(B[base_j,base_k+1]-B[base_j+1,base_k+1])
        U[0 * m + base_j, col] = 1.0
        V[base_j * p + (base_k + 1), col] = 1.0
        V[(base_j + 1) * p + (base_k + 1), col] = -1.0
        W[(base_k + 1) * n + 0, col] = 1.0
        W[(base_k + 1) * n + 1, col] = 1.0
        col += 1
        
        # M4: A[1,base_j+1]*(B[base_j+1,base_k]-B[base_j,base_k])
        U[1 * m + (base_j + 1), col] = 1.0
        V[(base_j + 1) * p + base_k, col] = 1.0
        V[base_j * p + base_k, col] = -1.0
        W[base_k * n + 0, col] = 1.0
        W[base_k * n + 1, col] = 1.0
        col += 1
        
        # M5: (A[0,base_j]+A[0,base_j+1])*B[base_j+1,base_k+1]
        U[0 * m + base_j, col] = 1.0
        U[0 * m + (base_j + 1), col] = 1.0
        V[(base_j + 1) * p + (base_k + 1), col] = 1.0
        W[base_k * n + 0, col] = -1.0
        W[(base_k + 1) * n + 0, col] = 1.0
        col += 1
        
        # M6: (A[1,base_j]-A[0,base_j])*(B[base_j,base_k]+B[base_j,base_k+1])
        U[1 * m + base_j, col] = 1.0
        U[0 * m + base_j, col] = -1.0
        V[base_j * p + base_k, col] = 1.0
        V[base_j * p + (base_k + 1), col] = 1.0
        W[(base_k + 1) * n + 1, col] = 1.0
        col += 1
        
        # M7: (A[0,base_j+1]-A[1,base_j+1])*(B[base_j+1,base_k]+B[base_j+1,base_k+1])
        U[0 * m + (base_j + 1), col] = 1.0
        U[1 * m + (base_j + 1), col] = -1.0
        V[(base_j + 1) * p + base_k, col] = 1.0
        V[(base_j + 1) * p + (base_k + 1), col] = 1.0
        W[base_k * n + 0, col] = 1.0
        col += 1
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded during construction")
    add_strassen_2x2x2(1, 0)
    # col = 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1, rank 4
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1
    # col = 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded during construction")
    add_strassen_2x2x2(1, 3)
    # col = 28
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Verify decomposition correctness
    if time.time() < deadline - 5:
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if not np.allclose(tensor, np.ones((n, m, p)), atol=1e-9):
            raise ValueError("Decomposition verification failed")
    
    # Convert to lists with integer entries
    U_list = [[int(round(val)) for val in row] for row in U.tolist()]
    V_list = [[int(round(val)) for val in row] for row in V.tolist()]
    W_list = [[int(round(val)) for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
