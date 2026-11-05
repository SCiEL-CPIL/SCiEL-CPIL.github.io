import numpy as np

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check shapes
    assert U.shape == (n*m, rank), f"U shape {U.shape} != ({n*m}, {rank})"
    assert V.shape == (m*p, rank), f"V shape {V.shape} != ({m*p}, {rank})"
    assert W.shape == (p*n, rank), f"W shape {W.shape} != ({p*n}, {rank})"
    
    # For each (i,j,k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                result = np.sum(u_val * v_val * w_val)
                if abs(result - 1.0) > 1e-9:
                    print(f"FAIL: i={i}, j={j}, k={k}, result={result}")
                    return False
    return True

def construct():
    n, m, p = 3, 4, 6
    rank = n * m * p  # Schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# Test
result = construct()
print(verify_decomposition(result["U"], result["V"], result["W"], 3, 4, 6))
