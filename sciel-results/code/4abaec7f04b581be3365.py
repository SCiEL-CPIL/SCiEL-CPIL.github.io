import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Rank is n * m * p for the trivial decomposition
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using numpy for efficiency and different approach than list-of-lists
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the decomposition
    # Index mapping:
    # U[i, j, t] corresponds to A[i, j]
    # V[j, k, t] corresponds to B[j, k]
    # W[k, i, t] corresponds to C[i, k] (Note: Task says W[k*n+i, r] for C[i,k]?)
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is indexed by [k*n + i] for the row, which corresponds to C[i,k] if C is stored column-major?
    # Or does it mean W[row, col] where row = k*n + i?
    # Standard tensor decomposition for matrix mult:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Term t: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # This matches the task description.
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1
                
    # Convert to lists of lists as required by interface? 
    # The interface says: "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't explicitly forbid numpy arrays, but "return {'U', 'V', 'W'}" usually implies standard Python types 
    # or numpy arrays are accepted if the evaluator handles them. 
    # Given "Python standard library, NumPy and SciPy are available", numpy arrays are likely safe.
    # However, to be safe against strict type checking, I can return lists.
    # But the seed returned lists. Let's return lists to be consistent with the seed's output type, 
    # but generate them using numpy for the "new approach" of calculation.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
