import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms for a 2x2x2 block
    # Each term is a list of (U_indices, V_indices, W_indices)
    # Indices are (row, col) in the local 2x2 matrices
    # We will map these to global indices
    
    for b in range(3):
        base_rank = b * 7
        # Global index offsets
        # U: A[i,j] -> i*m + j
        # V: B[j,k] -> j*p + k
        # W: C[i,k] -> k*n + i
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        U[0][base_rank] += 1
        U[3][base_rank] += 1
        # V: B00, B11
        V[0 * p + 2*b][base_rank] += 1
        V[1 * p + 2*b + 1][base_rank] += 1
        # W: C00, C11
        W[2*b * n + 0][base_rank] += 1
        W[2*b + 1 * n + 1][base_rank] += 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10, A11
        U[2][base_rank + 1] += 1
        U[3][base_rank + 1] += 1
        # V: B00
        V[0 * p + 2*b][base_rank + 1] += 1
        # W: C10 - C11
        W[2*b * n + 1][base_rank + 1] += 1
        W[2*b + 1 * n + 1][base_rank + 1] -= 1
        
        # Term 3: M3 = A00(B01-B11)
        # U: A00
        U[0][base_rank + 2] += 1
        # V: B01, -B11
        V[0 * p + 2*b + 1][base_rank + 2] += 1
        V[1 * p + 2*b + 1][base_rank + 2] -= 1
        # W: C01 + C11
        W[2*b + 1 * n + 0][base_rank + 2] += 1
        W[2*b + 1 * n + 1][base_rank + 2] += 1
        
        # Term 4: M4 = A11(B10-B00)
        # U: A11
        U[3][base_rank + 3] += 1
        # V: B10, -B00
        V[1 * p + 2*b][base_rank + 3] += 1
        V[0 * p + 2*b][base_rank + 3] -= 1
        # W: C00 + C10
        W[2*b * n + 0][base_rank + 3] += 1
        W[2*b * n + 1][base_rank + 3] += 1
        
        # Term 5: M5 = (A00+A01)B11
        # U: A00, A01
        U[0][base_rank + 4] += 1
        U[1][base_rank + 4] += 1
        # V: B11
        V[1 * p + 2*b + 1][base_rank + 4] += 1
        # W: -C00 + C01
        W[2*b * n + 0][base_rank + 4] -= 1
        W[2*b + 1 * n + 0][base_rank + 4] += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # U: A10, -A00
        U[2][base_rank + 5] += 1
        U[0][base_rank + 5] -= 1
        # V: B00, B01
        V[0 * p + 2*b][base_rank + 5] += 1
        V[0 * p + 2*b + 1][base_rank + 5] += 1
        # W: C11
        W[2*b + 1 * n + 1][base_rank + 5] += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # U: A01, -A11
        U[1][base_rank + 6] += 1
        U[3][base_rank + 6] -= 1
        # V: B10, B11
        V[1 * p + 2*b][base_rank + 6] += 1
        V[1 * p + 2*b + 1][base_rank + 6] += 1
        # W: C00
        W[2*b * n + 0][base_rank + 6] += 1

    return {"U": U, "V": V, "W": W}
