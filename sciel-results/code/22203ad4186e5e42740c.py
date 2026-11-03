import numpy as np

def construct():
    # Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # W index: k*2 + i
    
    # U factors A (2x2 -> 4 entries)
    # V factors B (2x2 -> 4 entries)
    # W factors C (2x2 -> 4 entries)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_strassen = np.array([
        [1, 0, 0, 0, 0, 0, 0], # A00
        [0, 0, 1, 0, 1, -1, 0], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [1, 0, -1, 1, 1, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 0, -1, 0, 1], # C00 (k=0, i=0) -> index 0
        [0, 0, 1, 0, 1, 0, 0], # C01 (k=1, i=0) -> index 1
        [0, 1, 0, 1, 0, 0, 0], # C10 (k=0, i=1) -> index 2
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1) -> index 3
    ], dtype=float)
    
    # 2x2x3 decomposition
    # A is 2x2, B is 2x3, C is 2x3
    # W index: k*2 + i (k=0,1,2)
    
    # Use Strassen for first 2 cols (rank 7)
    # Use Schoolbook for 3rd col (rank 4)
    # Total rank 11
    
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))
    W_223 = np.zeros((6, 11))
    
    # First 7 cols: Strassen on B[:, 0:2]
    # B indices: 0,1 (col 0), 2,3 (col 1), 4,5 (col 2)
    # Strassen V expects B00, B01, B10, B11
    # Map: B00->V[0], B01->V[1], B10->V[2], B11->V[3]
    U_223[:, :7] = U_strassen
    V_223[0:4, :7] = V_strassen
    # C indices: C00->0, C01->1, C10->2, C11->3
    # Map to W indices:
    # C00 (i=0,k=0) -> idx 0
    # C01 (i=0,k=1) -> idx 1
    # C10 (i=1,k=0) -> idx 2
    # C11 (i=1,k=1) -> idx 3
    W_223[0:4, :7] = W_strassen
    
    # Next 4 cols: Schoolbook for 3rd col (k=2)
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    # B02 is V index 4, B12 is V index 5
    # C02 is W index 4 (k=2, i=0)
    # C12 is W index 5 (k=2, i=1)
    
    # Term 8: A00 * B02
    U_223[0, 7] = 1
    V_223[4, 7] = 1
    W_223[4, 7] = 1
    
    # Term 9: A01 * B12
    U_223[1, 8] = 1
    V_223[5, 8] = 1
    W_223[4, 8] = 1
    
    # Term 10: A10 * B02
    U_223[2, 9] = 1
    V_223[4, 9] = 1
    W_223[5, 9] = 1
    
    # Term 11: A11 * B12
    U_223[3, 10] = 1
    V_223[5, 10] = 1
    W_223[5, 10] = 1
    
    # Kronecker Product
    # U = kron(U_223, U_strassen)
    # V = kron(V_223, V_strassen)
    # W = kron(W_223, W_strassen)
    
    # Note: numpy kron flattens in row-major order.
    # This matches the index ordering if the sub-tensors are ordered consistently.
    
    U_final = np.kron(U_223, U_strassen)
    V_final = np.kron(V_223, V_strassen)
    W_final = np.kron(W_223, W_strassen)
    
    # Convert to list of lists of integers
    # Entries are integers or half-integers? Strassen uses integers.
    # Convert to int if possible
    
    U_list = U_final.astype(int).tolist()
    V_list = V_final.astype(int).tolist()
    W_list = W_final.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
