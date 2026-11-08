import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = 2, 3, 3
    num_outputs = p * n
    num_inputs_a = n * m
    num_inputs_b = m * p

    def make_tensor(U, V, W, rank):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def target_tensor():
        T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ia = i * m + j
                    ib = j * p + k
                    out_idx = k * n + i
                    T[ia, ib, out_idx] = 1.0
        return T

    target = target_tensor()

    # Rank 17 construction
    rank = 17
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))

    # Indices:
    # A (2x3): 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B (3x3): 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C (2x3): 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2)
    # Note: W index is k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5

    # Strassen for top-left 2x2 block (j in {0,1}, k in {0,1})
    # A indices: 0,1 (row 0, cols 0,1) and 3,4 (row 1, cols 0,1)
    # B indices: 0,1 (row 0, cols 0,1) and 3,4 (row 1, cols 0,1)
    # C indices: 0,1 (col 0, rows 0,1) and 2,3 (col 1, rows 0,1)

    # M1 = (A00+A11)(B00+B11)
    # A: 0, 4. B: 0, 4. C: 0,1,2,3 (with signs)
    # Standard Strassen:
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A10) B11
    # M6 = (A11 - A00) (B00 + B10)
    # M7 = (A01 - A11) (B11 + B10)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6

    # Term 0: M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11

    # Term 1: M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10, C11

    # Term 2: M3
    U[0, 2] = 1
    V[3, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C01, C11

    # Term 3: M4
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1 # C00, C10

    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00, C01

    # Term 5: M6
    U[4, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1 # C11

    # Term 6: M7
    U[1, 6] = 1; U[4, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[0, 6] = -1 # C01, C00

    # Schoolbook for remaining parts
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C00..C11 already handled by Strassen?
    # Strassen handled C00, C01, C10, C11.
    # We need C02 and C12.
    # C02 involves B col 2.
    # C12 involves B col 2.
    
    # Term 7: A00 * B02 -> C02
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    
    # Term 8: A01 * B12 -> C02
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    
    # Term 9: A10 * B02 -> C12
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    
    # Term 10: A11 * B12 -> C12
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    
    # C00, C10 also have contributions from A02, A12?
    # No, C00 = sum_j A0j Bj0.
    # Strassen used B00, B10.
    # We missed A02*B20 and A12*B20.
    # And C01, C11 missed A02*B21 and A12*B21.
    
    # Term 11: A02 * B20 -> C00
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    
    # Term 12: A02 * B21 -> C01
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    
    # Term 13: A12 * B20 -> C10
    U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
    
    # Term 14: A12 * B21 -> C11
    U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1
    
    # Term 15: A02 * B22 -> C02
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    
    # Term 16: A12 * B22 -> C12
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    computed = make_tensor(U, V, W, rank)
    max_err = np.max(np.abs(computed - target))

    if max_err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 18 (schoolbook)
    rank = 18
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U[ia, r] = 1
                V[ib, r] = 1
                W[out_idx, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
