import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # Rank 44: 4 * 7 (Strassen for 2x4x4) + 16 (Schoolbook for 1x4x4)
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Indices:
        # A: rows i0..i0+1, cols j0..j0+1
        # B: rows j0..j0+1, cols k0..k0+1
        # C: rows i0..i0+1, cols k0..k0+1
        
        # U index: (i) * m + (j)
        # V index: (j) * p + (k)
        # W index: (k) * n + (i)
        
        def add_term(u_indices, v_indices, w_indices, val):
            nonlocal col
            for ui in u_indices:
                U[ui, col] = val
            for vi in v_indices:
                V[vi, col] = val
            for wi in w_indices:
                W[wi, col] = val
            col += 1

        # M1 = (A11 + A22)(B11 + B22)
        u_m1 = [(i0)*m + j0, (i0+1)*m + (j0+1)]
        v_m1 = [(j0)*p + k0, (j0+1)*p + (k0+1)]
        w_m1 = [(k0)*n + i0, (k0+1)*n + (i0+1)]
        add_term(u_m1, v_m1, w_m1, 1)
        
        # M2 = (A21 + A22)B11
        u_m2 = [(i0+1)*m + j0, (i0+1)*m + (j0+1)]
        v_m2 = [(j0)*p + k0]
        w_m2 = [(k0)*n + (i0+1), (k0+1)*n + (i0+1)]
        # Correction: M2 contributes to C21 and C22?
        # C21 = A21 B11 + A22 B21.
        # M2 = (A21+A22)B11 = A21 B11 + A22 B11.
        # We need C21 and C22.
        # C21 gets M2 - M?
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B11)
        # P4 = A22(B22-B12)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Let's stick to the explicit additions/subtractions in W.
        # But here we add rank-1 terms.
        # Each Pk is one rank-1 term.
        
        # We need to map Pk to U, V, W.
        
        # P1: U: A11, A22. V: B11, B22. W: C11, C22 (with signs?)
        # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # C11 needs A11B11 + A22B22 - ...
        # C22 needs A11B11 + A22B22 - ...
        
        # My previous helper in the seed was adding terms directly to C elements.
        # Let's reuse the logic from the seed which worked (Score 1.0909).
        
    # Re-implementing the seed's add_strassen_222 logic explicitly to be safe and clear.
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        
        # Indices helper
        def u(i, j): return (i_off+i)*m + (j_off+j)
        def v(j, k): return (j_off+j)*p + (k_off+k)
        def w(k, i): return (k_off+k)*n + (i_off+i)

        # P1: (A00+A11)(B00+B11)
        U[u(0,0), col] = 1
        U[u(1,1), col] = 1
        V[v(0,0), col] = 1
        V[v(1,1), col] = 1
        W[w(0,0), col] = 1 # C00
        W[w(1,1), col] = 1 # C11
        col += 1

        # P2: (A10+A11)B00
        U[u(1,0), col] = 1
        U[u(1,1), col] = 1
        V[v(0,0), col] = 1
        W[w(0,1), col] = 1 # C10? No.
        # C10 = A10 B00 + A11 B10.
        # P2 = A10 B00 + A11 B00.
        # We need C10 and C11.
        # Strassen formula:
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Let's map the P's to the C's correctly.
        # P1 contributes to C00 and C11?
        # P1 = (A00+A11)(B00+B11)
        # C00 = P1 + P4 - P5 + P7
        # C11 = P1 - P2 + P3 + P6
        # So W for P1 should have 1 at C00 and 1 at C11.
        
        # Wait, my previous code had:
        # W[w(0,0), col] = 1
        # W[w(1,1), col] = 1
        # This matches.
        
        # P2: (A10+A11)B00
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        # So P2 contributes +1 to C10 and -1 to C11.
        U[u(1,0), col] = 1
        U[u(1,1), col] = 1
        V[v(0,0), col] = 1
        W[w(0,1), col] = 1 # C10 (k=0, i=1)
        W[w(1,1), col] = -1 # C11 (k=1, i=1)
        col += 1

        # P3: A00(B10-B00)
        # C01 = P3 + P5
        # C11 = P1 - P2 + P3 + P6
        # So P3 contributes +1 to C01 and +1 to C11.
        U[u(0,0), col] = 1
        V[v(1,0), col] = 1
        V[v(0,0), col] = -1
        W[w(0,0), col] = 1 # C01 (k=0, i=0)? No, C01 is k=1?
        # W index: k*n + i.
        # C01: i=0, k=1. Index = 1*3 + 0 = 3.
        # w(1, 0) = (k_off+1)*n + i_off.
        # w(0, 1) = (k_off+0)*n + i_off+1.
        # C00: i=0, k=0. Index 0.
        # C01: i=0, k=1. Index 3.
        # C10: i=1, k=0. Index 1.
        # C11: i=1, k=1. Index 4.
        
        # Check w function: w(k, i) = (k_off+k)*n + (i_off+i).
        # C00 -> w(0,0)
        # C01 -> w(1,0)
        # C10 -> w(0,1)
        # C11 -> w(1,1)
        
        # Redo P3:
        # Contributes to C01 (+1) and C11 (+1).
        U[u(0,0), col] = 1
        V[v(1,0), col] = 1
        V[v(0,0), col] = -1
        W[w(1,0), col] = 1 # C01
        W[w(1,1), col] = 1 # C11
        col += 1

        # P4: A11(B11-B01)
        # C00 = P1 + P4 - P5 + P7
        # C10 = P2 + P4
        # Contributes to C00 (+1) and C10 (+1).
        U[u(1,1), col] = 1
        V[v(1,1), col] = 1
        V[v(0,1), col] = -1
        W[w(0,0), col] = 1 # C00
        W[w(0,1), col] = 1 # C10
        col += 1

        # P5: (A00+A01)B11
        # C01 = P3 + P5
        # C00 = P1 + P4 - P5 + P7
        # Contributes to C01 (+1) and C00 (-1).
        U[u(0,0), col] = 1
        U[u(0,1), col] = 1
        V[v(1,1), col] = 1
        W[w(1,0), col] = 1 # C01
        W[w(0,0), col] = -1 # C00
        col += 1

        # P6: (A10-A00)(B00+B01)
        # C11 = P1 - P2 + P3 + P6
        # Contributes to C11 (+1).
        # Note: P6 only contributes to C11?
        # C11 = P1 - P2 + P3 + P6. Yes.
        # Wait, does it contribute elsewhere?
        # C01 = P3 + P5. No.
        # C10 = P2 + P4. No.
        # C00 = P1 + P4 - P5 + P7. No.
        # So only C11.
        U[u(1,0), col] = 1
        U[u(0,0), col] = -1
        V[v(0,0), col] = 1
        V[v(0,1), col] = 1
        W[w(1,1), col] = 1 # C11
        col += 1

        # P7: (A01-A11)(B10+B11)
        # C00 = P1 + P4 - P5 + P7
        # Contributes to C00 (+1).
        U[u(0,1), col] = 1
        U[u(1,1), col] = -1
        V[v(1,0), col] = 1
        V[v(1,1), col] = 1
        W[w(0,0), col] = 1 # C00
        col += 1

    # Apply Strassen to the 2x4x4 block (rows 0-1 of A)
    # Split B into 2x2 blocks.
    add_strassen_222(0, 0, 0) # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_222(0, 0, 2) # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_222(0, 2, 0) # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_222(0, 2, 2) # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # Apply Schoolbook to the 1x4x4 block (row 2 of A)
    # C[2, j] = sum_k A[2, k] B[k, j]
    for j in range(4):
        for k in range(4):
            U[2*m + k, col] = 1 # A[2, k]
            V[k*p + j, col] = 1 # B[k, j]
            W[j*n + 2, col] = 1 # C[2, j]
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
