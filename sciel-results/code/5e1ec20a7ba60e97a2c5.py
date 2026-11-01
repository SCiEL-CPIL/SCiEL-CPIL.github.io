import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # The problem is equivalent to 4 independent 2x2 matrix multiplications:
    # A (2x2) * B (2x16) -> C (2x16)
    # We can view B as 4 blocks of 2x2 matrices: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # Wait, B is 2x16. A is 2x2.
    # C = A * B.
    # C[i, j] = sum_k A[i, k] B[k, j].
    # Since A is 2x2, C[i, j] = A[i, 0] B[0, j] + A[i, 1] B[1, j].
    # This is just a linear combination of the rows of B.
    # This is NOT 4 parallel 2x2 multiplications in the sense of A being 4 blocks.
    # A is 2x2. B is 2x16.
    # Let's re-read the task: "multiplying a 2 by 2 matrix by a 2 by 16 matrix".
    # A: 2x2, B: 2x16, C: 2x16.
    # This is a "fat" matrix multiplication.
    # Can we use Strassen?
    # Strassen applies to 2x2 * 2x2.
    # We can view B as a block matrix with 4 blocks of size 2x2? No, B is 2x16.
    # We can view B as 4 columns of 2x4?
    # Let's partition B into 4 blocks of size 2x2? No, 16/2 = 8 blocks of 2x2?
    # B is 2x16. We can partition B into 8 blocks of size 2x2?
    # B = [B0, B1, B2, B3, B4, B5, B6, B7] where each Bi is 2x2.
    # Then C = A * B = [A*B0, A*B1, ..., A*B7].
    # Each A*Bi is a 2x2 matrix multiplication.
    # If we use Strassen for each, rank is 7 per block.
    # Total rank = 8 * 7 = 56.
    # Current rank is 64.
    # 56 < 64. This is an improvement.
    
    # Let's verify the decomposition.
    # A = [[a, b], [c, d]]
    # Bi = [[e, f], [g, h]]
    # C_i = A * Bi = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M5 + M3
    # C12 = M4 + M5
    # C21 = M2 + M7
    # C22 = M1 - M4 + M6
    
    # We have 8 such blocks.
    # Total rank = 8 * 7 = 56.
    
    # Can we do better?
    # Maybe use a larger Strassen variant?
    # Or exploit the linearity?
    # The rank of the tensor for 2x2 * 2x16 is at most 56.
    # Is it lower?
    # The lower bound for 2x2 * 2x16 is not well known, but 56 is a valid upper bound.
    
    # Let's implement the 56-rank decomposition.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 4 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Indices:
    # U index: i * m + k (i in 0..1, k in 0..1) -> 4 entries
    # V index: k * p + j (k in 0..1, j in 0..15) -> 32 entries
    # W index: j * n + i (j in 0..15, i in 0..1) -> 32 entries (Trace convention W[k*n+i] where k is output col? 
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W is p*n. Here p=16, n=2. So W is 32x rank.
    # Index in W: col * n + row.
    # C is 2x16. C[i, j].
    # So W index for C[i, j] is j * 2 + i.
    
    # Let's map the 8 blocks.
    # Block b (0..7) corresponds to columns 2*b and 2*b+1 of B and C.
    # For each block b:
    #   Input A: A[0,0], A[0,1], A[1,0], A[1,1]
    #   Input B_b: B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
    #   Output C_b: C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1]
    
    # Strassen terms for block b:
    # M1_b = (A00 + A11) * (B00_b + B11_b)
    # M2_b = (A10 + A11) * B00_b
    # M3_b = A00 * (B01_b - B11_b)
    # M4_b = A11 * (B10_b - B00_b)
    # M5_b = (A00 + A01) * B11_b
    # M6_b = (A10 - A00) * (B00_b + B01_b)
    # M7_b = (A01 - A11) * (B10_b + B11_b)
    
    # Output:
    # C00_b = M1_b - M5_b + M3_b
    # C01_b = M4_b + M5_b
    # C10_b = M2_b + M7_b
    # C11_b = M1_b - M4_b + M6_b
    
    # We need to fill U, V, W for each term.
    # Term t (0..6) for block b (0..7).
    # Total terms = 8 * 7 = 56.
    
    # Mapping A indices to U:
    # A00 -> U[0]
    # A01 -> U[1]
    # A10 -> U[2]
    # A11 -> U[3]
    
    # Mapping B indices to V:
    # B00_b -> V[0 * 16 + 2b] = V[2b]
    # B01_b -> V[0 * 16 + 2b+1] = V[2b+1]
    # B10_b -> V[1 * 16 + 2b] = V[16 + 2b]
    # B11_b -> V[1 * 16 + 2b+1] = V[16 + 2b+1]
    
    # Mapping C indices to W:
    # C00_b -> W[2b * 2 + 0] = W[4b]
    # C01_b -> W[2b+1 * 2 + 0] = W[4b+2]
    # C10_b -> W[2b * 2 + 1] = W[4b+1]
    # C11_b -> W[2b+1 * 2 + 1] = W[4b+3]
    
    # Let's define the Strassen terms explicitly.
    # Each term is a product of a linear form in A and a linear form in B_b.
    # U stores coefficients for A.
    # V stores coefficients for B_b.
    # W stores coefficients for C_b (with signs).
    
    # Term 1: (A00 + A11) * (B00_b + B11_b)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1] (for B00_b, B01_b, B10_b, B11_b) -> Actually V is global.
    # We need to set V indices for B00_b and B11_b to 1.
    # W: C00_b (+1), C11_b (-1)
    
    # Term 2: (A10 + A11) * B00_b
    # U: [0, 0, 1, 1]
    # V: B00_b = 1
    # W: C10_b (+1)
    
    # Term 3: A00 * (B01_b - B11_b)
    # U: [1, 0, 0, 0]
    # V: B01_b = 1, B11_b = -1
    # W: C00_b (+1)
    
    # Term 4: A11 * (B10_b - B00_b)
    # U: [0, 0, 0, 1]
    # V: B10_b = 1, B00_b = -1
    # W: C01_b (+1), C11_b (-1)
    
    # Term 5: (A00 + A01) * B11_b
    # U: [1, 1, 0, 0]
    # V: B11_b = 1
    # W: C00_b (-1), C01_b (+1)
    
    # Term 6: (A10 - A00) * (B00_b + B01_b)
    # U: [-1, 0, 1, 0]
    # V: B00_b = 1, B01_b = 1
    # W: C11_b (+1)
    
    # Term 7: (A01 - A11) * (B10_b + B11_b)
    # U: [0, 1, 0, -1]
    # V: B10_b = 1, B11_b = 1
    # W: C10_b (+1)
    
    # Let's code this.
    
    for b in range(8):
        base_col = b * 7
        # B indices
        b00_idx = 2 * b
        b01_idx = 2 * b + 1
        b10_idx = 16 + 2 * b
        b11_idx = 16 + 2 * b + 1
        
        # W indices
        c00_idx = 4 * b
        c01_idx = 4 * b + 2
        c10_idx = 4 * b + 1
        c11_idx = 4 * b + 3
        
        # Term 1
        col = base_col + 0
        U[0][col] += 1
        U[3][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] -= 1
        
        # Term 2
        col = base_col + 1
        U[2][col] += 1
        U[3][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        
        # Term 3
        col = base_col + 2
        U[0][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c00_idx][col] += 1
        
        # Term 4
        col = base_col + 3
        U[3][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] -= 1
        
        # Term 5
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        
        # Term 6
        col = base_col + 5
        U[0][col] -= 1
        U[2][col] += 1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c11_idx][col] += 1
        
        # Term 7
        col = base_col + 6
        U[1][col] += 1
        U[3][col] -= 1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c10_idx][col] += 1

    return {"U": U, "V": V, "W": W}
