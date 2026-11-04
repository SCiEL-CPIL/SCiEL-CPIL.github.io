import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We'll use a block-Strassen approach.
    # The matrix A is 2x4, B is 4x8, C is 2x8.
    # We can view B as 4x4 blocks of size 1x2 (since 8 = 4*2).
    # Actually, let's partition A into 2x2 blocks: A = [A00, A01] where A00, A01 are 2x2.
    # And B into 2x4 blocks: B = [B0; B1] where B0, B1 are 2x4? No, B is 4x8.
    # Let's partition B into 4 blocks of 1x2: B0, B1, B2, B3 each 1x2? No.
    # Better: Partition B into 2x2 blocks of size 2x4? No, 4x8 = 2*2 x 2*4.
    # Let's try: A is 2x4. B is 4x8.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_top (2x8) and B_bottom (2x8).
    # Then C = A_left * B_top + A_right * B_bottom.
    # This is two 2x2 * 2x8 multiplications.
    # Each 2x2 * 2x8 can be done with Strassen-like methods?
    # Or simply, we can treat this as 8 independent 2x2 * 1x2 multiplications? No.
    
    # Let's stick to the proven 56-rank Strassen-based solution from the seed.
    # The seed achieved 56. Can we do better?
    # Theoretical lower bound for 2x4x8?
    # Schoolbook: 2*4*8 = 64.
    # Strassen on 2x2 blocks:
    # If we split A into 2x2 and 2x2, and B into 2x4 and 2x4 (vertical split).
    # C = A0 * B0 + A1 * B1.
    # A0 is 2x2, B0 is 2x4.
    # We can split B0 into two 2x2 blocks: B00, B01.
    # So C = A0 * [B00, B01] + A1 * [B10, B11].
    # This is 4 separate 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 can be done in rank 7 (Strassen).
    # Total rank = 4 * 7 = 28?
    # Wait, Strassen for 2x2 * 2x2 has rank 7.
    # But the factors are not independent?
    # Let's check the structure.
    # C00 = A00 B00 + A10 B10
    # C01 = A00 B01 + A10 B11
    # C02 = A01 B00 + A11 B10
    # C03 = A01 B01 + A11 B11
    # ... and so on for row 1 of C.
    # This is actually 8 separate 2x2 * 2x2 multiplications? No.
    # A is 2x4. Let's index A[i, j] for i=0,1 and j=0,1,2,3.
    # B is 4x8. Let's index B[j, k] for j=0,1,2,3 and k=0..7.
    # C[i, k] = sum_j A[i, j] B[j, k].
    #
    # Let's group j into {0,1} and {2,3}.
    # C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[1, k] + A[i, 2]B[2, k] + A[i, 3]B[3, k].
    #
    # Let's group k into {0,1}, {2,3}, {4,5}, {6,7}.
    # For each group of k, we have a 2x2 * 4x2 multiplication?
    # No, for a fixed k-group (2 columns), we have:
    # C[i, k0..k1] = sum_j A[i, j] B[j, k0..k1].
    # This is a 2x4 matrix times a 4x2 matrix, resulting in 2x2.
    # There are 4 such groups (k0=0,2,4,6).
    # So we have 4 independent 2x4 * 4x2 multiplications.
    #
    # What is the rank of 2x4 * 4x2?
    # Schoolbook: 2*4*2 = 16.
    # Can we do better?
    # We can use Strassen-like techniques.
    # Let A be 2x4, B be 4x2.
    # Split A into A0 (2x2) and A1 (2x2).
    # Split B into B0 (2x2) and B1 (2x2).
    # C = A0 B0 + A1 B1.
    # This is sum of two 2x2 * 2x2 products.
    # Each 2x2 * 2x2 has rank 7.
    # So C = A0 B0 + A1 B1 can be done with 7 + 7 = 14 terms?
    # Wait, the factors are shared? No, A0, A1, B0, B1 are distinct.
    # So rank is at most 14.
    #
    # Total rank for 4 groups: 4 * 14 = 56.
    # This matches the seed's 56.
    #
    # Can we do better than 14 for 2x4 * 4x2?
    # The tensor for 2x4 * 4x2 is equivalent to the tensor for 2x2 * 2x2 + 2x2 * 2x2.
    # Is there a more efficient decomposition?
    # Laderman's algorithm for 3x3 * 3x3 has rank 19.
    # For 2x2 * 2x2, Strassen is 7.
    # For 2x4 * 4x2, we can view it as multiplying two 2x2 block matrices.
    # A = [A00 A01], B = [B00; B10].
    # C = A00 B00 + A01 B10.
    # This is exactly the sum of two 2x2 * 2x2 products.
    # The rank of the sum of two independent rank-7 tensors is 14.
    # It is unlikely to be less than 14 without some clever sharing, but the blocks are independent.
    #
    # So 56 seems to be a strong candidate for the minimal rank.
    #
    # Let's implement the 56-rank decomposition.
    # We have 4 groups of k: k0 in {0, 2, 4, 6}.
    # For each k0, we do a 2x4 * 4x2 multiplication with rank 14.
    #
    # For a single 2x4 * 4x2:
    # A is 2x4: indices (i, j) with i in {0,1}, j in {0,1,2,3}.
    # B is 4x2: indices (j, k) with j in {0,1,2,3}, k in {0,1}.
    # C is 2x2: indices (i, k) with i in {0,1}, k in {0,1}.
    #
    # Split A into A0 (j=0,1) and A1 (j=2,3).
    # Split B into B0 (j=0,1) and B1 (j=2,3).
    # C = A0 B0 + A1 B1.
    #
    # Let's use Strassen for each 2x2 * 2x2.
    # For A0 B0:
    # M1 = (A00 + A01)(B00 + B01)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B01)  # Wait, standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We need to apply this to A0 B0 and A1 B1 separately.
    #
    # Let's define the terms for A0 B0 (where A0 is rows 0,1 and cols 0,1 of A; B0 is rows 0,1 and cols 0,1 of B).
    # And for A1 B1 (where A1 is rows 0,1 and cols 2,3 of A; B1 is rows 2,3 and cols 0,1 of B).
    #
    # Total rank = 7 + 7 = 14 per k-group.
    # Total rank = 4 * 14 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    col = 0
    
    for k0 in [0, 2, 4, 6]:
        # k0 is the start of the 2-column block in B and C
        # Local k indices: 0, 1 correspond to global k0, k0+1
        
        # --- Part 1: A0 * B0 ---
        # A0: i in {0,1}, j in {0,1}
        # B0: j in {0,1}, k in {0,1} (global k0, k0+1)
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: (0,0), A11: (1,1)
        # B00: (0,k0), B11: (1,k0+1)
        add_term(
            {(0, 0): 1, (1, 1): 1},
            {(0, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M2 = (A10 + A11)B00
        # A10: (1,0), A11: (1,1)
        # B00: (0,k0)
        add_term(
            {(1, 0): 1, (1, 1): 1},
            {(0, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        
        # M3 = A00(B01 - B11)
        # A00: (0,0)
        # B01: (0,k0+1), B11: (1,k0+1)
        add_term(
            {(0, 0): 1},
            {(0, k0+1): 1, (1, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M4 = A11(B10 - B00)
        # A11: (1,1)
        # B10: (1,k0), B00: (0,k0)
        add_term(
            {(1, 1): 1},
            {(1, k0): 1, (0, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        
        # M5 = (A00 + A01)B11
        # A00: (0,0), A01: (0,1)
        # B11: (1,k0+1)
        add_term(
            {(0, 0): 1, (0, 1): 1},
            {(1, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10: (1,0), A00: (0,0)
        # B00: (0,k0), B01: (0,k0+1)
        add_term(
            {(1, 0): 1, (0, 0): -1},
            {(0, k0): 1, (0, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01: (0,1), A11: (1,1)
        # B10: (1,k0), B11: (1,k0+1)
        add_term(
            {(0, 1): 1, (1, 1): -1},
            {(1, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1
        
        # --- Part 2: A1 * B1 ---
        # A1: i in {0,1}, j in {2,3}
        # B1: j in {2,3}, k in {0,1} (global k0, k0+1)
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: (0,2), A11: (1,3)
        # B00: (2,k0), B11: (3,k0+1)
        add_term(
            {(0, 2): 1, (1, 3): 1},
            {(2, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M2 = (A10 + A11)B00
        # A10: (1,2), A11: (1,3)
        # B00: (2,k0)
        add_term(
            {(1, 2): 1, (1, 3): 1},
            {(2, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        
        # M3 = A00(B01 - B11)
        # A00: (0,2)
        # B01: (2,k0+1), B11: (3,k0+1)
        add_term(
            {(0, 2): 1},
            {(2, k0+1): 1, (3, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M4 = A11(B10 - B00)
        # A11: (1,3)
        # B10: (3,k0), B00: (2,k0)
        add_term(
            {(1, 3): 1},
            {(3, k0): 1, (2, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        
        # M5 = (A00 + A01)B11
        # A00: (0,2), A01: (0,3)
        # B11: (3,k0+1)
        add_term(
            {(0, 2): 1, (0, 3): 1},
            {(3, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10: (1,2), A00: (0,2)
        # B00: (2,k0), B01: (2,k0+1)
        add_term(
            {(1, 2): 1, (0, 2): -1},
            {(2, k0): 1, (2, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01: (0,3), A11: (1,3)
        # B10: (3,k0), B11: (3,k0+1)
        add_term(
            {(0, 3): 1, (1, 3): -1},
            {(3, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1
    
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }

def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
    # This function is defined inside construct in the original, 
    # but I'll define it globally for clarity, passing U, V, W?
    # No, the original defines it inside. I should stick to that or make it global.
    # Let's make it global and pass U, V, W.
    pass

# Wait, I need to fix the add_term function. It needs access to U, V, W.
# I'll redefine construct to include the helper.

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col_idx] += val

    col = 0
    
    for k0 in [0, 2, 4, 6]:
        # --- Part 1: A0 * B0 ---
        add_term(
            {(0, 0): 1, (1, 1): 1},
            {(0, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 0): 1, (1, 1): 1},
            {(0, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        add_term(
            {(0, 0): 1},
            {(0, k0+1): 1, (1, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 1): 1},
            {(1, k0): 1, (0, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 0): 1, (0, 1): 1},
            {(1, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        add_term(
            {(1, 0): 1, (0, 0): -1},
            {(0, k0): 1, (0, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 1): 1, (1, 1): -1},
            {(1, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1
        
        # --- Part 2: A1 * B1 ---
        add_term(
            {(0, 2): 1, (1, 3): 1},
            {(2, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 2): 1, (1, 3): 1},
            {(2, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        add_term(
            {(0, 2): 1},
            {(2, k0+1): 1, (3, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 3): 1},
            {(3, k0): 1, (2, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 2): 1, (0, 3): 1},
            {(3, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        add_term(
            {(1, 2): 1, (0, 2): -1},
            {(2, k0): 1, (2, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 3): 1, (1, 3): -1},
            {(3, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1
    
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    if not np.allclose(T_recon, T_target, atol=1e-9):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
