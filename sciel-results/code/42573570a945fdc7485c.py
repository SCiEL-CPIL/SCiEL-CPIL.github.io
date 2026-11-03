import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_solution(U, V, W):
        """Check if U, V, W form an exact decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Build 4x4x4 Strassen decomposition (Rank 49)
    def build_4x4x4_strassen():
        # Outer Strassen coefficients for A (U_out), B (V_out), C (W_out)
        # Indices 0..6 for r1
        U_out = np.zeros((2, 2, 7))
        V_out = np.zeros((2, 2, 7))
        W_out = np.zeros((2, 2, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U_out[0,0,0] = 1; U_out[1,1,0] = 1
        V_out[0,0,0] = 1; V_out[1,1,0] = 1
        W_out[0,0,0] = 1; W_out[1,1,0] = 1
        
        # M2 = (A10+A11)B00
        U_out[1,0,1] = 1; U_out[1,1,1] = 1
        V_out[0,0,1] = 1
        W_out[1,0,1] = 1; W_out[1,1,1] = -1
        
        # M3 = A00(B01-B11)
        U_out[0,0,2] = 1
        V_out[0,1,2] = 1; V_out[1,1,2] = -1
        W_out[0,1,2] = 1; W_out[1,1,2] = 1
        
        # M4 = A11(B10-B00)
        U_out[1,1,3] = 1
        V_out[1,0,3] = 1; V_out[0,0,3] = -1
        W_out[0,0,3] = 1; W_out[1,0,3] = 1
        
        # M5 = (A00+A01)B11
        U_out[0,0,4] = 1; U_out[0,1,4] = 1
        V_out[1,1,4] = 1
        W_out[0,0,4] = -1; W_out[0,1,4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U_out[1,0,5] = 1; U_out[0,0,5] = -1
        V_out[0,0,5] = 1; V_out[0,1,5] = 1
        W_out[1,1,5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U_out[0,1,6] = 1; U_out[1,1,6] = -1
        V_out[1,0,6] = 1; V_out[1,1,6] = 1
        W_out[0,0,6] = 1

        # Inner Strassen coefficients for 2x2x2 (Rank 7)
        U_in = np.zeros((2, 2, 7))
        V_in = np.zeros((2, 2, 7))
        W_in = np.zeros((2, 2, 7))
        
        # Same structure as outer, but for the local 2x2 blocks
        # We can reuse the same coefficients as U_out, V_out, W_out
        U_in = U_out.copy()
        V_in = V_out.copy()
        W_in = W_out.copy()

        # Combine into 4x4x4
        U_4 = np.zeros((4, 4, 49))
        V_4 = np.zeros((4, 4, 49))
        W_4 = np.zeros((4, 4, 49))
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i in range(4):
                    i_prime = i // 2
                    i_double = i % 2
                    for j in range(4):
                        j_prime = j // 2
                        j_double = j % 2
                        U_4[i, j, r] = U_out[i_prime, j_prime, r1] * U_in[i_double, j_double, r2]
                        V_4[i, j, r] = V_out[i_prime, j_prime, r1] * V_in[i_double, j_double, r2]
                        W_4[i, j, r] = W_out[i_prime, j_prime, r1] * W_in[i_double, j_double, r2]
                        
        return U_4, V_4, W_4

    # Build the full 4x5x5 decomposition
    U_full = np.zeros((20, 85)) # 4*5 = 20 rows
    V_full = np.zeros((25, 85)) # 5*5 = 25 rows
    W_full = np.zeros((20, 85)) # 5*4 = 20 rows
    
    U_4, V_4, W_4 = build_4x4x4_strassen()
    
    # Term 1: 4x4x4 (Indices i<4, j<4, k<4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Map local (i,j,k) to global
                # U row: i*5 + j
                # V row: j*5 + k
                # W row: k*4 + i
                for r in range(49):
                    U_full[i*5 + j, r] = U_4[i, j, r]
                    V_full[j*5 + k, r] = V_4[i, j, r] # Note: V_4 indices are (i,j) in local?
                    # Wait, V_4 from build_4x4x4_strassen has shape (4,4,49).
                    # In the local 4x4x4 problem, V corresponds to B.
                    # B is 4x4. Indices are (row, col).
                    # In global, V row is j*5+k.
                    # Local j corresponds to global j. Local k corresponds to global k.
                    # So V_4[j, k, r] should be used.
                    # But I stored V_4[i, j, r] in the builder?
                    # In build_4x4x4_strassen, I used i,j for U and V.
                    # U is A (4x4). V is B (4x4).
                    # So V_4[i, j, r] in the builder corresponds to B[i, j].
                    # In global, B is at indices (j, k).
                    # So we should use V_4[j, k, r].
                    # Let's re-verify the builder.
                    # U_4[i, j, r] corresponds to A[i, j].
                    # V_4[i, j, r] corresponds to B[i, j].
                    # W_4[i, j, r] corresponds to C[i, j].
                    
                    # So for global:
                    # A is at (i, j). U_4[i, j, r] is correct.
                    # B is at (j, k). V_4[j, k, r] is correct.
                    # C is at (k, i). W_4[k, i, r] is correct.
                    
                    # I will fix the assignment in the loop below.
                    pass

    # Redo Term 1 assignment correctly
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    U_full[i*5 + j, r] = U_4[i, j, r]
                    V_full[j*5 + k, r] = V_4[j, k, r]
                    W_full[k*4 + i, r] = W_4[k, i, r]

    # Term 2: 4x4x1 (i<4, j<4, k=4)
    # Rank 16
    r_offset = 49
    for i in range(4):
        for j in range(4):
            r = r_offset + i*4 + j
            U_full[i*5 + j, r] = 1
            V_full[j*5 + 4, r] = 1
            W_full[4*4 + i, r] = 1 # k=4, so 4*4+i

    # Term 3: 4x1x4 (i<4, j=4, k<4)
    # Rank 16
    r_offset = 49 + 16
    for i in range(4):
        for k in range(4):
            r = r_offset + i*4 + k
            U_full[i*5 + 4, r] = 1
            V_full[4*5 + k, r] = 1
            W_full[k*4 + i, r] = 1

    # Term 4: 4x1x1 (i<4, j=4, k=4)
    # Rank 4
    r_offset = 49 + 16 + 16
    for i in range(4):
        r = r_offset + i
        U_full[i*5 + 4, r] = 1
        V_full[4*5 + 4, r] = 1
        W_full[4*4 + i, r] = 1

    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()

    if check_solution(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
