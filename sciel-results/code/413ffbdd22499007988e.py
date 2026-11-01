import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Strassen parameters for 2x2x2
    # U, V, W are 7x4 matrices
    # U[i, r] corresponds to A[i]
    # V[j, r] corresponds to B[j]
    # W[k, r] corresponds to C[k]
    # Indices: A is 2x2 (4 elements), B is 2x2 (4 elements), C is 2x2 (4 elements)
    
    # Standard Strassen decomposition
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # C11 = m1 - m4 + m5 + m7
    # C12 = m3 + m4 + m6
    # C21 = m2 + m5 - m1 - m7
    # C22 = m1 - m3 + m2 - m6
    
    # Let's map these to vectors
    # A indices: 0:11, 1:12, 2:21, 3:22
    # B indices: 0:11, 1:12, 2:21, 3:22
    # C indices: 0:11, 1:12, 2:21, 3:22
    
    U_strassen = [
        [1, 0, 0, 1], # m1: a11, a22
        [0, 0, 1, 1], # m2: a21, a22
        [1, 0, 0, 0], # m3: a11
        [0, 0, 0, 1], # m4: a22
        [1, 1, 0, 0], # m5: a11, a12
        [-1, 0, 1, 0], # m6: a21, a11
        [0, 1, 0, -1]  # m7: a12, a22
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # m1: b11, b22
        [1, 0, 0, 0], # m2: b11
        [0, 1, 0, -1], # m3: b12, b22
        [-1, 0, 1, 0], # m4: b21, b11
        [0, 0, 0, 1], # m5: b22
        [1, 1, 0, 0], # m6: b11, b12
        [0, 0, 1, 1]  # m7: b21, b22
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # c11: m1, c12: m7? No.
        # c11 = m1 - m4 + m5 + m7
        # c12 = m3 + m4 + m6
        # c21 = m2 + m5 - m1 - m7
        # c22 = m1 - m3 + m2 - m6
        
        # Row 0 (c11): m1(1), m2(0), m3(0), m4(-1), m5(1), m6(0), m7(1)
        [1, 0, 0, -1, 1, 0, 1],
        # Row 1 (c12): m1(0), m2(0), m3(1), m4(1), m5(0), m6(1), m7(0)
        [0, 0, 1, 1, 0, 1, 0],
        # Row 2 (c21): m1(-1), m2(1), m3(0), m4(0), m5(1), m6(0), m7(-1)
        [-1, 1, 0, 0, 1, 0, -1],
        # Row 3 (c22): m1(1), m2(1), m3(-1), m4(0), m5(0), m6(-1), m7(0)
        [1, 1, -1, 0, 0, -1, 0]
    ]
    
    # Verify Strassen 2x2x2
    def verify_strassen_22():
        U = np.array(U_strassen, dtype=float)
        V = np.array(V_strassen, dtype=float)
        W = np.array(W_strassen, dtype=float)
        
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        C_check = np.zeros(4)
        for r in range(7):
            scalar = np.dot(U[:, r], A_flat) * np.dot(V[:, r], B_flat)
            C_check += W[:, r] * scalar
            
        if not np.allclose(C_check, C_expected.flatten(), atol=1e-10):
            return False
        return True

    if not verify_strassen_22():
        # Fallback to schoolbook if Strassen fails (should not happen)
        pass

    def build_strassen_28():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in range(num_blocks):
            for r in range(7):
                # U: A is 2x2, indices 0..3
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                
                # V: B is 2x8. Block t uses columns 2t, 2t+1.
                # B indices for this block:
                # Row 0, Col 2t -> index 2t
                # Row 0, Col 2t+1 -> index 2t+1
                # Row 1, Col 2t -> index 8 + 2t
                # Row 1, Col 2t+1 -> index 8 + 2t + 1
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                
                # W: C is 2x8. Block t uses columns 2t, 2t+1.
                # C indices for this block:
                # Row 0, Col 2t -> index 2t
                # Row 0, Col 2t+1 -> index 2t+1
                # Row 1, Col 2t -> index 8 + 2t
                # Row 1, Col 2t+1 -> index 8 + 2t + 1
                
                # Wait, the trace convention is W[k*n+i, r].
                # C is n x p. Index in flat array is k*n + i?
                # Usually C[i, k] is stored as i*p + k or k*n + i.
                # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
                # This implies C is indexed by k (col) and i (row).
                # So C[i, k] corresponds to W[k*n + i].
                
                # For block t (cols 2t, 2t+1):
                # k=2t, i=0 -> W[2t*n + 0]
                # k=2t, i=1 -> W[2t*n + 1]
                # k=2t+1, i=0 -> W[(2t+1)*n + 0]
                # k=2t+1, i=1 -> W[(2t+1)*n + 1]
                
                # In the previous code:
                # w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                # This assumed n=2.
                # 2t*n + 0 = 4t
                # 2t*n + 1 = 4t + 1
                # (2t+1)*n + 0 = 4t + 2
                # (2t+1)*n + 1 = 4t + 3
                
                # The previous code mapped:
                # W_strassen[r][0] (c11) -> W[4t]
                # W_strassen[r][1] (c12) -> W[4t+2] ??
                # W_strassen[r][2] (c21) -> W[4t+1] ??
                # W_strassen[r][3] (c22) -> W[4t+3]
                
                # Let's check the mapping in the previous code:
                # w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                # l=0 (c11) -> 4t
                # l=1 (c12) -> 4t+2
                # l=2 (c21) -> 4t+1
                # l=3 (c22) -> 4t+3
                
                # My W_strassen rows:
                # Row 0: c11
                # Row 1: c12
                # Row 2: c21
                # Row 3: c22
                
                # So:
                # c11 (l=0) -> W[4t] (k=2t, i=0) Correct.
                # c12 (l=1) -> W[4t+2] (k=2t+1, i=0) Correct.
                # c21 (l=2) -> W[4t+1] (k=2t, i=1) Correct.
                # c22 (l=3) -> W[4t+3] (k=2t+1, i=1) Correct.
                
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                
                col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Use exact integer check with a few random integer matrices
        for trial in range(5):
            A_rand = np.random.randint(-10, 10, (n, m))
            B_rand = np.random.randint(-10, 10, (m, p))
            C_expected = A_rand @ B_rand
            
            C_check = np.zeros((n, p), dtype=int)
            A_flat = A_rand.flatten()
            B_flat = B_rand.flatten()
            
            for r in range(rank):
                scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
                w_vec = W_arr[:, r]
                for k in range(p):
                    for i in range(n):
                        C_check[i, k] += w_vec[k * n + i] * scalar
            
            if not np.array_equal(C_check, C_expected):
                return False
        return True

    best = build_strassen_28()
    best_rank = 28
    
    if not verify_decomposition(best["U"], best["V"], best["W"], best_rank, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_school
    
    return best
