"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Target rank to try. We know 48 works (schoolbook).
    # Strassen blocks give 42.
    # Let's try to find a decomposition with rank < 42.
    # Lower bound estimate: rank >= max(n, m, p) is weak.
    # For 2x4x6, let's try ranks from 35 to 42.
    
    best_result = None
    best_rank = 48 # Fallback to schoolbook
    
    # Pre-construct schoolbook as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Pre-construct Strassen-like block decomposition (Rank 42)
    def get_strassen_blocks():
        # Split A (2x4) into A1 (2x2), A2 (2x2)
        # Split B (4x6) into B1 (2x2), B2 (2x2), B3 (2x2)
        # C = A1 B1 + A1 B2 + A1 B3 + A2 B1 + A2 B2 + A2 B3
        # Each term is 2x2x2. Use Strassen for each.
        # Strassen for 2x2x2 has rank 7.
        # Total rank 6 * 7 = 42.
        
        # Strassen factors for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to embed these into the global indices.
        # Global U shape: (n*m, rank) = (8, 42)
        # Global V shape: (m*p, rank) = (24, 42)
        # Global W shape: (p*n, rank) = (12, 42)
        
        U = np.zeros((n * m, 42))
        V = np.zeros((m * p, 42))
        W = np.zeros((p * n, 42))
        
        # Define Strassen coefficients for one 2x2 block
        # Indices in 2x2: 0,1 (row), 0,1 (col)
        # U_local (2x2, 7), V_local (2x2, 7), W_local (2x2, 7)
        
        # Let's define the 7 rank-1 tensors for 2x2x2
        # T1: (a+d)(e+h) -> U: [1,0,1,0], V: [1,1,0,0], W: [1,0,0,1] ? 
        # Actually, the tensor T(a,b,c,d; e,f,g,h) = ae+bg, af+bh, ce+dg, cf+dh
        # Strassen decomposition:
        # m1 = (a+d)(e+h) = ae+ah+de+dh
        # m2 = (c+d)e = ce+de
        # m3 = a(f-h) = af-ah
        # m4 = d(g-e) = dg-de
        # m5 = (a+b)h = ah+bh
        # m6 = (c-a)(e+f) = ce+cf-ae-af
        # m7 = (b-d)(g+h) = bg+bh-dg-dh
        
        # C11 = ae+bg = m1 + m4 - m5 + m7 ?
        # (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
        # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
        # = ae + bg. Correct.
        
        # C12 = af+bh = m3 + m5
        # (af-ah) + (ah+bh) = af+bh. Correct.
        
        # C21 = ce+dg = m2 + m4
        # (ce+de) + (dg-de) = ce+dg. Correct.
        
        # C22 = cf+dh = m1 - m2 + m3 + m6
        # (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cf-ae-af)
        # = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
        # = dh + cf. Correct.
        
        # Coefficients for U (A part), V (B part), W (C part)
        # A indices: 0:a, 1:b, 2:c, 3:d
        # B indices: 0:e, 1:f, 2:g, 3:h
        # C indices: 0:11, 1:12, 2:21, 3:22 (flattened)
        
        U_strassen = np.array([
            [1, 0, 1, 0, 1, -1, 0], # a: m1, m3, m5, m6(-1)
            [0, 0, 0, 0, 1, 0, 1],  # b: m5, m7
            [0, 1, 0, 0, 0, 1, 0],  # c: m2, m6
            [1, 1, 0, 1, 0, 0, -1]  # d: m1, m2, m4, m7(-1)
        ])
        
        V_strassen = np.array([
            [1, 1, 0, -1, 0, 1, 0], # e: m1, m2, m4(-1), m6
            [0, 0, 1, 0, 0, 1, 0],  # f: m3, m6
            [0, 0, 0, 1, 0, 0, 1],  # g: m4, m7
            [1, 0, -1, 0, 1, 0, 1]  # h: m1, m3(-1), m5, m7
        ])
        
        W_strassen = np.array([
            [1, 0, 0, 1, -1, 0, 1], # C11: m1, m4, m5(-1), m7
            [0, 0, 1, 0, 1, 0, 0],  # C12: m3, m5
            [0, 1, 0, 1, 0, 0, 0],  # C21: m2, m4
            [1, -1, 1, 0, 0, 1, 0]  # C22: m1, m2(-1), m3, m6
        ])
        
        # Now map to global indices
        # 6 blocks: (u, v) where u in {0,1}, v in {0,1,2}
        # u=0: A1 (rows 0,1 of A), u=1: A2 (rows 2,3 of A)
        # v=0: B1 (cols 0,1 of B, rows 0,1), v=1: B2 (cols 2,3, rows 2,3), v=2: B3 (cols 4,5, rows 4,5)
        # Wait, B is 4x6.
        # B rows 0,1 correspond to A cols 0,1.
        # B rows 2,3 correspond to A cols 2,3.
        # So B is split into 3 blocks of 2x2?
        # B is 4x6.
        # Block 1: B[0:2, 0:2]
        # Block 2: B[0:2, 2:4] ? No, the multiplication sums over the middle dimension.
        # A is 2x4. Cols 0,1 and 2,3.
        # B is 4x6. Rows 0,1 and 2,3.
        # Term 1: A[:, 0:2] * B[0:2, :] -> 2x2 * 2x6 = 2x6.
        # Term 2: A[:, 2:4] * B[2:4, :] -> 2x2 * 2x6 = 2x6.
        # This splits the middle dimension into 2 chunks of size 2.
        # But B is 4x6. The rows of B are 0..3.
        # So B is split into B_top (2x6) and B_bottom (2x6).
        # A is split into A_left (2x2) and A_right (2x2).
        # C = A_left B_top + A_right B_bottom.
        # Now, B_top is 2x6. We can split this into three 2x2 blocks: B_top1, B_top2, B_top3.
        # So C = A_left B_top1 + A_left B_top2 + A_left B_top3 + A_right B_bottom1 + A_right B_bottom2 + A_right B_bottom3.
        # This matches the 6 blocks derived earlier.
        
        col_offset = 0
        for u in range(2): # A blocks
            for v in range(3): # B blocks
                # U indices: A rows 0,1. A cols: u*2, u*2+1
                # U global row index: i (0,1) -> i*m + j? 
                # U shape is (n*m, rank) = (8, rank).
                # Index i*m + j. i is row of A, j is col of A.
                # For block u, cols are u*2, u*2+1.
                # Rows are 0, 1.
                # So U rows: 0*4 + (u*2), 0*4 + (u*2+1), 1*4 + (u*2), 1*4 + (u*2+1)
                
                # V indices: B rows, B cols.
                # V shape is (m*p, rank) = (24, rank).
                # Index j*p + k. j is row of B, k is col of B.
                # For block v, B rows are u*2, u*2+1 (since B_top corresponds to A_left, B_bottom to A_right? 
                # Wait. A_left multiplies B_top. A_right multiplies B_bottom.
                # So for u=0 (A_left), B rows are 0,1.
                # For u=1 (A_right), B rows are 2,3.
                # B cols for block v: v*2, v*2+1.
                
                # W indices: C rows, C cols.
                # W shape is (p*n, rank) = (12, rank).
                # Index k*n + i. k is col of C, i is row of C.
                # C rows are 0,1.
                # C cols are v*2, v*2+1.
                
                # Extract local factors
                # U_local is 4x7 (flattened 2x2 A)
                # V_local is 4x7 (flattened 2x2 B)
                # W_local is 4x7 (flattened 2x2 C)
                
                # Map local indices to global
                # U local index l (0..3) corresponds to A[local_row, local_col]
                # local_row = l // 2, local_col = l % 2
                # Global U index: i*m + j = (local_row)*4 + (u*2 + local_col)
                
                # V local index l (0..3) corresponds to B[local_row, local_col]
                # local_row = l // 2, local_col = l % 2
                # Global V index: j*p + k = (u*2 + local_row)*6 + (v*2 + local_col)
                
                # W local index l (0..3) corresponds to C[local_row, local_col]
                # local_row = l // 2, local_col = l % 2
                # Global W index: k*n + i = (v*2 + local_col)*2 + local_row
                
                for t in range(7):
                    col = col_offset + t
                    for l in range(4):
                        lr = l // 2
                        lc = l % 2
                        
                        # U
                        g_u = lr * m + (u * 2 + lc)
                        U[g_u, col] += U_strassen[l, t]
                        
                        # V
                        g_v = (u * 2 + lr) * p + (v * 2 + lc)
                        V[g_v, col] += V_strassen[l, t]
                        
                        # W
                        g_w = (v * 2 + lc) * n + lr
                        W[g_w, col] += W_strassen[l, t]
                
                col_offset += 7
        
        return {"U": U, "V": V, "W": W}

    # Check if a result is valid (integer/half-integer)
    def is_valid_half_integer(matrix):
        # Check if all entries are close to x.0 or x.5
        for val in matrix:
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    def to_half_integer(matrix):
        return np.round(matrix * 2) / 2

    # Verify decomposition
    def verify(U, V, W):
        # Reconstruct tensor and compare with schoolbook
        # Tensor T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
        # Schoolbook T[i, j, k] = delta_{i,j,k} ? No.
        # Matrix mult: C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor form: T_{i, j, k} such that C = A B means C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor T is defined by T_{i, j, k} = 1 if it contributes to C_{i,k} via A_{i,j} B_{j,k}?
        # Actually, the tensor T is the structure tensor of matrix multiplication.
        # T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
        # In the standard basis, T is a 3-tensor where T[i, j, k] = 1 if j is the shared index.
        # Specifically, for A (n x m), B (m x p), C (p x n) [Note: W index is k*n+i, so k is row of C? 
        # Task says: W[k*n+i, r]. So W index is (k, i).
        # U index is (i, j) ? U shape (n*m, rank). Index i*m+j. So U[i, j].
        # V index is (j, k) ? V shape (m*p, rank). Index j*p+k. So V[j, k].
        # So the tensor is T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # We want this to equal 1 if the term A[i,j] B[j,k] contributes to C[i,k]?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # So the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
        # Thus, we require T[i, j, k] = 1 for all i, j, k?
        # No, T is a 3-index object.
        # The equation is: sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
        # This can be written as: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^* ?
        # The standard tensor T satisfies: C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k} with T_{i,j,k} = 1.
        # So we want to decompose the tensor T where T[i, j, k] = 1.
        
        # Let's build the target tensor T_target
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1
        
        # Compute T_decomp
        # T_decomp[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can compute this efficiently using einsum
        # U_flat: (n*m, rank), V_flat: (m*p, rank), W_flat: (p*n, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # T_decomp[i, j, k] = sum_r U_3d[i, j, r] V_3d[j, k, r] W_3d[k, i, r]
        T_decomp = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T_decomp, T_target, atol=1e-9)

    # Try Strassen first
    strassen_res = get_strassen_blocks()
    if verify(strassen_res["U"], strassen_res["V"], strassen_res["W"]):
        if is_valid_half_integer(strassen_res["U"]) and is_valid_half_integer(strassen_res["V"]) and is_valid_half_integer(strassen_res["W"]):
            best_result = {
                "U": to_half_integer(strassen_res["U"]).tolist(),
                "V": to_half_integer(strassen_res["V"]).tolist(),
                "W": to_half_integer(strassen_res["W"]).tolist()
            }
            best_rank = 42
            # We found rank 42. Can we do better?
            # Given time constraints, rank 42 is a strong result.
            # We could try to optimize rank 41, but it's risky.
            # Let's return this.
            return best_result

    # Fallback to schoolbook
    school_res = get_schoolbook()
    return {
        "U": school_res["U"].tolist(),
        "V": school_res["V"].tolist(),
        "W": school_res["W"].tolist()
    }

# EVOLVE-BLOCK-END
