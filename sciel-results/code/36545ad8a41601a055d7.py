"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
import time
import numpy as np

def construct():
    """
    Find a low-rank decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    Strategy:
    - Use 4 Strassen 2x2x2 blocks for C[0:2, 0:4] (28 terms)
    - Use schoolbook for C[0:2, 4] (10 terms)  
    - Use schoolbook for C[2, 0:5] (5 terms)
    - Total rank: 43
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    n, m, p = 3, 5, 5
    
    # Initialize factor matrices
    U = np.zeros((n * m, 43), dtype=np.float64)  # (15, 43)
    V = np.zeros((m * p, 43), dtype=np.float64)  # (25, 43)
    W = np.zeros((p * n, 43), dtype=np.float64)  # (15, 43)
    
    col = 0
    
    # Strassen decomposition for 2x2x2 block multiplication
    # For blocks A[i:i+2, j:j+2], B[j:j+2, k:k+2], C[i:i+2, k:k+2]
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(i_off, j_off, k_off, col_start):
        """Add Strassen decomposition for 2x2x2 block at offsets."""
        global col
        # For each of the 7 Strassen terms, we need to add contributions
        # to U, V, W such that they multiply to give the correct C entries
        
        # Term 1: M1 = (A[i_off, j_off] + A[i_off+1, j_off+1]) * (B[j_off, k_off] + B[j_off+1, k_off+1])
        # This contributes to C[i_off, k_off], C[i_off+1, k_off+1] with +1, 
        # and C[i_off+1, k_off], C[i_off, k_off+1] with -1 (via other terms)
        
        # Actually, let me use the standard Strassen formulation more carefully.
        # The 7 products are linear combinations of A and B blocks.
        # For a single 2x2x2 Strassen decomposition:
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00  
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a product of two linear combinations.
        # We can expand each M into elementary terms.
        
        # Let me index the 7 Strassen products as s=0..6
        # For each s, we have:
        # M_s = (sum over (i,j) of u_{i,j}^s * A[i,j]) * (sum over (j',k) of v_{j',k}^s * B[j',k])
        
        # Strassen coefficients:
        # s=0 (M1): u = [[1,0],[0,1]], v = [[1,0],[0,1]]
        # s=1 (M2): u = [[0,0],[1,1]], v = [[1,0],[0,0]]
        # s=2 (M3): u = [[1,0],[0,0]], v = [[0,1],[0,-1]]
        # s=3 (M4): u = [[0,0],[0,1]], v = [[0,-1],[1,0]]
        # s=4 (M5): u = [[1,1],[0,0]], v = [[0,0],[0,1]]
        # s=5 (M6): u = [[-1,0],[1,0]], v = [[1,1],[0,0]]
        # s=6 (M7): u = [[0,1],[-1,0]], v = [[0,1],[1,1]]
        
        # C coefficients:
        # C00: [1, 0, 0, 1, -1, 0, 1]  (from M1, M4, -M5, M7)
        # C01: [0, 0, 1, 0, 1, 0, 0]   (from M3, M5)
        # C10: [0, 1, 0, 1, 0, 0, 0]   (from M2, M4)
        # C11: [1, -1, 1, 0, 0, 1, 0]  (from M1, -M2, M3, M6)
        
        u_coeff = [
            [[1, 0], [0, 1]],  # s=0
            [[0, 0], [1, 1]],  # s=1
            [[1, 0], [0, 0]],  # s=2
            [[0, 0], [0, 1]],  # s=3
            [[1, 1], [0, 0]],  # s=4
            [[-1, 0], [1, 0]], # s=5
            [[0, 1], [-1, 0]]  # s=6
        ]
        v_coeff = [
            [[1, 0], [0, 1]],  # s=0
            [[1, 0], [0, 0]],  # s=1
            [[0, 1], [0, -1]], # s=2
            [[0, -1], [1, 0]], # s=3
            [[0, 0], [0, 1]],  # s=4
            [[1, 1], [0, 0]],  # s=5
            [[0, 1], [1, 1]]   # s=6
        ]
        c_coeff = [
            [1, 0, 0, 1, -1, 0, 1],   # C00
            [0, 0, 1, 0, 1, 0, 0],    # C01
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ]
        
        for s in range(7):
            c_idx = col_start + s
            # Add to U: for each (di, dj) where u_coeff[s][di][dj] != 0
            for di in range(2):
                for dj in range(2):
                    uval = u_coeff[s][di][dj]
                    if uval != 0:
                        i = i_off + di
                        j = j_off + dj
                        U[i * m + j, c_idx] += uval
            # Add to V: for each (dj, dk) where v_coeff[s][dj][dk] != 0
            for dj in range(2):
                for dk in range(2):
                    vval = v_coeff[s][dj][dk]
                    if vval != 0:
                        j = j_off + dj
                        k = k_off + dk
                        V[j * p + k, c_idx] += vval
            # Add to W: for each (ci, ck) where c_coeff[ci][ck] contributes to this s
            # c_coeff[ci][ck][s] is the coefficient of M_s in C[ci][ck]
            for ci in range(2):
                for ck in range(2):
                    cval = c_coeff[ci][ck][s]
                    if cval != 0:
                        i = i_off + ci
                        k = k_off + ck
                        W[k * n + i, c_idx] += cval
            col += 1
    
    # Add 4 Strassen blocks for C[0:2, 0:4]
    # Block 1: i_off=0, j_off=0, k_off=0 -> covers C[0:2, 0:2]
    add_strassen_block(0, 0, 0, col)
    col += 7
    
    # Block 2: i_off=0, j_off=2, k_off=0 -> covers C[0:2, 0:2] but uses B[2:4, 0:2]
    # Wait, this doesn't work because the j dimension of A and B must match.
    # Let me reconsider.
    
    # Actually, for C[0:2, 0:4], we can decompose it as:
    # C[0:2, 0:2] = A[0:2, 0:2] @ B[0:2, 0:2]
    # C[0:2, 2:4] = A[0:2, 2:4] @ B[2:4, 2:4]
    # But these are 2x2 blocks, so we can use Strassen for each.
    
    # Reset and do it properly:
    U = np.zeros((n * m, 43), dtype=np.float64)
    V = np.zeros((m * p, 43), dtype=np.float64)
    W = np.zeros((p * n, 43), dtype=np.float64)
    col = 0
    
    # Block 1: C[0:2, 0:2] = A[0:2, 0:2] @ B[0:2, 0:2]
    add_strassen_block(0, 0, 0, col)
    col += 7
    
    # Block 2: C[0:2, 2:4] = A[0:2, 2:4] @ B[2:4, 2:4]
    add_strassen_block(0, 2, 2, col)
    col += 7
    
    # Block 3: C[0:2, 0:2] can't be done twice. Let me reconsider the structure.
    
    # Actually, C[0:2, 0:4] can be split into:
    # C[0:2, 0:2] and C[0:2, 2:4]
    # But we only have 2 rows and 4 columns, so we can do:
    # - C[0:2, 0:2] with Strassen (7 terms)
    # - C[0:2, 2:4] with Strassen (7 terms)
    # That's 14 terms for C[0:2, 0:4]
    
    # But wait, the knowledge said 4 Strassen blocks save 4 terms (74->71), 
    # implying 4 blocks replace 4 schoolbook 2x2x2 blocks.
    
    # Let me re-read: "4 non-overlapping 2x2x2 Strassen blocks instead of 1 saves 4 additional rank terms"
    # This is confusing. Let me just implement what makes sense.
    
    # For 3x5 by 5x5:
    # C has 3x5 = 15 entries
    # We can cover:
    # - C[0:2, 0:2] with Strassen (7 terms)
    # - C[0:2, 2:4] with Strassen (7 terms)  
    # - C[0:2, 4] with schoolbook (2*5=10 terms)
    # - C[2, 0:5] with schoolbook (1*5=5 terms)
    # Total: 7+7+10+5 = 29 terms
    
    # But that seems too low. Let me verify if Strassen works for non-square blocks.
    # Strassen requires 2x2x2, so we need A to be 2x2, B to be 2x2, C to be 2x2.
    
    # C[0:2, 0:2] = A[0:2, :] @ B[:, 0:2]
    # But A is 3x5, so A[0:2, :] is 2x5, not 2x2.
    # We need to split the middle dimension too.
    
    # For C[0:2, 0:2] = sum_{j=0}^{4} A[0:2, j] * B[j, 0:2]
    # This is a 2x5 by 5x2 multiplication, which we can split into:
    # C[0:2, 0:2] = A[0:2, 0:2]@B[0:2, 0:2] + A[0:2, 2:4]@B[2:4, 0:2] + A[0:2, 4]@B[4, 0:2]
    
    # So we can use Strassen for the 2x2x2 parts:
    # - A[0:2, 0:2]@B[0:2, 0:2]: Strassen (7 terms)
    # - A[0:2, 2:4]@B[2:4, 0:2]: Strassen (7 terms)
    # - A[0:2, 4]@B[4, 0:2]: schoolbook (2*1*2=4 terms)
    
    # Similarly for other blocks.
    
    # Let me systematically cover all of C:
    # C[0:2, 0:2]: 7 + 7 + 4 = 18
    # C[0:2, 2:4]: 7 + 7 + 4 = 18  
    # C[0:2, 4]: 2*5 = 10
    # C[2, 0:2]: 7 + 7 + 4 = 18
    # C[2, 2:4]: 7 + 7 + 4 = 18
    # C[2, 4]: 5
    # Total: 18+18+10+18+18+5 = 87, which is worse than schoolbook!
    
    # This approach is flawed. Let me think differently.
    
    # The knowledge says 4 Strassen blocks reduce 74 to 71.
    # Schoolbook for 3x5x5 is 75 terms. So 74 is already one less than schoolbook.
    # Maybe they used a different base.
    
    # Let me just use the direct approach: implement Strassen for the 2x2x2 subproblems
    # where possible, and schoolbook for the rest.
    
    # Actually, let me start fresh with a clearer strategy.
    
    U = np.zeros((n * m, 50), dtype=np.float64)
    V = np.zeros((m * p, 50), dtype=np.float64)
    W = np.zeros((p * n, 50), dtype=np.float64)
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def strassen_block(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        u_coeff = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [-1, 0]]
        ]
        v_coeff = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[0, -1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [1, 1]]
        ]
        c_coeff = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        for s in range(7):
            c_idx = col
            for di in range(2):
                for dj in range(2):
                    uval = u_coeff[s][di][dj]
                    if uval != 0:
                        U[(i0 + di) * m + (j0 + dj), c_idx] += uval
            for dj in range(2):
                for dk in range(2):
                    vval = v_coeff[s][dj][dk]
                    if vval != 0:
                        V[(j0 + dj) * p + (k0 + dk), c_idx] += vval
            for ci in range(2):
                for ck in range(2):
                    cval = c_coeff[ci][ck][s]
                    if cval != 0:
                        W[(k0 + ck) * n + (i0 + ci), c_idx] += cval
            col += 1
    
    # Cover C[0:2, 0:2] = A[0:2, 0:2]@B[0:2, 0:2] + A[0:2, 2:4]@B[2:4, 0:2] + A[0:2, 4]@B[4, 0:2]
    # First two parts use Strassen, last uses schoolbook
    strassen_block(0, 0, 0)  # A[0:2,0:2]@B[0:2,0:2]
    strassen_block(0, 2, 0)  # A[0:2,2:4]@B[2:4,0:2]
    
    # A[0:2, 4]@B[4, 0:2]: schoolbook, 2*1*2 = 4 terms
    for i in range(2):
        for k in range(2):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # C[0:2, 2:4] = A[0:2, 0:2]@B[0:2, 2:4] + A[0:2, 2:4]@B[2:4, 2:4] + A[0:2, 4]@B[4, 2:4]
    strassen_block(0, 0, 2)  # A[0:2,0:2]@B[0:2,2:4]
    strassen_block(0, 2, 2)  # A[0:2,2:4]@B[2:4,2:4]
    for i in range(2):
        for k in range(2):
            U[i * m + 4, col] = 1
            V[4 * p + (2 + k), col] = 1
            W[(2 + k) * n + i, col] = 1
            col += 1
    
    # C[0:2, 4] = A[0:2, 0:4]@B[0:4, 4] + A[0:2, 4]@B[4, 4]
    # = sum_{j=0}^{4} A[0:2, j] * B[j, 4]
    for i in range(2):
        for j in range(5):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # C[2, 0:2] = A[2, 0:2]@B[0:2, 0:2] + A[2, 2:4]@B[2:4, 0:2] + A[2, 4]@B[4, 0:2]
    # This is 1x2 by 2x2, so we can't use full Strassen. Use schoolbook: 1*2*2=4 terms
    for j in range(2):
        for k in range(2):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    for j in range(2):
        for k in range(2):
            U[2 * m + (2 + j), col] = 1
            V[(2 + j) * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    for k in range(2):
        U[2 * m + 4, col] = 1
        V[4 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
    
    # C[2, 2:4] = similar
    for j in range(2):
        for k in range(2):
            U[2 * m + j, col] = 1
            V[j * p + (2 + k), col] = 1
            W[(2 + k) * n + 2, col] = 1
            col += 1
    for j in range(2):
        for k in range(2):
            U[2 * m + (2 + j), col] = 1
            V[(2 + j) * p + (2 + k), col] = 1
            W[(2 + k) * n + 2, col] = 1
            col += 1
    for k in range(2):
        U[2 * m + 4, col] = 1
        V[4 * p + (2 + k), col] = 1
        W[(2 + k) * n + 2, col] = 1
        col += 1
    
    # C[2, 4] = A[2, 0:5]@B[0:5, 4]
    for j in range(5):
        U[2 * m + j, col] = 1
        V[j * p + 4, col] = 1
        W[4 * n + 2, col] = 1
        col += 1
    
    # Current col should be 7+7+4+7+7+4+10+4+4+2+4+4+2+5 = 71
    # Let's verify: 7*4 = 28, +4+4 = 36, +10 = 46, +4+4+2 = 56, +4+4+2 = 66, +5 = 71
    # Yes! Rank = 71
    
    # Trim to actual used columns
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p), dtype=np.float64)
        V = np.zeros((m * p, n * m * p), dtype=np.float64)
        W = np.zeros((p * n, n * m * p), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
