import time
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        if len(V[0]) != r or len(W[0]) != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_strassen_28():
        # Structure:
        # We view A as [A1 | A2] where A1 is 2x1 and A2 is 2x2.
        # We view B as [B1; B2] where B1 is 1x5 and B2 is 2x5.
        # C = A1*B1 + A2*B2.
        
        # A2*B2 uses Strassen (rank 7).
        # A1*B1 is a 2x1 * 1x5 multiplication (outer product).
        # We can exploit that A1 is a vector.
        
        # Let's try a specific known decomposition or construct one.
        # Actually, the "measured" improvement to 28 suggests a specific pattern exists.
        # Let's try to build it.
        
        # Strassen for 2x2 * 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For A1 (2x1) and B1 (1x5):
        # A1 = [a; b], B1 = [c1 c2 c3 c4 c5]
        # A1*B1 = [[a*c1, ..., a*c5], [b*c1, ..., b*c5]]
        # This is rank 2 (outer product of two vectors).
        # But we are multiplying 2x3 and 3x5.
        # The split is on the middle dimension 3.
        # A = [A_col0 | A_col1 | A_col2]
        # B = [B_row0; B_row1; B_row2]
        
        # If we split 3 into 1+2:
        # C = A_col0 * B_row0 + A_12 * B_12
        # A_12 is 2x2, B_12 is 2x5.
        # A_12 * B_12 can be done by extending Strassen?
        # Or just standard 2x2 * 2x5?
        # 2x2 * 2x5 standard rank: 2*2*5 = 20? No, we can do better.
        # Actually, 2x2 * 2xn can be done with rank 2n + 4? Or something.
        
        # Let's look at the "28" claim.
        # 30 - 28 = 2 saved.
        # Maybe the 1x1 part saves 2?
        # Standard 1x1 * 1x5 is rank 5.
        # Standard 2x2 * 2x5 is rank 20.
        # Total 25? No, standard is 30.
        
        # Wait, standard 2x3 * 3x5 is 30.
        # If we use Strassen on the 2x2 block of the 3x3?
        # No, the 3x3 is the contraction.
        
        # Let's try a different structural approach:
        # Use the fact that we can solve for the coefficients exactly.
        # We know rank 28 is possible.
        # Let's try to find a sparse integer solution for rank 28.
        
        # We will use a randomized local search but with a strong bias towards
        # the "Strassen-like" structure where the 2x2 sub-blocks are handled specially.
        
        # Actually, let's just try to find ANY rank 28 solution via random search
        # but with a tighter constraint: U and V must be sparse with entries in {-1, 0, 1}.
        
        target_rank = 28
        U = np.zeros((6, target_rank), dtype=int)
        V = np.zeros((15, target_rank), dtype=int)
        
        # Initialize with random sparse patterns
        # Density ~ 0.5
        mask_u = np.random.rand(6, target_rank) < 0.5
        mask_v = np.random.rand(15, target_rank) < 0.5
        U[mask_u] = np.random.choice([-1, 1], size=(6, target_rank))[mask_u]
        V[mask_v] = np.random.choice([-1, 1], size=(15, target_rank))[mask_v]
        
        # Solve for W
        A_mat = np.zeros((30, target_rank * 10))
        b = np.ones(30)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j, :].astype(float)
                    v_row = V[j*p+k, :].astype(float)
                    coeffs = u_row * v_row
                    start_idx = (k*n+i) * target_rank
                    A_mat[idx, start_idx:start_idx+target_rank] = coeffs
                    idx += 1
        
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A_mat, b, rcond=None)
        except:
            return None
            
        W_sol = x.reshape(10, target_rank)
        
        # Check if W is close to integers or half-integers
        # Try rounding to integers
        W_int = np.round(W_sol)
        if np.allclose(W_int, W_sol, atol=1e-4):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_int.tolist()
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
        # Try rounding to half-integers
        W_half = np.round(W_sol * 2) / 2
        if np.allclose(W_half, W_sol, atol=1e-4):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_half.tolist()
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
        return None

    # Try the structured search a few times
    for _ in range(100):
        if time.time() > deadline:
            break
        res = try_strassen_28()
        if res:
            return res

    return schoolbook()
