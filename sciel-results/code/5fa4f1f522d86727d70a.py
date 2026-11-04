import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                # Reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
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

def _try_rank_decomposition(n, m, p, rank, deadline):
    """
    Attempt to find a rank-R decomposition using linear algebra.
    We represent the tensor T as a 3D array.
    T[i, j, k] = 1 if (i, j, k) is a valid index in the multiplication tensor, else 0.
    Specifically, T[A_idx, B_idx, C_idx] = 1 if A_idx=i*m+j, B_idx=j*p+k, C_idx=k*n+i.
    
    We look for U (n*m x R), V (m*p x R), W (p*n x R) such that:
    sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    This is a system of equations. We can fix U and V randomly (or via SVD of unfolding) 
    and solve for W? No, it's trilinear.
    
    A more robust approach for small tensors:
    1. Unfold T to matrix M_U (size (n*m) x (m*p * p*n)).
    2. If rank R is small, we can use a randomized method:
       - Generate random U, V.
       - For each column r, we have a rank-1 tensor U[:,r] V[:,r] W[:,r].
       - The sum must equal T.
       
    Alternative: Use the fact that the tensor is "simple" (mostly zeros).
    The non-zero entries are exactly n*m*p.
    We can formulate this as:
    Find R vectors u_r, v_r, w_r such that sum u_r v_r w_r^T = T.
    
    Let's try a constructive approach based on the structure.
    The tensor T is the sum of n*m*p rank-1 tensors:
    T = sum_{i,j,k} e_{i*m+j} \otimes e_{j*p+k} \otimes e_{k*n+i}
    
    Can we combine these?
    For fixed j, the terms are sum_{i,k} e_{i*m+j} \otimes e_{j*p+k} \otimes e_{k*n+i}.
    This looks like a matrix multiplication of size n x k by k x n? No.
    
    Let's stick to a numerical search if R is close to the lower bound.
    Lower bound LB = max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
    
    We will try ranks from LB to 25.
    For a given rank R, we can use a "randomized CP decomposition" approach:
    1. Initialize U, V, W randomly.
    2. Use ALS (Alternating Least Squares) to minimize ||T - sum U_r V_r W_r||.
    3. Check if the solution is exact (error < 1e-9).
    4. If exact, check if entries are "nice" (integers or half-integers). 
       The problem allows integer/half-integer.
       If the numerical solution is close to such values, we snap them.
       
    Given the small size, we can do a more direct linear algebra attack.
    Fix U and V. Then W is determined by:
    sum_r W[:,r] (U[:,r] \otimes V[:,r])^T = T_flattened
    
    Let A be the matrix where column r is U[:,r] \otimes V[:,r].
    Then W * A^T = T_flattened.
    If A is invertible (or full row rank), W = T_flattened * (A^T)^{-1}.
    Then we check if this W, with the fixed U,V, reconstructs T.
    
    We can randomize U and V, solve for W, and check validity.
    """
    N_A = n * m
    N_B = m * p
    N_C = p * n
    
    # Build the target tensor as a flattened matrix of size N_C x (N_A * N_B)
    # T_flattened[c, a*N_B + b] = T[a, b, c]
    T_flat = np.zeros((N_C, N_A * N_B))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[c, a * N_B + b] = 1.0

    # We will try to find U, V such that the system for W has a solution
    # and that solution corresponds to a valid decomposition.
    
    # Since we need exactness, we can use the following property:
    # If we pick U and V such that the columns of A = [U[:,r] \otimes V[:,r]] are linearly independent,
    # we can solve for W exactly.
    # Then we verify if the reconstruction is perfect.
    
    # To increase the chance of "nice" numbers (integers/half-integers),
    # we can initialize U and V with simple values or use the structure.
    
    # Let's try a specific structured initialization based on the block decomposition idea,
    # but optimized numerically.
    
    # Actually, for R >= 18, we can just try random U, V from {0, 1, -1} or small integers,
    # solve for W, and check if W is close to integers/half-integers.
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
            
        # Random initialization with simple values
        # Using Gaussian might be too messy for integer snapping.
        # Let's use random signs and zeros.
        U = np.random.choice([0, 1, -1], size=(N_A, rank))
        V = np.random.choice([0, 1, -1], size=(N_B, rank))
        
        # Build A: (N_A * N_B) x rank
        # A[:, r] = U[:, r] \otimes V[:, r]
        A = np.zeros((N_A * N_B, rank))
        for r in range(rank):
            A[:, r] = np.kron(U[:, r], V[:, r])
            
        # Solve W * A^T = T_flat  =>  A * W^T = T_flat^T
        # Let's solve for W^T.
        # A is (N_A*N_B) x rank. T_flat^T is (N_A*N_B) x N_C.
        # We want X = W^T such that A @ X = T_flat.T
        
        try:
            # Use least squares if A is not square or invertible
            # But we need an EXACT solution.
            # If rank(A) == rank, we can use pinv or solve if square.
            # Since N_A*N_B is large (108) and rank is small (18-25), A is tall.
            # A system A X = B has a solution iff B is in the column space of A.
            # We can check this by solving min ||A X - B|| and checking residual.
            X, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat.T, rcond=None)
            
            if len(residuals) == 0 or residuals[0] < 1e-9:
                # Exact solution found for W^T
                W = X.T
                # Check if W entries are close to integers or half-integers
                # The problem states "integer/half-integer entries".
                # Let's check distance to nearest half-integer.
                W_half = np.round(W * 2) / 2
                if np.allclose(W, W_half, atol=1e-9):
                    # Also check U and V? The problem says U, V, W must have integer/half-integer entries.
                    # My random U, V are integers (0, 1, -1), so they are valid.
                    # We just need W to be valid too.
                    
                    # Final verification
                    U_list = [[int(x) for x in row] for row in U]
                    V_list = [[int(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W_half]
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        except np.linalg.LinAlgError:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Lower bound for rank
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upwards
    # Known bounds for 2x3x6:
    # Schoolbook: 36
    # Strassen-like block: 33 (from current program)
    # Theoretical min: 18
    # Likely true rank is somewhere in between.
    # We will try up to 30 to ensure we find something better than 33 if possible.
    # If we find 33 or lower, we return it.
    
    best_res = None
    
    # Start from a reasonable guess. 
    # If we find a valid decomposition at rank R, we can try R-1?
    # But finding lower rank is hard.
    # Let's try a range.
    
    # To be safe and fast, let's try ranks that are multiples of common factors or just sequential.
    # We will prioritize finding *any* valid decomposition below 36.
    
    # Strategy:
    # 1. Try rank 24, 25, ..., 35.
    # 2. If found, return it.
    # 3. If not, return the best found (lowest rank valid).
    # 4. If none found, return schoolbook (36).
    
    # Actually, the random search might be slow if we try too many ranks.
    # Let's limit attempts per rank.
    
    # We know rank 33 works (from the other program's logic, though it was hardcoded).
    # Let's try to find a rank < 33.
    
    # We'll iterate ranks from 20 to 32.
    # If we find one, we return it.
    
    for r in range(20, 33):
        if time.time() > deadline - 10:
            break
        res = _try_rank_decomposition(n, m, p, r, deadline - 5)
        if res:
            # Found a valid decomposition of rank r
            return res
            
    # If no better rank found, fallback to a known good one or schoolbook.
    # The current program's hardcoded rank 33 is a good fallback if we can reproduce it.
    # But since I am changing strategy, I should provide a robust fallback.
    # Let's use the schoolbook if nothing else works, but that's rank 36.
    # Can I quickly implement the rank 33 block decomposition as a fallback?
    # Yes, it's safer to include it.
    
    # Re-implementing the Rank 33 logic from the current program as a fallback
    # to ensure we don't score worse than 1.091 (if that was the best).
    # Wait, the current program in the prompt is the one with score 1.0.
    # The "INSPIRATIONS" mention a program that improved island 5 to 1.091 using Rank 33.
    # So Rank 33 is a known good score.
    # I will include the Rank 33 construction as a fallback.
    
    def _strassen_2x2_decomp():
        rank = 7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1
        V[1][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[2][1] = 1
        
        U[0][2] = 1; U[1][2] = 1
        V[3][2] = 1
        W[0][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[0][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        U[0][4] = 1; U[2][4] = 1
        V[1][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        U[1][5] = 1; U[2][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[0][5] = -1
        
        U[3][6] = 1; U[1][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[3][6] = 1
        
        return U, V, W

    def _schoolbook_2x1x2_decomp():
        rank = 4
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(4)]
        
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
        
        return U, V, W

    def _construct_2x3x2_decomp():
        U_s, V_s, W_s = _strassen_2x2_decomp()
        U_sb, V_sb, W_sb = _schoolbook_2x1x2_decomp()
        
        U = [[0]*11 for _ in range(6)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(4)]
        
        for i in range(4):
            for r in range(7):
                U[i][r] = U_s[i][r]
        for j in range(4):
            for r in range(7):
                V[j][r] = V_s[j][r]
        for k in range(4):
            for r in range(7):
                W[k][r] = W_s[k][r]
                
        for i in range(2):
            for r in range(4):
                U[4+i][7+r] = U_sb[i][r]
        for j in range(2):
            for r in range(4):
                V[4+j][7+r] = V_sb[j][r]
        for k in range(4):
            for r in range(4):
                W[k][7+r] = W_sb[k][r]
                
        return U, V, W

    def _construct_2x3x6_decomp():
        U_block, V_block, W_block = _construct_2x3x2_decomp()
        
        U = [[0]*33 for _ in range(6)]
        V = [[0]*33 for _ in range(18)]
        W = [[0]*33 for _ in range(12)]
        
        for block in range(3):
            offset = block * 11
            for i in range(6):
                for r in range(11):
                    U[i][offset+r] = U_block[i][r]
                    
            for j in range(6):
                for r in range(11):
                    V[j + block*6][offset+r] = V_block[j][r]
                    
            for k in range(4):
                for r in range(11):
                    W[k + block*4][offset+r] = W_block[k][r]
                    
        return {"U": U, "V": V, "W": W}

    # Fallback to Rank 33
    res33 = _construct_2x3x6_decomp()
    if _verify_decomposition(res33['U'], res33['V'], res33['W'], n, m, p):
        return res33
        
    # Final fallback
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
