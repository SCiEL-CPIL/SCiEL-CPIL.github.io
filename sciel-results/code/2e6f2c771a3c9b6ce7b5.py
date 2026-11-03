import time
import numpy as np
from itertools import product

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
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Strategy: Structured Block Search
    # We look for a decomposition with rank R < 30.
    # The middle dimension is m=3. We can view the tensor as T_{i,j,k}.
    # A common structure for beating schoolbook is to exploit the fact that 
    # some combinations of j and k can be shared.
    
    # For 2x3x5, let's try to construct a rank 28 or 29 solution.
    # We will use a "Pattern Search" where we fix the sparsity pattern 
    # of U and V to have a specific block structure, and solve for W.
    
    # Instead of random U, V, we use a "Greedy Basis" approach:
    # We want to cover all 30 (i,j,k) indices.
    # Each term c contributes to U[i,j,c] * V[j,k,c] * W[k,i,c].
    # If we fix U and V to have entries in {-1, 0, 1}, the system for W is linear.
    # The issue is that W must be integer/half-integer.
    
    # New Idea: Use the "Strassen" logic for the 3x3 part if possible, 
    # but since one dim is 2 and other is 5, it's irregular.
    
    # Let's try a specific structured search for Rank 29.
    # We start from the schoolbook solution (Rank 30) and try to merge two terms.
    # If we can express the contribution of term A and term B as a single term C 
    # (or a linear combination that fits the integer constraint), we reduce rank.
    
    # Merging terms:
    # Term a: U_a, V_a, W_a
    # Term b: U_b, V_b, W_b
    # We want to replace them with Term c: U_c, V_c, W_c
    # Such that U_a V_a W_a + U_b V_b W_b = U_c V_c W_c
    # This is generally hard.
    
    # Alternative: Direct construction for Rank 28.
    # We will define U, V, W with specific sparsity patterns.
    # Pattern:
    # U (6xR): 2 non-zeros per column?
    # V (15xR): 2 non-zeros per column?
    # W (10xR): 1 non-zero per column? (If W is 1-sparse, it's a permutation matrix essentially, 
    # which might be too restrictive).
    
    # Let's try a "Randomized Algebraic Search" with stronger constraints on W.
    # We assume W has entries in {0, 1, -1} (or half-integers).
    # We iterate over sparse patterns for U and V, solve for W, and check integrality.
    
    def try_rank(R):
        if time.time() > deadline:
            return None
            
        # We will try to find a solution where W is 1-sparse or 2-sparse?
        # Actually, let's just solve for W and check if it's "close" to integers.
        # To increase the chance of integer solutions, we can scale U and V.
        # If we pick U, V from {0, 1, -1}, the coefficients in the linear system 
        # for W are products of these, so they are integers.
        # The solution W = A^-1 b might be rational.
        # We check if W is integer or half-integer.
        
        # To speed up, we use a specific heuristic for U and V:
        # "Dense" in the j-dimension (middle) to allow sharing.
        
        max_attempts = 2000
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
                
            # Generate random sparse U, V
            # We want U[i*m+j] to be non-zero for specific (i,j)
            # and V[j*p+k] to be non-zero for specific (j,k)
            
            # Let's try a "Block" structure on the middle index j (0,1,2).
            # Maybe split j into {0} and {1,2}?
            
            # Random generation with controlled sparsity
            # U: 6 x R
            # V: 15 x R
            
            # Heuristic: Make U and V "balanced".
            # Each column c in U should have non-zeros for i=0 and i=1?
            # No, let's just use random masks.
            
            # Density tuning: 
            # If density is too high, the system is over-determined and might not have integer solutions.
            # If too low, it's under-determined.
            
            # Let's try to make U and V have exactly 2 non-zeros per column on average.
            
            U = np.zeros((6, R), dtype=int)
            V = np.zeros((15, R), dtype=int)
            
            for c in range(R):
                # Pick 2 random rows for U
                rows_u = np.random.choice(6, size=2, replace=False)
                for r in rows_u:
                    U[r, c] = np.random.choice([-1, 1])
                
                # Pick 2 random rows for V
                rows_v = np.random.choice(15, size=2, replace=False)
                for r in rows_v:
                    V[r, c] = np.random.choice([-1, 1])
            
            # Solve for W
            # System: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # Linear in W.
            
            A = np.zeros((30, R * 10))
            b = np.ones(30)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m+j, :].astype(float)
                        v_row = V[j*p+k, :].astype(float)
                        coeffs = u_row * v_row
                        start_idx = (k*n+i) * R
                        A[idx, start_idx:start_idx+R] = coeffs
                        idx += 1
            
            try:
                # Use lstsq
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(10, R)
            
            # Check if W_sol is close to integers or half-integers
            # We allow a small tolerance for floating point errors
            tol = 1e-5
            
            # Check integers
            W_int = np.round(W_sol)
            if np.allclose(W_sol, W_int, atol=tol):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_int.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": U_list, "W": W_list} # Note: return correct dict
            
            # Check half-integers
            W_half = np.round(W_sol * 2) / 2.0
            if np.allclose(W_sol, W_half, atol=tol):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_half.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Check if W_sol can be "cleaned" by scaling?
            # No, U and V are fixed.
            
            # Optional: Try to perturb U or V slightly if W is close?
            # This is complex. Let's stick to the current loop.
            
        return None

    # Try ranks 29 down to 25
    for r in range(29, 24, -1):
        if time.time() > deadline:
            break
        res = try_rank(r)
        if res:
            return res

    return schoolbook()
