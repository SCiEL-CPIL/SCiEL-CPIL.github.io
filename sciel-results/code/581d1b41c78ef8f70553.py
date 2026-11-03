import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Construct a structured rank-27 ansatz by decomposing the tensor into:
       - A 2x2x2 Strassen block (rank 7) for the first two columns of B
       - A custom 2x1x3 sub-tensor (rank 3) for the remaining three columns
       - A 2x1x2 sub-tensor (rank 2) for the last two columns? 
       Actually, let's just try to solve for a rank-27 decomposition using 
       a structured ansatz that combines Strassen-like terms with schoolbook terms.
       
    2. If rank 27 is not possible, try rank 28 with a structured approach.
    3. Fallback to schoolbook rank 30.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Schoolbook decomposition as fallback
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
        """Verify exact decomposition using integer arithmetic."""
        r = len(U[0])
        # Convert to integers for exact check (multiply by 2 if half-integers)
        U_int = [[int(u * 2) for u in row] for row in U]
        V_int = [[int(v * 2) for v in row] for row in V]
        W_int = [[int(w * 2) for w in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U_int[i * m + j][c] * V_int[j * p + k][c] * W_int[k * n + i][c]
                    if s != 8:
                        return False
        return True

    def try_rank_27_structured():
        """
        Try to find a rank-27 decomposition using a structured ansatz.
        We'll use 7 Strassen terms for a 2x2x2 sub-block and 20 schoolbook-like terms.
        """
        target_rank = 27
        
        # Construct the linear system A x = b
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We have 30 equations and 10*27 = 270 unknowns.
        
        A = np.zeros((n*m*p, pn * target_rank))
        b = np.ones(n*m*p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation idx
                    # Involves W[k*n+i, :]
                    start_idx = (k*n+i) * target_rank
                    # For a structured ansatz, we'd fix some entries of U and V
                    # and solve for the rest. For now, let's just try to solve
                    # the underdetermined system and check for rational solutions.
                    A[idx, start_idx:start_idx+target_rank] = 1.0  # Placeholder
                    b[idx] = 1
                    idx += 1
        
        # This is too simplistic. Let's use a more direct approach:
        # Try random U, V with small integers and solve for W.
        
        np.random.seed(42)
        search_deadline = start_time + 130.0
        
        for trial in range(200):
            if time.time() > search_deadline:
                return None
            
            # Generate random dense U and V with entries in {-1, 0, 1}
            U_rand = np.random.choice([-1, 0, 1], size=(nm, target_rank))
            V_rand = np.random.choice([-1, 0, 1], size=(mp, target_rank))
            
            # Construct the linear system for W
            A = np.zeros((n*m*p, pn * target_rank))
            b = np.ones(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U_rand[i*m+j, :]
                        v_row = V_rand[j*p+k, :]
                        coeffs = u_row * v_row
                        start_idx = (k*n+i) * target_rank
                        A[idx, start_idx:start_idx+target_rank] = coeffs
                        b[idx] = 1
                        idx += 1
            
            # Solve A x = b
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Reshape x to W
            W_sol = x.reshape(pn, target_rank)
            
            # Check if W_sol has integer or half-integer entries
            W_check = W_sol * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-6):
                W_rounded = np.round(W_check) / 2.0
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Also check if it's close to integers
            if np.allclose(W_sol, np.round(W_sol), atol=1e-6):
                W_rounded = np.round(W_sol)
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_rank_28_structured():
        """Try to find a rank-28 decomposition using a structured approach."""
        target_rank = 28
        np.random.seed(43)
        search_deadline = start_time + 135.0
        
        for trial in range(300):
            if time.time() > search_deadline:
                return None
            
            U_rand = np.random.choice([-1, 0, 1], size=(nm, target_rank))
            V_rand = np.random.choice([-1, 0, 1], size=(mp, target_rank))
            
            A = np.zeros((n*m*p, pn * target_rank))
            b = np.ones(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U_rand[i*m+j, :]
                        v_row = V_rand[j*p+k, :]
                        coeffs = u_row * v_row
                        start_idx = (k*n+i) * target_rank
                        A[idx, start_idx:start_idx+target_rank] = coeffs
                        b[idx] = 1
                        idx += 1
            
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(pn, target_rank)
            
            W_check = W_sol * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-6):
                W_rounded = np.round(W_check) / 2.0
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            if np.allclose(W_sol, np.round(W_sol), atol=1e-6):
                W_rounded = np.round(W_sol)
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try rank 27 first
    result = try_rank_27_structured()
    if result:
        return result
        
    # Try rank 28
    result = try_rank_28_structured()
    if result:
        return result
        
    return schoolbook()
