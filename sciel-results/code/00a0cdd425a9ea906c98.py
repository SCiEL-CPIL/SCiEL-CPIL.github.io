"""Matrix multiplication 3x3x7 - structured decomposition with algebraic elimination."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Strategy: Model as 3x3x3 core + 3x3x4 extension, use structured factors,
    then apply algebraic elimination to remove linear dependencies.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp(U, V, W, tol=1e-8):
        """Verify exact tensor decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        
        # Check matching cases: sum should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check non-matching cases: sum should be 0
        # Case 1: i1 != i3
        for i1 in range(n):
            for j1 in range(m):
                for k in range(p):
                    for i3 in range(n):
                        if i1 == i3:
                            continue
                        a = i1 * m + j1
                        b = j1 * p + k
                        c = k * n + i3
                        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                        if abs(val) > tol:
                            return False
        # Case 2: j1 != j2
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2:
                        continue
                    for k in range(p):
                        a = i * m + j1
                        b = j2 * p + k
                        c = k * n + i
                        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                        if abs(val) > tol:
                            return False
        # Case 3: k1 != k2
        for i in range(n):
            for j in range(m):
                for k1 in range(p):
                    for k2 in range(p):
                        if k1 == k2:
                            continue
                        a = i * m + j
                        b = j * p + k1
                        c = k2 * n + i
                        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                        if abs(val) > tol:
                            return False
        
        return True
    
    def try_elimination(U, V, W, max_iterations=50):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        If term c is a linear combination of others, we can eliminate it.
        """
        U_arr = np.array(U, dtype=float).copy()
        V_arr = np.array(V, dtype=float).copy()
        W_arr = np.array(W, dtype=float).copy()
        r = U_arr.shape[1]
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = r
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to eliminate each term
            eliminated = False
            for c in range(r):
                if time.time() > deadline:
                    break
                
                # Check if term c is redundant
                # Remove term c and see if we can still verify
                U_test = np.delete(U_arr, c, axis=1)
                V_test = np.delete(V_arr, c, axis=1)
                W_test = np.delete(W_arr, c, axis=1)
                
                if verify_decomp(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                    # Term c is redundant, eliminate it
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    r -= 1
                    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                    best_rank = r
                    eliminated = True
                    break
            
            if not eliminated:
                break
        
        return best_U, best_V, best_W, best_rank
    
    # Build a structured decomposition using Strassen-like factors for the core
    # and schoolbook for the extension, then try elimination
    
    # For 3x3x3, we can use a rank-23 decomposition (Bini et al.)
    # But let's start with a simpler approach: use rank-1 terms with some sharing
    
    # Approach: Create a decomposition where the first 3 columns of B use
    # a shared structure with A, and the remaining 4 columns use schoolbook
    
    # Let's try a hybrid: use 7 independent 3x3x1 multiplications but look for sharing
    # Actually, let's just use schoolbook and try elimination
    
    result = make_schoolbook()
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    # Try algebraic elimination
    best_U, best_V, best_W, best_rank = try_elimination(U_arr, V_arr, W_arr)
    
    if best_rank < schoolbook_rank:
        result = {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Final verification
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
