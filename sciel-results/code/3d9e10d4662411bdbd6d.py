"""Matrix multiplication 3x3 by 3x4: find rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # leave 10s safety margin

def construct():
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def verify_decomposition(U, V, W):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # must equal 1 for all (i,j,k) in the multiplication tensor.
    # Actually, the matrix multiplication tensor is:
    # T[a,b,c] where a = i (row of A, 0..n-1), b = j (col of A / row of B, 0..m-1), c = k (col of B, 0..p-1)
    # T[i,j,k] = 1 if and only if the (i,k) entry of A*B gets contribution from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def try_rank(r):
    """Try to find a decomposition of rank r. Returns (U, V, W) or None."""
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    
    # For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: Use the fact that for small dimensions, we can try to find
    # decompositions using known results or systematic search.
    
    # Known lower bounds: For 3x3 * 3x4, the rank is at least max(n*m, m*p, p*n) = max(9,12,12) = 12
    # Actually, a better lower bound: rank >= max(n*p, m*p, n*m) for square, but for rectangular...
    # The rank of matrix multiplication [n,m,p] is at least max(n*m, m*p, p*n) / something...
    # Actually the trivial lower bound is max(n*m, m*p, p*n) is NOT correct.
    # The correct lower bound is: rank >= max(n, m, p) * something... 
    # Actually, rank >= n*m*p / max(n,m,p) is not standard.
    # Standard: rank([n,m,p]) >= max(n*m, m*p, p*n) is wrong.
    # The correct lower bound is rank >= n*m*p / (n+m+p-2) approximately, but exact is complex.
    # For 3x3x4: schoolbook is 36. Known best for 3x3x3 is 23 (Strassen-like).
    # For 3x3x4, let's try ranks from 12 upward.
    
    # Approach: For each rank r, set up the system and try to solve.
    # This is a system of polynomial equations (degree 3 in the variables).
    # For small r, we can try to find solutions.
    
    # Alternative: Use the "sum of products" structure.
    # T[i,j,k] = sum_{c=1}^{r} U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}
    
    # For rank 12 (the minimum possible given p*n = 12 and m*p = 12):
    # If rank = 12, then W is 12x12, V is 12x12, U is 9x12.
    # This is an underdetermined system potentially.
    
    # Let's try a different approach: exploit symmetry and structure.
    # For 3x3 * 3x4, note that it's like two 3x3 multiplications sharing the first factor.
    
    # Actually, let's just try to find ANY decomposition with rank < 36.
    # Start with rank 12 and go up.
    
    # For a systematic search, we can fix some structure.
    # Key insight: If we can find a decomposition where the W matrix has a special form...
    
    # Let's try: for each rank r from 12 to 35, attempt to find a solution.
    # We'll use a greedy/iterative approach.
    
    for r in range(12, 36):
        if time.time() - START_TIME > TIME_LIMIT:
            return None
        
        # For rank r, we have r "terms" each being a rank-1 tensor.
        # Term c: U_c[i,j] * V_c[j,k] * W_c[k,i]
        # We need sum_c of these = 1 for all (i,j,k).
        
        # Strategy: Try to find r rank-1 tensors whose sum gives the multiplication tensor.
        # For simplicity, try structured solutions.
        
        # Approach 1: Try to extend Strassen-like ideas.
        # For 3x3x3, Strassen gives rank 23. For 3x3x4, we can think of it as
        # computing two 3x3 products (columns 0-2 and column 3 of B separately),
        # but that's not quite right since it's one 3x4 product.
        
        # Let's try a direct algebraic approach for small ranks.
        # Fix the structure: assume W_c[k,i] takes values in {-1, 0, 1} for simplicity,
        # and try to find U_c and V_c accordingly.
        
        # Actually, let's try a more principled approach:
        # The tensor is T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,4).
        # We want T = sum_c u_c ⊗ v_c ⊗ w_c where (u_c)_{i,j} = U[i*m+j, c], etc.
        
        # For rank r, this is equivalent to finding r triples of matrices
        # (A_c, B_c, C_c) where A_c is 3x3, B_c is 3x4, C_c is 4x3,
        # such that for all i,j,k: sum_c A_c[i,j] * B_c[j,k] * C_c[k,i] = 1.
        
        # This is a system of 3*3*4 = 36 equations in r*(9+12+12) = 33r unknowns.
        # For r >= 2, it's underdetermined, so solutions likely exist.
        
        # Let's try to find a solution for small r using a simple iterative method.
        # Initialize randomly and use gradient descent on the residual.
        
        import random
        random.seed(42)
        
        best_solution = None
        best_residual = float('inf')
        
        for attempt in range(5):
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Initialize
            U_vals = np.random.randn(n*m, r) * 0.5
            V_vals = np.random.randn(m*p, r) * 0.5
            W_vals = np.random.randn(p*n, r) * 0.5
            
            # Iterate
            lr = 0.01
            for it in range(200):
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                    
                # Compute residual
                residual = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U_vals[i*m+j, c] * V_vals[j*p+k, c] * W_vals[k*n+i, c]
                            residual[i,j,k] = s - 1.0
                
                res_norm = np.linalg.norm(residual)
                if res_norm < 1e-10:
                    # Found exact solution (within numerical precision)
                    # Try to round to rationals
                    U_int = np.round(U_vals * 2) / 2
                    V_int = np.round(V_vals * 2) / 2
                    W_int = np.round(W_vals * 2) / 2
                    
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(r):
                                    s += U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c]
                                if abs(s - 1.0) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        # Convert to list of lists
                        U_list = U_int.tolist()
                        V_list = V_int.tolist()
                        W_list = W_int.tolist()
                        return (U_list, V_list, W_list)
                    
                    # If rounding failed, try with smaller scale
                    for scale in [1, 2, 4, 8]:
                        U_int = np.round(U_vals * scale) / scale
                        V_int = np.round(V_vals * scale) / scale
                        W_int = np.round(W_vals * scale) / scale
                        
                        ok = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = 0
                                    for c in range(r):
                                        s += U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c]
                                    if abs(s - 1.0) > 1e-9:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        
                        if ok:
                            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
                    # Use the floating point solution as-is (convert to fraction if possible)
                    U_frac = np.array([[Fraction(U_vals[i,j]).limit_denominator(1000) for j in range(r)] for i in range(n*m)])
                    V_frac = np.array([[Fraction(V_vals[i,j]).limit_denominator(1000) for j in range(r)] for i in range(m*p)])
                    W_frac = np.array([[Fraction(W_vals[i,j]).limit_denominator(1000) for j in range(r)] for i in range(p*n)])
                    
                    # Verify with fractions
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for c in range(r):
                                    s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                                if s != 1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        U_list = [[float(x) for x in row] for row in U_frac]
                        V_list = [[float(x) for x in row] for row in V_frac]
                        W_list = [[float(x) for x in row] for row in W_frac]
                        return (U_list, V_list, W_list)
                    
                    # Just use the float values
                    return (U_vals.tolist(), V_vals.tolist(), W_vals.tolist())
                
                # Gradient descent step
                # d(residual[i,j,k])/d(U[i*m+j, c]) = V[j*p+k, c] * W[k*n+i, c]
                # d(residual[i,j,k])/d(V[j*p+k, c]) = U[i*m+j, c] * W[k*n+i, c]
                # d(residual[i,j,k])/d(W[k*n+i, c]) = U[i*m+j, c] * V[j*p+k, c]
                
                dU = np.zeros_like(U_vals)
                dV = np.zeros_like(V_vals)
                dW = np.zeros_like(W_vals)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                dU[i*m+j, c] += residual[i,j,k] * V_vals[j*p+k, c] * W_vals[k*n+i, c]
                                dV[j*p+k, c] += residual[i,j,k] * U_vals[i*m+j, c] * W_vals[k*n+i, c]
                                dW[k*n+i, c] += residual[i,j,k] * U_vals[i*m+j, c] * V_vals[j*p+k, c]
                
                U_vals -= lr * dU
                V_vals -= lr * dV
                W_vals -= lr * dW
                
                # Decay learning rate
                if it % 50 == 0 and it > 0:
                    lr *= 0.5
        
        # If we got here without finding a solution for this rank, try next rank
    
    return None

# Try to find a better decomposition
def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Try ranks from 12 up to 35
    for r in range(12, 36):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Verify
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return construct_schoolbook()

def construct_schoolbook():
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
