"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify_fast(U, V, W):
        """Faster verification using einsum"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _snap_to_half_int(arr):
        """Snap values to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _solve_linear_system(A, b):
        """Solve Ax = b using least squares, return None if no exact solution"""
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.allclose(A @ x, b, altol=1e-10):
                return x
        except:
            pass
        return None

    def _try_fixed_W_decomposition(rank, W_pattern, max_iter=100):
        """
        Try to find U, V such that for all (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        Fix W to W_pattern, then alternate solving for U and V.
        """
        W = W_pattern.copy()
        
        # Initialize U and V randomly from {-1, 0, 1}
        np.random.seed(42)
        U = np.random.choice([-1, 0, 1], size=(n*m, rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, rank))
        
        num_equations = n * m * p
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                return None
            
            # Fix U and W, solve for V
            A_v = np.zeros((num_equations, m * p * rank))
            b_v = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            unk_idx = (j * p + k) * rank + r
                            A_v[eq_idx, unk_idx] = U[i*m+j, r] * W[k*n+i, r]
                        eq_idx += 1
            
            V_flat = _solve_linear_system(A_v, b_v)
            if V_flat is None:
                return None
            V = V_flat.reshape(m * p, rank)
            
            # Fix V and W, solve for U
            A_u = np.zeros((num_equations, n * m * rank))
            b_u = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            unk_idx = (i * m + j) * rank + r
                            A_u[eq_idx, unk_idx] = V[j*p+k, r] * W[k*n+i, r]
                        eq_idx += 1
            
            U_flat = _solve_linear_system(A_u, b_u)
            if U_flat is None:
                return None
            U = U_flat.reshape(n * m, rank)
        
        # Verify the solution
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            U = _snap_to_half_int(U)
            V = _snap_to_half_int(V)
            W = _snap_to_half_int(W)
            if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def _try_systematic_search(rank):
        """
        Systematic search for rank-r decomposition.
        Try different W patterns and solve for U, V.
        """
        # Try multiple W patterns
        patterns = []
        
        # Pattern 1: All ones
        W1 = np.ones((p*n, rank))
        patterns.append(W1)
        
        # Pattern 2: (-1)^k
        W2 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W2[k*n+i, r] = (-1)**k
        patterns.append(W2)
        
        # Pattern 3: 1 if k==0 else 0
        W3 = np.zeros((p*n, rank))
        for i in range(n):
            for r in range(rank):
                W3[0*n+i, r] = 1
        patterns.append(W3)
        
        # Pattern 4: r+1
        W4 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W4[k*n+i, r] = r + 1
        patterns.append(W4)
        
        # Pattern 5: 1 if (k+i) % 2 == 0 else -1
        W5 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W5[k*n+i, r] = 1 if (k+i) % 2 == 0 else -1
        patterns.append(W5)
        
        # Pattern 6: 1 if r < rank//2 else -1
        W6 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W6[k*n+i, r] = 1 if r < rank // 2 else -1
        patterns.append(W6)
        
        # Pattern 7: Random with seed
        np.random.seed(rank)
        W7 = np.random.choice([-1, 0, 1], size=(p*n, rank))
        patterns.append(W7)
        
        for W_pattern in patterns:
            if time.time() - start_time > time_limit:
                break
            result = _try_fixed_W_decomposition(rank, W_pattern)
            if result is not None:
                return result
        
        return None

    # Try to find lower-rank decompositions
    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ranks from 12 to rank_schoolbook - 1
    for r in range(12, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_systematic_search(r)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        U = [[float(x) for x in row] for row in U]
        V = [[float(x) for x in row] for row in V]
        W = [[float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
