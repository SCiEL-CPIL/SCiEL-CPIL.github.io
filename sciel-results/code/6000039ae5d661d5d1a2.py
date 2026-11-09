import time
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    global START_TIME, TIME_LIMIT
    
    # Trivial rank decomposition (baseline)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    def verify_decomposition_numpy(U, V, W, n, m, p):
        """Verify using NumPy broadcasting."""
        try:
            U_arr = np.array(U, dtype=np.int64).reshape(n, m, -1)
            V_arr = np.array(V, dtype=np.int64).reshape(m, p, -1)
            W_arr = np.array(W, dtype=np.int64).reshape(p, n, -1)
            result = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr)
            return np.all(result == 1)
        except Exception:
            return False

    def verify_decomposition_f2(U, V, W, n, m, p):
        """Verify over F2."""
        try:
            U_arr = np.array(U, dtype=np.int64).reshape(n, m, -1) % 2
            V_arr = np.array(V, dtype=np.int64).reshape(m, p, -1) % 2
            W_arr = np.array(W, dtype=np.int64).reshape(p, n, -1) % 2
            result = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr) % 2
            return np.all(result == 1)
        except Exception:
            return False

    # F5-based support search: find sparse supports over F5, then lift to integers
    def f5_support_search(r, n, m, p, max_iterations=15000):
        """Search for F5 decomposition with sparse supports, then try to lift."""
        np.random.seed(54321)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            # Generate sparse random matrices over F5
            U = np.random.randint(0, 5, size=(num_u, r))
            V = np.random.randint(0, 5, size=(num_v, r))
            W = np.random.randint(0, 5, size=(num_w, r))
            
            # Make them sparse: zero out ~70% of entries
            U = np.where(np.random.rand(num_u, r) < 0.3, U, 0) % 5
            V = np.where(np.random.rand(num_v, r) < 0.3, V, 0) % 5
            W = np.where(np.random.rand(num_w, r) < 0.3, W, 0) % 5
            
            # Check if this is a valid F5 decomposition
            U_arr = U.reshape(n, m, -1)
            V_arr = V.reshape(m, p, -1)
            W_arr = W.reshape(p, n, -1)
            result = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr) % 5
            if np.all(result == 1):
                # Found F5 decomposition - check if it lifts to integers
                if verify_decomposition_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Try sign flips and other lifts
                for su in [1, -1]:
                    for sv in [1, -1]:
                        for sw in [1, -1]:
                            U2 = (U * su).tolist()
                            V2 = (V * sv).tolist()
                            W2 = (W * sw).tolist()
                            if verify_decomposition_numpy(U2, V2, W2, n, m, p):
                                return {"U": U2, "V": V2, "W": W2}
        
        return None

    # Direct integer search with {-1, 0, 1} coefficients
    def integer_search(r, n, m, p, max_iterations=15000):
        """Direct search for integer decomposition."""
        np.random.seed(67890)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            U = np.random.choice([-1, 0, 1], size=(num_u, r))
            V = np.random.choice([-1, 0, 1], size=(num_v, r))
            W = np.random.choice([-1, 0, 1], size=(num_w, r))
            
            if verify_decomposition_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # F2-based support search: find sparse supports over F2, then lift to integers
    def f2_support_search(r, n, m, p, max_iterations=15000):
        """Search for F2 decomposition with sparse supports, then try to lift."""
        np.random.seed(12345)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            # Generate sparse random matrices over F2
            U = (np.random.rand(num_u, r) < 0.3).astype(int)
            V = (np.random.rand(num_v, r) < 0.3).astype(int)
            W = (np.random.rand(num_w, r) < 0.3).astype(int)
            
            if verify_decomposition_f2(U, V, W, n, m, p):
                # Found F2 decomposition - check if it lifts to integers
                if verify_decomposition_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Try sign flips for lifting
                for su in [1, -1]:
                    for sv in [1, -1]:
                        for sw in [1, -1]:
                            U2 = (U * su).tolist()
                            V2 = (V * sv).tolist()
                            W2 = (W * sw).tolist()
                            if verify_decomposition_numpy(U2, V2, W2, n, m, p):
                                return {"U": U2, "V": V2, "W": W2}
        
        return None

    return best_result
# EVOLVE-BLOCK-END
