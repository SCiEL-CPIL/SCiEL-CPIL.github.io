<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _build_trivial_decomposition(n, m, p):
    """Build the trivial rank-(n*m*p) decomposition."""
    rank = n * m * p
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

def _try_rank_r(n, m, p, rank, seed, time_budget):
    """
    Try to find a rank-r decomposition by:
    1. Generating random U, V with half-integer entries
    2. Solving the linear system for W
    3. Checking if W is close to half-integers
    """
    start_time = time.time()
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    np.random.seed(seed)
    
    num_vars = p_n * rank
    num_eqs = n * m * p
    
    # Pre-compute the structure of the linear system
    # For each equation (i,j,k), the coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
    
    # We'll use a sparse-friendly approach: build the system incrementally
    # But for small systems, dense is fine
    
    vals = np.array([-1, -0.5, 0, 0.5, 1])
    
    max_attempts = 100000
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        # Generate U and V with structured sparsity to increase chance of half-integer W
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        
        # Use a mix of sparse and dense patterns
        if attempt % 3 == 0:
            # Sparse pattern: each row has ~2 non-zeros
            for i in range(n_m):
                nz_cols = np.random.choice(rank, size=min(2, rank), replace=False)
                U[i, nz_cols] = np.random.choice([-1, 1], size=len(nz_cols))
            for i in range(m_p):
                nz_cols = np.random.choice(rank, size=min(2, rank), replace=False)
                V[i, nz_cols] = np.random.choice([-1, 1], size=len(nz_cols))
        elif attempt % 3 == 1:
            # Medium density
            for i in range(n_m):
                for r in range(rank):
                    if np.random.random() < 0.25:
                        U[i, r] = np.random.choice([-1, 1])
            for i in range(m_p):
                for r in range(rank):
                    if np.random.random() < 0.2:
                        V[i, r] = np.random.choice([-1, 1])
        else:
            # Dense with half-integers
            U = np.random.choice(vals, size=(n_m, rank))
            V = np.random.choice(vals, size=(m_p, rank))
        
        # Build the linear system for W
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            A_mat[eq_idx, var_idx] = coeff
                    eq_idx += 1
        
        # Solve for W using least squares
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, rank)
        
        # Check if W is close to half-integers
        W_2x = W * 2
        W_rounded_2x = np.round(W_2x)
        diff = np.abs(W_2x - W_rounded_2x)
        
        if np.max(diff) > 0.05:  # Tighter tolerance
            continue
        
        W_rounded = W_rounded_2x / 2
        
        # Verify the decomposition exactly
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
>>>>>>> REPLACE
<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _try_rank20_structured(n, m, p, time_budget=120.0):
    """
    Try to find a rank-20 decomposition for 2x3 * 3x4.
    
    Strategy: Use the fact that we can share computations.
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    We can define intermediate quantities to reduce the number of multiplications.
    
    Let's try:
    M0 = A[0,0]
    M1 = A[0,1]
    M2 = A[0,2]
    M3 = A[1,0]
    M4 = A[1,1]
    M5 = A[1,2]
    M6 = B[0,0]
    M7 = B[0,1]
    M8 = B[0,2]
    M9 = B[0,3]
    M10 = B[1,0]
    M11 = B[1,1]
    M12 = B[1,2]
    M13 = B[1,3]
    M14 = B[2,0]
    M15 = B[2,1]
    M16 = B[2,2]
    M17 = B[2,3]
    
    Then:
    C[0,0] = M0*M6 + M1*M10 + M2*M14
    C[0,1] = M0*M7 + M1*M11 + M2*M15
    C[0,2] = M0*M8 + M1*M12 + M2*M16
    C[0,3] = M0*M9 + M1*M13 + M2*M17
    C[1,0] = M3*M6 + M4*M10 + M5*M14
    C[1,1] = M3*M7 + M4*M11 + M5*M15
    C[1,2] = M3*M8 + M4*M12 + M5*M16
    C[1,3] = M3*M9 + M4*M13 + M5*M17
    
    This is 24 multiplications. To reduce, we can try to factor.
    
    For example, for row 0:
    C[0,k] = M0*B[0,k] + M1*B[1,k] + M2*B[2,k]
    
    This is a dot product of (M0,M1,M2) with column k of B.
    
    We can't easily reduce this below 3 multiplications per output without additional structure.
    
    Let's try a different approach: use the tensor decomposition directly.
    
    We need to find rank-1 tensors (u_r, v_r, w_r) such that:
    sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all (i,j,k)
    
    Let's try to find a rank-20 solution by solving the system.
    """
    start_time = time.time()
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    # We'll try to find a rank-20 decomposition by fixing U and V to have a structured form
    # and solving for W.
    
    # Key insight: if we fix U and V, the equations are linear in W.
    # We need W to have half-integer entries.
    
    # Let's try: set U to have a specific pattern where each row has at most 2 non-zero entries,
    # and V similarly. This reduces the effective rank of the coefficient matrix.
    
    # Actually, let's try a more direct approach:
    # Generate random U, V with entries in {-1, 0, 1}, solve for W, and check if W can be
    # rounded to half-integers while still satisfying the equations.
    
    # The key improvement: after solving for W, instead of just rounding and checking,
    # we can try to find the nearest half-integer point that satisfies the equations.
    
    np.random.seed(42)
    
    for attempt in range(50000):
        if time.time() - start_time > time_budget:
            break
        
        rank = 20
        
        # Generate U and V with sparse structure to make the system more likely to have
        # a half-integer solution
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        
        # Each entry of U is non-zero with probability 0.3, value in {-1, 1}
        for i in range(n_m):
            for r in range(rank):
                if np.random.random() < 0.3:
                    U[i, r] = np.random.choice([-1, 1])
        
        # Each entry of V is non-zero with probability 0.2, value in {-1, 1}
        for i in range(m_p):
            for r in range(rank):
                if np.random.random() < 0.2:
                    V[i, r] = np.random.choice([-1, 1])
        
        # Build the linear system for W
        num_vars = p_n * rank
        num_eqs = n * m * p  # 24
        
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        
        # Check if the system is consistent (rank of A_mat should be <= num_eqs)
        # and if a solution exists
        try:
            rank_A = np.linalg.matrix_rank(A_mat)
            if rank_A > num_eqs:
                continue
            
            # Solve for minimum-norm solution
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, rank)
        
        # Check if W is close to half-integers
        W_2x = W * 2
        W_rounded_2x = np.round(W_2x)
        diff = np.abs(W_2x - W_rounded_2x)
        
        if np.max(diff) > 0.1:
            continue
        
        W_rounded = W_rounded_2x / 2
        
        # Verify the decomposition
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank20_dense(n, m, p, time_budget=120.0):
    """
    Try a denser random search for rank-20.
    """
    start_time = time.time()
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    np.random.seed(123)
    
    for attempt in range(30000):
        if time.time() - start_time > time_budget:
            break
        
        rank = 20
        
        # Generate U and V with entries in {-1, -0.5, 0, 0.5, 1}
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = np.random.choice(vals, size=(n_m, rank))
        V = np.random.choice(vals, size=(m_p, rank))
        
        # Build the linear system for W
        num_vars = p_n * rank
        num_eqs = n * m * p  # 24
        
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, rank)
        W_2x = W * 2
        W_rounded_2x = np.round(W_2x)
        diff = np.abs(W_2x - W_rounded_2x)
        
        if np.max(diff) > 0.1:
            continue
        
        W_rounded = W_rounded_2x / 2
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank19(n, m, p, time_budget=60.0):
    """Try to find a rank-19 decomposition."""
    start_time = time.time()
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    np.random.seed(456)
    
    for attempt in range(20000):
        if time.time() - start_time > time_budget:
            break
        
        rank = 19
        
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = np.random.choice(vals, size=(n_m, rank))
        V = np.random.choice(vals, size=(m_p, rank))
        
        num_vars = p_n * rank
        num_eqs = n * m * p
        
        A_mat = np.zeros((num_eqs, num_vars))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        var_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, rank)
        W_2x = W * 2
        W_rounded_2x = np.round(W_2x)
        diff = np.abs(W_2x - W_rounded_2x)
        
        if np.max(diff) > 0.1:
            continue
        
        W_rounded = W_rounded_2x / 2
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_rounded.tolist()
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    
    # Try rank 20 first (most likely to succeed)
    remaining = deadline - time.time()
    if remaining > 5:
        result = _try_rank20_structured(n, m, p, time_budget=min(50, remaining - 5))
        if result is not None:
            return result
    
    remaining = deadline - time.time()
    if remaining > 5:
        result = _try_rank20_dense(n, m, p, time_budget=min(40, remaining - 5))
        if result is not None:
            return result
    
    remaining = deadline - time.time()
    if remaining > 5:
        result = _try_rank19(n, m, p, time_budget=min(30, remaining - 5))
        if result is not None:
            return result
    
    # Fallback: trivial rank-24 decomposition
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    
    # Try decreasing ranks, starting from lower to higher
    # The trivial rank is 24. We try 19, 20, 21, 22, 23 in order.
    # If we find a lower rank, we return it immediately.
    
    ranks_to_try = [19, 20, 21, 22, 23]
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021]
    
    for rank in ranks_to_try:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time budget for this rank
        time_budget = min(30, remaining - 5)
        
        # Try multiple seeds for this rank
        for seed in seeds:
            remaining = deadline - time.time()
            if remaining < 5:
                break
            
            result = _try_rank_r(n, m, p, rank, seed, time_budget=min(10, remaining - 3))
            if result is not None:
                return result
    
    # Fallback: trivial rank-24 decomposition
    return _build_trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
