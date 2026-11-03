import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def make_schoolbook(rank):
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
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def check_half_integer(mat, tol=1e-6):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > tol):
            return False
        return True
    
    def round_to_half_integer(mat, tol=1e-4):
        """Round entries to nearest half-integer if close enough."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > tol):
            return None
        return (rounded / 2.0).tolist()
    
    def is_valid_solution(U, V, W, rank):
        """Full validation: exact decomposition + half-integer entries."""
        if not verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
            return False
        if not check_half_integer(U.tolist()):
            return False
        if not check_half_integer(V.tolist()):
            return False
        if not check_half_integer(W.tolist()):
            return False
        return True
    
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    rng = np.random.RandomState(42)
    
    # Pre-compute the structure for the linear system
    # For fixed U,V, the system is: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,rank)
    
    def build_A_matrix(U, V, rank):
        """Build the coefficient matrix for the linear system in W."""
        num_eqs = n * m * p
        num_unk = p * n * rank
        
        A = np.zeros((num_eqs, num_unk))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r_idx in range(rank):
                        col = (k * n + i) * rank + r_idx
                        A[eq_idx, col] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
        
        return A
    
    def solve_for_W(U, V, rank):
        """Given U and V, solve for W using the linear system."""
        A = build_A_matrix(U, V, rank)
        b = np.ones(n * m * p)
        
        try:
            w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            return None
        
        # Check if solution is exact
        if residuals.size > 0:
            if residuals[0] > 1e-8:
                return None
        
        w_vec = w_vec.reshape(p * n, rank)
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r_idx in range(rank):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * w_vec[k * n + i, r_idx]
                    if abs(val - 1) > 1e-8:
                        return None
        
        return w_vec
    
    # Strategy: Use "partial trace" / slice-sum duality to reduce the problem.
    # Core idea: The 3x3x5 multiplication tensor can be viewed as extending a 3x3x3 core.
    # The extra output dimensions (k=3,4) act as linear functionals.
    # We search for a rank-R decomposition where the first 3 output slices (k=0,1,2)
    # form a valid 3x3x3 decomposition, and the remaining slices are consistent.
    
    def try_core_extension(target_rank):
        """
        Try to find a decomposition by first solving the 3x3x3 core (k=0,1,2)
        and then extending to k=3,4.
        
        The 3x3x3 core has rank at least 21 (Strassen's algorithm for 3x3).
        We try to find a rank-R decomposition for the full 3x3x5 problem
        where the U and V factors are shared across all k slices.
        """
        if time.time() > deadline - 20:
            return None
        
        # Generate candidate U and V patterns focused on the core structure
        candidates = []
        
        # Pattern type 1: Core-focused sparse patterns
        # The 3x3x3 core requires U[i*m+j, r] and V[j*p+k, r] for k=0,1,2
        # We generate patterns that are dense in the core region
        for trial in range(80):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for r_idx in range(target_rank):
                # U: pick 2-3 rows, biased toward core
                u_count = rng.randint(2, 4)
                # Bias toward rows that correspond to core j indices
                u_rows = rng.choice(n * m, size=u_count, replace=False)
                for row in u_rows:
                    U[int(row), r_idx] = rng.choice([-1, 1])
                
                # V: pick 2-3 rows, biased toward core k indices (k=0,1,2)
                v_count = rng.randint(2, 4)
                # Bias: 70% chance to pick from core k rows (j*p+0, j*p+1, j*p+2)
                core_rows = [j * p + k for j in range(m) for k in range(3)]
                if rng.random() < 0.7:
                    v_rows = rng.choice(core_rows, size=v_count, replace=False)
                else:
                    v_rows = rng.choice(m * p, size=v_count, replace=False)
                for row in v_rows:
                    V[int(row), r_idx] = rng.choice([-1, 1])
            
            candidates.append((U, V))
        
        # Pattern type 2: Block-structured with core emphasis
        for trial in range(50):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for r_idx in range(target_rank):
                # U: focus on a specific i or j
                focus_i = r_idx % n
                focus_j = (r_idx // n) % m
                U[focus_i * m + focus_j, r_idx] = 1
                # Add some randomness
                if rng.random() < 0.3:
                    extra_row = rng.randint(n * m)
                    if extra_row != focus_i * m + focus_j:
                        U[extra_row, r_idx] = rng.choice([-1, 1])
                
                # V: focus on a specific j or k, biased to core
                focus_j2 = r_idx % m
                if rng.random() < 0.6:
                    focus_k = rng.randint(3)  # Core k
                else:
                    focus_k = rng.randint(p)
                V[focus_j2 * p + focus_k, r_idx] = 1
                if rng.random() < 0.3:
                    extra_row = rng.randint(m * p)
                    if extra_row != focus_j2 * p + focus_k:
                        V[extra_row, r_idx] = rng.choice([-1, 1])
            
            candidates.append((U, V))
        
        # Pattern type 3: Strassen-inspired for the core
        # Strassen's algorithm for 3x3 uses 19 multiplications
        # We adapt this idea by creating patterns that mimic the Strassen structure
        for trial in range(40):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for r_idx in range(target_rank):
                # Create a pattern based on Strassen-like combinations
                i_idx = r_idx % n
                j_idx = (r_idx // n) % m
                
                # U: combination of 2-3 elements in row i_idx
                num_u = rng.randint(2, 4)
                u_cols = rng.choice(m, size=num_u, replace=False)
                for col in u_cols:
                    U[i_idx * m + col, r_idx] = rng.choice([-1, 1])
                
                # V: combination of 2-3 elements in column j_idx, biased to core
                num_v = rng.randint(2, 4)
                if rng.random() < 0.6:
                    v_rows = rng.choice(3, size=num_v, replace=False)  # Core k
                else:
                    v_rows = rng.choice(p, size=num_v, replace=False)
                for row in v_rows:
                    V[j_idx * p + row, r_idx] = rng.choice([-1, 1])
            
            candidates.append((U, V))
        
        # Try each candidate
        for U, V in candidates:
            if time.time() > deadline - 15:
                break
            
            W = solve_for_W(U, V, target_rank)
            if W is None:
                continue
            
            # Check if W is close to half-integers
            W_half = round_to_half_integer(W.tolist(), tol=1e-3)
            if W_half is None:
                continue
            
            # Also check U and V are half-integers (they should be by construction)
            if not check_half_integer(U.tolist()):
                continue
            if not check_half_integer(V.tolist()):
                continue
            
            # Final verification with rounded W
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_half
            
            if is_valid_solution(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Strategy 2: Direct search with improved patterns
    def try_structured_decomposition(target_rank):
        """Try to find a decomposition at the given rank using structured patterns."""
        if time.time() > deadline - 20:
            return None
        
        # Generate candidate U and V patterns
        candidates = []
        
        # Pattern type 1: Sparse with 2-3 nonzeros per column
        for trial in range(50):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for r_idx in range(target_rank):
                # U: pick 2-3 rows
                u_count = rng.randint(2, 4)
                u_rows = rng.choice(n * m, size=u_count, replace=False)
                for row in u_rows:
                    U[int(row), r_idx] = rng.choice([-1, 1])
                
                # V: pick 2-3 rows
                v_count = rng.randint(2, 4)
                v_rows = rng.choice(m * p, size=v_count, replace=False)
                for row in v_rows:
                    V[int(row), r_idx] = rng.choice([-1, 1])
            
            candidates.append((U, V))
        
        # Pattern type 2: Block-structured
        for trial in range(30):
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for r_idx in range(target_rank):
                # U: focus on a specific i or j
                focus_i = r_idx % n
                focus_j = (r_idx // n) % m
                U[focus_i * m + focus_j, r_idx] = 1
                # Add some randomness
                if rng.random() < 0.3:
                    extra_row = rng.randint(n * m)
                    if extra_row != focus_i * m + focus_j:
                        U[extra_row, r_idx] = rng.choice([-1, 1])
                
                # V: focus on a specific j or k
                focus_j2 = r_idx % m
                focus_k = (r_idx // m) % p
                V[focus_j2 * p + focus_k, r_idx] = 1
                if rng.random() < 0.3:
                    extra_row = rng.randint(m * p)
                    if extra_row != focus_j2 * p + focus_k:
                        V[extra_row, r_idx] = rng.choice([-1, 1])
            
            candidates.append((U, V))
        
        # Try each candidate
        for U, V in candidates:
            if time.time() > deadline - 15:
                break
            
            W = solve_for_W(U, V, target_rank)
            if W is None:
                continue
            
            # Check if W is close to half-integers
            W_half = round_to_half_integer(W.tolist(), tol=1e-3)
            if W_half is None:
                continue
            
            # Also check U and V are half-integers (they should be by construction)
            if not check_half_integer(U.tolist()):
                continue
            if not check_half_integer(V.tolist()):
                continue
            
            # Final verification with rounded W
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_half
            
            if is_valid_solution(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 44 down to 36
    for target_rank in range(rank_schoolbook - 1, 35, -1):
        if time.time() > deadline - 25:
            break
        
        # First try the core extension strategy
        result = try_core_extension(target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
        
        # Then try the general structured decomposition
        result = try_structured_decomposition(target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
