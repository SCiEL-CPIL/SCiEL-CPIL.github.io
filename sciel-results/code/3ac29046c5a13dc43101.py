"""Combined program: rank search for 3x3x6 matmul via shared-U block decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are within tol of half-integers. Return rounded half-integers if valid."""
    doubled = np.asarray(arr, dtype=float) * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n x m x p matmul tensor."""
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i * m + j, j * p + k, k * n + i] = 1.0
    
    return np.allclose(T, expected, atol=tol)

def _search_3x3x3_structured(rank, max_attempts=20000, deadline=None, seed=None):
    """Search for a rank-R decomposition of 3x3x3 matmul tensor using structured approach.
    
    Key insight: For 3x3x3, we need U(9xR), V(9xR), W(9xR) such that
    sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k.
    
    We use a smarter search: fix U to have a structured pattern and solve for V, W.
    """
    n, m, p = 3, 3, 3
    dim = n * m  # 9
    
    if seed is not None:
        np.random.seed(seed)
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Strategy: Use a structured U with small integer entries
        # Try different patterns
        pattern = attempt % 5
        
        if pattern == 0:
            # Sparse random
            density = 0.3 + 0.1 * (attempt % 4)
            U = (np.random.rand(dim, rank) < density) * np.random.randint(-1, 2, size=(dim, rank))
        elif pattern == 1:
            # Dense random with small values
            U = np.random.randint(-1, 2, size=(dim, rank))
        elif pattern == 2:
            # Structured: first rank//3 columns have 1s in specific rows
            U = np.zeros((dim, rank))
            for r in range(rank):
                row = (r * 3) % dim
                U[row, r] = 1
                if r + 1 < rank:
                    U[(row + 1) % dim, r] = 1
        elif pattern == 3:
            # Binary with balanced density
            U = np.random.randint(0, 2, size=(dim, rank))
            # Ensure each column has some non-zero
            for r in range(rank):
                if U[:, r].sum() == 0:
                    U[np.random.randint(0, dim), r] = 1
        else:
            # Mixed: some 1s and -1s
            U = np.random.choice([-1, 0, 1], size=(dim, rank), p=[0.3, 0.4, 0.3])
        
        # For each (i,k) pair, we need to solve for W[k*3+i, :]
        # The constraint is: sum_j U[i*3+j,:] * V[j*3+k,:] * W[k*3+i,:] = 1
        # This is a linear system in W for fixed U, V
        
        # But V is also unknown. Let's try a different approach:
        # Fix U, and for each k, solve for V[:,k_block] and W[k_block,:]
        
        # Actually, let's use the fact that for 3x3x3, the tensor has 27 nonzero entries.
        # We can set up a system of equations.
        
        # Alternative: use the "slice" approach
        # For each k (0,1,2), the k-th slice of the tensor is:
        # T[:,:,k] where T[i,j,k] = sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r]
        # This should equal 1 for all i,j.
        
        # Let's try: fix U, then for each k, we need V[:, k*3:(k+1)*3] and W[k*3:(k+1)*3, :]
        # such that sum_j U[i*3+j,:] * V[j*3+k,:] * W[k*3+i,:] = 1
        
        # This is a bilinear system. Let's linearize by fixing one and solving for the other.
        
        # Strategy: Fix U. For each k, let A_k = U[:, :] (9xR), 
        # B_k = V[:, k*3:(k+1)*3] (9x3), C_k = W[k*3:(k+1)*3, :] (3xR)
        # We need: for all i,j: sum_r sum_s U[i*3+j, r] * B_k[j*3+k - k*3, r] * C_k[i, r] = 1
        # Wait, this is getting complicated. Let me use a simpler approach.
        
        # Simplest: fix U and V, solve for W.
        V = (np.random.rand(dim, rank) < 0.5) * np.random.randint(-1, 2, size=(dim, rank))
        
        W = np.zeros((dim, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # M[j, r] = U[i*3+j, r] * V[j*3+k, r]
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W[k * n + i, :] = w_check
        
        if valid:
            # Full verification
            if _verify_full(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _search_3x3x3_with_shared_U(rank, max_attempts=50000, deadline=None):
    """Search for 3x3x3 decomposition, returning (U, V, W) as lists."""
    result = _search_3x3x3_structured(rank, max_attempts=max_attempts, deadline=deadline)
    if result is not None:
        return result
    
    # Try with different random seeds
    for seed in range(10):
        if deadline is not None and time.time() > deadline:
            break
        result = _search_3x3x3_structured(rank, max_attempts=5000, deadline=deadline, seed=seed)
        if result is not None:
            return result
    
    return None

def _build_3x3x6_from_3x3x3(U3, V3, W3):
    """Build 3x3x6 decomposition from 3x3x3 decomposition.
    
    Key insight: U can be SHARED between both blocks since A is the same.
    Total rank = rank_333 (not 2 * rank_333).
    
    For block 1 (k=0,1,2): V uses columns 0-2 of B, W uses rows 0-8 of C
    For block 2 (k=3,4,5): V uses columns 3-5 of B, W uses rows 9-17 of C
    
    U_total = U3 (same for both blocks, so rank stays the same)
    V_total: 18 x rank, where rows 0-8 correspond to B cols 0-2, rows 9-17 to B cols 3-5
    W_total: 18 x rank, where rows 0-8 correspond to C rows 0-8, rows 9-17 to C rows 9-17
    """
    rank = len(U3[0])
    n, m, p = 3, 3, 6
    
    U_total = U3[:]  # Shared, shape 9 x rank
    
    V_total = [[0] * rank for _ in range(m * p)]  # 18 x rank
    W_total = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    # Block 1: k_local = 0,1,2 -> k_global = 0,1,2
    for j in range(m):
        for k_local in range(3):
            k_global = k_local
            for r in range(rank):
                V_total[j * p + k_global][r] = V3[j * 3 + k_local][r]
    
    for k_local in range(3):
        for i in range(n):
            for r in range(rank):
                W_total[k_local * n + i][r] = W3[k_local * 3 + i][r]
    
    # Block 2: k_local = 0,1,2 -> k_global = 3,4,5
    for j in range(m):
        for k_local in range(3):
            k_global = 3 + k_local
            for r in range(rank):
                V_total[j * p + k_global][r] = V3[j * 3 + k_local][r]
    
    for k_local in range(3):
        for i in range(n):
            for r in range(rank):
                W_total[(3 + k_local) * n + i][r] = W3[k_local * 3 + i][r]
    
    return U_total, V_total, W_total

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Strategy: Find 3x3x3 decomposition, then build 3x3x6 with shared U
    # Total rank = rank_333 (shared U)
    
    # Try ranks for 3x3x3. Known: schoolbook is 27. Lower bounds exist.
    # Let's try from low to high.
    for rank_333 in range(19, 28):
        if time.time() > deadline:
            break
        if rank_333 >= best_rank:
            break
        
        result = _search_3x3x3_with_shared_U(rank_333, max_attempts=20000, deadline=deadline)
        if result is not None:
            U3, V3, W3 = result
            U_total, V_total, W_total = _build_3x3x6_from_3x3x3(U3, V3, W3)
            
            if _verify_full(U_total, V_total, W_total, n, m, p):
                if rank_333 < best_rank:
                    best_rank = rank_333
                    best_result = {"U": U_total, "V": V_total, "W": W_total}
                break
    
    # If no 3x3x3 solution found, try direct 3x3x6 search
    if best_result is None:
        for rank in [30, 36, 40, 42, 44, 46, 48, 50, 52]:
            if time.time() > deadline:
                break
            if rank >= best_rank:
                break
            
            # Direct search for 3x3x6
            found = False
            for attempt in range(2000):
                if time.time() > deadline:
                    break
                
                density = 0.4
                U_rand = (np.random.rand(n * m, rank) < density) * np.random.randint(-1, 2, size=(n * m, rank))
                V_rand = (np.random.rand(m * p, rank) < density) * np.random.randint(-1, 2, size=(m * p, rank))
                
                W_sol = np.zeros((p * n, rank))
                valid = True
                
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                        
                        try:
                            w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        except:
                            valid = False
                            break
                        
                        w_check = _check_half_integer(w_vec)
                        if w_check is None:
                            valid = False
                            break
                        W_sol[k * n + i, :] = w_check
                
                if valid:
                    if _verify_full(U_rand, V_rand, W_sol, n, m, p):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_sol.tolist()}
                        found = True
                        break
            
            if found:
                break
    
    # Return best result or schoolbook fallback
    if best_result is not None:
        return best_result
    
    # Fallback: Schoolbook rank 54
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
