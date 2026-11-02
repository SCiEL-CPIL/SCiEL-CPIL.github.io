import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check if entries are half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _find_solution_for_rank(rank, time_limit=10.0, max_restarts=200):
    """
    Try to find a valid decomposition of the given rank by fixing U and V randomly
    and solving for W using the block-diagonal structure of the linear system.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        rng = np.random.RandomState(restart + 999)
        
        # Generate random sparse integer U and V
        # Entries in {-1, 0, 1}
        U = rng.choice([-1, 0, 1], size=(n * m, rank))
        V = rng.choice([-1, 0, 1], size=(m * p, rank))
        
        W = np.zeros((p * n, rank), dtype=float)
        valid = True
        
        # The system decouples for each (i, k) pair
        # For fixed i, k, the variables are W[k*n + i, 0:rank]
        # The equations are for j = 0..m-1:
        # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
        # Let A_block[j, r] = U[i*m + j, r] * V[j*p + k, r]
        # We need A_block @ w_vec = 1_vec
        # To ensure w_vec are half-integers, we solve A_block @ v_vec = 2 * 1_vec for integer v_vec
        # and then w_vec = v_vec / 2.0
        
        for i in range(n):
            for k in range(p):
                # Construct A_block (m x rank)
                A_block = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A_block[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                # We need to find integer v such that A_block @ v = 2 * ones(m)
                # This is a linear Diophantine equation.
                # Since rank > m (usually), we can try to find a solution by picking m columns.
                # Or use least squares and round.
                
                # Try to find a sparse integer solution.
                # Strategy: Pick m columns that form an invertible matrix.
                # If det is +/- 1 or +/- 2, solution is integer or half-integer (scaled).
                # We want A_sub @ v_sub = 2 * ones(m).
                # v_sub = A_sub^-1 @ 2 * ones(m).
                # If det(A_sub) divides 2 * det(adj(A_sub) @ ones(m)), it's integer.
                
                found = False
                # Try all combinations of m columns from rank
                # rank is small (e.g., 19-30), m=3. Combinations C(rank, 3) is manageable.
                # However, doing this for every (i,k) in every restart might be slow.
                # Optimization: Only try a few random subsets of columns.
                
                # First, check if a trivial solution exists (e.g. one column is [2,0,0] etc)
                # Just use least squares as a heuristic to find a candidate, then verify.
                
                # Solve A_block @ v = 2 * ones(m) using lstsq
                rhs = 2.0 * np.ones(m)
                try:
                    v_sol, residuals, rank_A, s = np.linalg.lstsq(A_block, rhs, rcond=None)
                    # Check if v_sol is close to integers
                    v_rounded = np.round(v_sol)
                    if np.allclose(A_block @ v_rounded, rhs, atol=1e-9):
                        W[k * n + i, :] = v_rounded / 2.0
                        found = True
                    else:
                        # Try to find a sparse solution by zeroing out most variables
                        # This is harder. Let's just fail this restart if lstsq doesn't give integer.
                        # Or try to perturb U/V? No, U/V are fixed.
                        # We can try to solve for a subset of variables.
                        # If rank > m, we can set rank-m variables to 0.
                        # Let's try to find a subset of m columns that is invertible.
                        found = False
                        # Get indices of columns
                        cols = np.arange(rank)
                        # Shuffle to try different subsets
                        rng.shuffle(cols)
                        
                        # Try first few subsets
                        for start in range(0, rank - m + 1, 2): # Step by 2 to save time
                            subset_cols = cols[start:start+m]
                            A_sub = A_block[:, subset_cols]
                            det = np.linalg.det(A_sub)
                            if abs(det) > 1e-9:
                                # Solve A_sub @ v_sub = rhs
                                try:
                                    v_sub = np.linalg.solve(A_sub, rhs)
                                    v_sub_rounded = np.round(v_sub)
                                    if np.allclose(A_sub @ v_sub_rounded, rhs, atol=1e-9):
                                        v_full = np.zeros(rank)
                                        v_full[subset_cols] = v_sub_rounded
                                        W[k * n + i, :] = v_full / 2.0
                                        found = True
                                        break
                                except np.linalg.LinAlgError:
                                    continue
                        if not found:
                            valid = False
                            break
                except Exception:
                    valid = False
                    break
                    
                if not found:
                    valid = False
                    break
                    
            if not valid:
                break
        
        if valid:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Final verification
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Leave 5s margin
    
    # Try to find lower ranks
    # Start from 19 (theoretical min for 3x3x3 is 19, so 3x3x6 should be >= 19? 
    # Actually, 3x3x6 contains 3x3x3, so rank >= 19.
    # But 3x3x6 is two 3x3x3s. If they share structure, rank could be lower than 38.
    # Let's try from 19 upwards.
    
    for rank in range(19, 54):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            break
            
        # Allocate time based on rank. Lower ranks are harder? 
        # Actually, higher ranks are easier to find (more degrees of freedom).
        # But we want the MINIMAL rank. So we must check from low to high.
        # Give more time to lower ranks? Or just fixed time.
        # Let's give 2 seconds per rank attempt.
        result = _find_solution_for_rank(rank, time_limit=min(2.0, remaining_time - 1.0), max_restarts=100)
        
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook()
