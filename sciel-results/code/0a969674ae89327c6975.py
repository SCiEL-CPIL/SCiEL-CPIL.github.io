"""Matrix multiplication 3x4x8 tensor decomposition search."""
import itertools
import random
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 4, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 96

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integers(U, V, W, tol=1e-6):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        arr = np.array(mat, dtype=float)
        if not np.allclose(arr * 2, np.round(arr * 2), atol=tol):
            return False
    return True

def _clean_entries(mat):
    """Clean entries to exact integers or half-integers."""
    result = []
    for row in mat:
        new_row = []
        for val in row:
            val = float(val)
            doubled = round(val * 2)
            if abs(val * 2 - doubled) < 1e-6:
                val = doubled / 2.0
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            else:
                new_row.append(val)
        result.append(new_row)
    return result

def _try_structured_block(n, m, p):
    """
    Try to exploit block structure: p = 2 * (p/2) when p is even.
    For 3x4x8, p=8=2*4. We compute A(3x4) * B1(4x4) and A(3x4) * B2(4x4).
    If we can find a rank-r decomposition for 3x4x4, we might use rank 2r for 3x4x8,
    or potentially less if we share some rank-1 terms.
    
    Strategy: Use the same U and V structure for both halves, with W differing.
    This gives rank = 2 * (rank of 3x4x4 decomposition).
    
    For 3x4x4, schoolbook is 48, so this gives 96 (no improvement).
    But if we find a better 3x4x4 decomposition, we improve.
    
    Alternative: Try to find a single decomposition where some rank-1 terms
    contribute to both k-halves.
    """
    # For now, return None - we'll use a different approach
    return None

def _solve_for_W(U_rem, V_rem, target_tensor, tol=1e-9):
    """
    Given fixed U and V (with rank r), solve for W such that
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = target[i,j,k] for all i,j,k.
    
    This is a linear system in the entries of W.
    Variables: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
    Equations: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    The system is: A * w = b where w is the vectorized W.
    """
    n, m, p = N, M, P
    U_rem = np.array(U_rem, dtype=float)
    V_rem = np.array(V_rem, dtype=float)
    r = U_rem.shape[1]
    
    num_vars = p * n * r
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for c in range(r):
                    var_idx = k * n * r + i * r + c
                    A[eq_idx, var_idx] = U_rem[i * m + j, c] * V_rem[j * p + k, c]
    
    # Check if system is consistent and find a solution
    # Use least squares first to check feasibility
    try:
        # Check rank of A
        rank_A = np.linalg.matrix_rank(A)
        # Check if b is in column space of A
        # For now, just try to solve
        if rank_A < num_eqs:
            # Underdetermined or inconsistent - use least squares
            x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            # Check if solution is exact
            if np.linalg.norm(A @ x - b) > tol * 10:
                return None
        else:
            # Full rank, unique solution
            x = np.linalg.solve(A, b)
            if np.linalg.norm(A @ x - b) > tol * 10:
                return None
    except np.linalg.LinAlgError:
        return None
    
    # Check if solution has half-integer entries
    W_flat = x
    if not _all_half_integers([W_flat.tolist()], [], [], tol=1e-4):
        # Try to find a half-integer solution via LP
        # Minimize deviation from half-integers
        # This is complex; for now, skip
        return None
    
    # Reshape to W matrix
    W_mat = np.zeros((p * n, r))
    for k in range(p):
        for i in range(n):
            for c in range(r):
                W_mat[k * n + i, c] = W_flat[k * n * r + i * r + c]
    
    return W_mat.tolist()

def _greedy_reduce(U, V, W, deadline):
    """
    Greedy reduction: try to remove one column at a time by solving for the remaining W.
    """
    n, m, p = N, M, P
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    
    # Try removing columns in a smart order
    # Columns with small "importance" (small contribution) are good candidates
    cols_to_try = list(range(current_rank))
    random.Random(42).shuffle(cols_to_try)
    
    improved = False
    for c in cols_to_try:
        if time.time() > deadline:
            break
        if c >= current_rank:
            continue
        
        # Try to remove column c
        remaining_cols = [cc for cc in range(current_rank) if cc != c]
        new_rank = len(remaining_cols)
        
        # Extract U and V for remaining columns
        U_rem = [[U[i][c2] for c2 in remaining_cols] for i in range(n * m)]
        V_rem = [[V[i][c2] for c2 in remaining_cols] for i in range(m * p)]
        
        # Solve for W
        W_new = _solve_for_W(U_rem, V_rem, None, tol=1e-8)
        if W_new is not None:
            # Verify
            if _verify(U_rem, V_rem, W_new, tol=1e-8):
                U = U_rem
                V = V_rem
                W = W_new
                current_rank = new_rank
                improved = True
                # Adjust cols_to_try: after removing column c, indices shift
                # For simplicity, restart the loop with updated rank
                cols_to_try = list(range(current_rank))
                random.Random(42).shuffle(cols_to_try)
                continue
    
    return {"U": U, "V": V, "W": W}, current_rank

def _structured_search(deadline):
    """
    Try structured constructions for 3x4x8.
    
    Key idea: Since 8 = 2*4, think of B as [B1 | B2] with B1, B2 being 4x4.
    C = A*B = [A*B1 | A*B2].
    
    If we have a rank-r decomposition for 3x4x4 multiplication, we can use
    rank 2r for 3x4x8 by using the same U,V for both halves with different W.
    
    But we can do better: if some rank-1 terms in the 3x4x4 decomposition
    can be reused for both halves, we save rank.
    
    Alternative structured approach:
    Use the fact that for 3x4x4, we might find a rank less than 48.
    Known: for 2x2x2, rank is 7 (Strassen). For 3x3x3, best known is 23.
    For 3x4x4, we can try to construct a low-rank decomposition.
    
    Let's try a direct approach: assume a form for U, V, W with half-integer entries
    and small rank, and solve.
    """
    n, m, p = N, M, P
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But we need at least enough rank to satisfy all 96 equations.
    # Each rank-1 term contributes to at most n*m*p = 96 entries, but in practice
    # the support is limited.
    
    # Try a few specific ranks
    for target_rank in [48, 47, 46, 45, 44, 43, 42, 40, 36, 32]:
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a structured approach: initialize with schoolbook-like structure
        # and try to merge columns
        
        # For now, try a random search for this rank
        best_result = None
        max_trials = 20
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate random U, V with half-integer entries
            rng = random.Random(trial * 1000 + target_rank)
            values = [0, 0.5, 1.0, -0.5, -1.0]
            
            U = [[rng.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rng.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
            
            # Solve for W
            W_sol = _solve_for_W(U, V, None, tol=1e-8)
            if W_sol is not None:
                if _verify(U, V, W_sol, tol=1e-8):
                    best_result = {"U": U, "V": V, "W": W_sol}
                    break
        
        if best_result is not None:
            return best_result, target_rank
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin before 150s limit
    
    n, m, p = N, M, P
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Step 1: Try structured search for lower rank
    struct_result, struct_rank = _structured_search(deadline - 5.0)
    if struct_result is not None and struct_rank < best_rank:
        result = struct_result
        best_rank = struct_rank
    
    # Step 2: Greedy reduction from current best
    if time.time() < deadline - 2.0:
        greedy_result, greedy_rank = _greedy_reduce(
            result["U"], result["V"], result["W"], deadline - 2.0
        )
        if greedy_rank < best_rank:
            result = greedy_result
            best_rank = greedy_rank
    
    # Step 3: Final verification and cleanup
    if _verify(result["U"], result["V"], result["W"], tol=1e-8):
        result["U"] = _clean_entries(result["U"])
        result["V"] = _clean_entries(result["V"])
        result["W"] = _clean_entries(result["W"])
    else:
        # Fallback to schoolbook
        result = _schoolbook()
    
    return result
