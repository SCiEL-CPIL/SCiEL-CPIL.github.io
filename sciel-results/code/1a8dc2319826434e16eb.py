# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a rank-minimal exact decomposition for 3x3 matrix multiplication.
    Strategy:
    1. Start with the schoolbook rank-27 decomposition.
    2. Try to remove terms (increase sparsity) to lower rank.
    3. Fix U and V for the remaining terms (from schoolbook).
    4. Solve the linear system for W such that the tensor decomposition holds.
    5. If exact solution exists with integer/half-integer entries, return it.
    6. Otherwise, fallback to schoolbook rank-27.
    """
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit

    # 1. Generate schoolbook 27 terms
    U_school = np.zeros((9, 27))
    V_school = np.zeros((9, 27))
    W_school = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_school[i * 3 + j, col] = 1
                V_school[j * 3 + k, col] = 1
                W_school[k * 3 + i, col] = 1
                col += 1
                
    # Check if we have time to try rank reduction
    if time.time() - start_time > deadline - 10:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try ranks from 23 down to 20 (known lower bound is 19, but 23 is standard)
    # We will try to find a valid decomposition for each rank
    # The approach: Fix U and V from schoolbook, but zero out some columns to reduce rank?
    # No, that changes the subspace. The current program's approach was:
    # "Fix U and V for the remaining 23 terms (from schoolbook)"
    # This implies taking the first 23 columns of U_school and V_school.
    # This is a heuristic. Let's try it for various subsets if possible, 
    # but the prompt says "Keep this design". The design is:
    # 1. Start with schoolbook.
    # 2. Remove 4 terms (keep first 23).
    # 3. Fix U and V for remaining.
    # 4. Solve for W.
    
    # The "improvement" within this family might be trying different subsets of 23 terms
    # or verifying if the solution W has nice properties (integers/half-integers).
    # The task requires integer/half-integer entries.
    # The schoolbook solution has integer entries.
    # The solved W might be fractional. We need to check if they are half-integers (multiples of 0.5).
    
    def verify_and_clean(U, V, W, tol=1e-9):
        """
        Verify decomposition and check if entries are integers or half-integers.
        Returns True if valid and clean, False otherwise.
        """
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
        V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
        W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
        
        # Check integer/half-integer constraint
        # Multiply by 2 and check if close to integer
        W_scaled = W_arr * 2
        if not np.allclose(W_scaled, np.round(W_scaled), atol=tol):
            return False
        U_scaled = U_arr * 2
        if not np.allclose(U_scaled, np.round(U_scaled), atol=tol):
            return False
        V_scaled = V_arr * 2
        if not np.allclose(V_scaled, np.round(V_scaled), atol=tol):
            return False
            
        # Verify tensor decomposition
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * 3 + j, r] * V_arr[j * 3 + k, r] * W_arr[k * 3 + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try rank 23 first (as in current program)
    rank_target = 23
    U = U_school[:, :rank_target]
    V = V_school[:, :rank_target]
    
    # Construct linear system for W
    num_vars = 9 * rank_target
    A_mat = np.zeros((27, num_vars))
    b_vec = np.ones(27)
    
    eq_idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for r in range(rank_target):
                    var_idx = (k * 3 + i) * rank_target + r
                    A_mat[eq_idx, var_idx] = U[i * 3 + j, r] * V[j * 3 + k, r]
                eq_idx += 1
                
    # Solve for W
    try:
        W_flat, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W = W_flat.reshape((9, rank_target))
        
        if verify_and_clean(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # If rank 23 with first 23 terms failed (likely due to non-integer W or inconsistency),
    # try other subsets of 23 terms?
    # The current program just fell back to schoolbook.
    # Let's try a few random subsets of 23 terms to see if we get a better W.
    # This is a "refinement" within the family.
    
    import itertools
    # 27 choose 23 is 27 choose 4 = 17550. Too many to brute force quickly if each is expensive?
    # lstsq on 27x207 is fast. 17550 * fast is probably fine within 120s.
    # But let's be careful.
    
    # Let's try a limited number of random subsets if time permits.
    np.random.seed(42)
    indices = list(range(27))
    attempts = 0
    max_attempts = 500  # Safety
    
    while attempts < max_attempts and time.time() - start_time < deadline - 5:
        # Pick 23 random indices
        subset = np.random.choice(27, 23, replace=False)
        subset.sort()
        
        U_sub = U_school[:, subset]
        V_sub = V_school[:, subset]
        
        A_mat_sub = np.zeros((27, 23 * 9))
        eq_idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for r in range(23):
                        var_idx = (k * 3 + i) * 23 + r
                        A_mat_sub[eq_idx, var_idx] = U_sub[i * 3 + j, r] * V_sub[j * 3 + k, r]
                    eq_idx += 1
        
        try:
            W_flat_sub, _, _, _ = np.linalg.lstsq(A_mat_sub, b_vec, rcond=None)
            W_sub = W_flat_sub.reshape((9, 23))
            
            if verify_and_clean(U_sub, V_sub, W_sub):
                return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
        except Exception:
            pass
        attempts += 1

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
