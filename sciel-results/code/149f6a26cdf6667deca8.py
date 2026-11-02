import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W form a valid tensor decomposition of the n x m x p matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Trace convention: W[k*n + i, r]
    """
    T_decomp = np.zeros((n, m, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        # Vectorized update: T_decomp += outer(u_2d[i,j] * v_2d[j,k] * w_2d[k,i])
        # This is a 3D tensor product, but we can do it via broadcasting or loops.
        # Given small sizes, a loop is fine and clear.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T_decomp, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook():
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_rank_reduction(U_init, V_init, W_init, n, m, p, target_rank, max_iters=100, tol=1e-6):
    """
    Try to reduce rank by iteratively removing columns that are linear combinations
    of others, using a greedy approach.
    """
    U = np.array(U_init, dtype=float).copy()
    V = np.array(V_init, dtype=float).copy()
    W = np.array(W_init, dtype=float).copy()
    
    current_rank = U.shape[1]
    
    for iteration in range(max_iters):
        if current_rank <= target_rank:
            break
        
        # Try removing each column and check if the decomposition still holds
        best_remove = -1
        best_error = np.inf
        
        for r in range(current_rank):
            # Create masks to exclude column r
            mask = [c for c in range(current_rank) if c != r]
            
            if len(mask) < target_rank:
                continue
            
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            # Check if this sub-decomposition can still represent the full tensor
            # by checking if the removed column is in the span of the others
            # For simplicity, we'll just verify if the sub-decomposition works
            # (This is expensive but correct)
            
            T_decomp = np.zeros((n, m, p))
            for idx, r_sub in enumerate(mask):
                u_col = U[:, r_sub]
                v_col = V[:, r_sub]
                w_col = W[:, r_sub]
                
                u_2d = u_col.reshape(n, m)
                v_2d = v_col.reshape(m, p)
                w_2d = w_col.reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
            
            error = np.max(np.abs(T_decomp - 1.0))
            
            if error < best_error:
                best_error = error
                best_remove = r
        
        if best_remove >= 0 and best_error < tol:
            mask = [c for c in range(current_rank) if c != best_remove]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            current_rank -= 1
        else:
            break
    
    if current_rank <= target_rank:
        return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank], target_rank
    return None, None, None, current_rank

def _structured_search(n, m, p, max_rank_to_try, time_limit):
    """
    Try structured rank reductions using the algebraic properties of matmul.
    """
    start_time = time.time()
    
    # Start from schoolbook
    U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank incrementally
    for target_rank in range(rank - 1, max(0, rank - 20), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random perturbations of the schoolbook decomposition
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            # Create a slightly perturbed version
            U_pert = U.copy()
            V_pert = V.copy()
            W_pert = W.copy()
            
            # Add small random noise to encourage different linear dependencies
            noise_level = 1e-3
            U_pert += noise_level * np.random.randn(n * m, rank)
            V_pert += noise_level * np.random.randn(m * p, rank)
            W_pert += noise_level * np.random.randn(p * n, rank)
            
            # Try to reduce rank from this perturbed version
            U_reduced, V_reduced, W_reduced, new_rank = _try_rank_reduction(
                U_pert, V_pert, W_pert, n, m, p, target_rank, max_iters=50, tol=1e-5
            )
            
            if U_reduced is not None and new_rank < best_rank:
                # Verify the result
                if _verify(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), n, m, p, new_rank):
                    best_U, best_V, best_W, best_rank = U_reduced, V_reduced, W_reduced, new_rank
    
    return best_U, best_V, best_W, best_rank

def _random_search(n, m, p, deadline):
    """
    Random search for low-rank decompositions using half-integer entries.
    Inspired by inspiration 2.
    """
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Define a set of candidate values for entries
    # Focus on small integers and halves to keep "exact" feasibility high
    candidate_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try a range of ranks. 
    # Lower bound is at least max(n, m, p) but likely higher.
    # For 3x5x6, schoolbook is 90.
    # Let's try ranks from ~30 to 80.
    # We prioritize lower ranks.
    
    for r in range(30, 85):
        if time.time() > deadline - 5:
            break
        
        # Number of trials per rank
        # We can afford more trials for higher ranks if we have time, but let's be balanced.
        num_trials = 200
        if time.time() > deadline - 20:
            num_trials = 50
            
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Generate random factors
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Sparse random generation is often better for tensor rank problems
            # Let's try generating non-zero entries with some probability
            # Or just fully random from candidate_vals
            
            for i in range(n * m):
                for j in range(r):
                    U[i, j] = random.choice(candidate_vals)
            for i in range(m * p):
                for j in range(r):
                    V[i, j] = random.choice(candidate_vals)
            for i in range(p * n):
                for j in range(r):
                    W[i, j] = random.choice(candidate_vals)
            
            # Check decomposition
            # Efficient check:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We can vectorize this check.
            
            # Reshape for broadcasting? 
            # U is (n*m, r), V is (m*p, r), W is (p*n, r)
            # We want to compute for all i, j, k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Let's use a loop over i, j, k but vectorize over c
            # This is 3*5*6 = 90 checks. Each check is a dot product of length r.
            # This is very fast.
            
            valid = True
            for i in range(n):
                for j in range(m):
                    u_row = U[i*m + j]
                    for k in range(p):
                        v_row = V[j*p + k]
                        w_row = W[k*n + i]
                        # sum_c u_row[c] * v_row[c] * w_row[c]
                        s = np.dot(u_row * v_row, w_row)
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # If we find a low rank, we can stop or continue to find even lower
                # Since we iterate r from low to high, the first valid one is the best for this loop
                # But we might miss a lower rank if we started too high? No, we start at 30.
                # If we find one at r=40, we can break out of the inner loop and continue to next r?
                # No, we want the MINIMAL rank. So if we find one at r, we should check lower r's first.
                # But we are iterating r from 30 upwards. So the first success is the minimum found in this sequence.
                # We can break out of the r loop entirely if we find a solution?
                # Not necessarily, maybe r=29 works? But we started at 30.
                # So we should start from a lower bound.
                # Let's assume the minimum is >= 20.
                # If we find a solution at r, it's the best we've found so far.
                # We can continue searching for lower r? No, we are going up.
                # So if we find a solution at r, we can return it immediately?
                # Yes, because we are checking r in increasing order.
                return best_U, best_V, best_W, best_rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 3x5x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook (should always pass)
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
        pass
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Strategy 1: Random search for low ranks (from 30 to 80)
    # This is likely to find a better solution than the structured reduction from 90.
    U_rand, V_rand, W_rand, rank_rand = _random_search(n, m, p, deadline)
    
    if U_rand is not None and rank_rand < best_rank:
        # Verify random result
        if _verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p, rank_rand):
            best_U, best_V, best_W, best_rank = U_rand, V_rand, W_rand, rank_rand
    
    # Strategy 2: Structured search (if time remains)
    if time.time() < deadline - 20:
        U_struct, V_struct, W_struct, rank_struct = _structured_search(n, m, p, best_rank - 1, deadline - time.time() - 10)
        if U_struct is not None and rank_struct < best_rank:
            if _verify(U_struct.tolist(), V_struct.tolist(), W_struct.tolist(), n, m, p, rank_struct):
                best_U, best_V, best_W, best_rank = U_struct, V_struct, W_struct, rank_struct
    
    # Final verification
    if not _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, best_rank):
        # Fall back to schoolbook if verification fails
        U, V, W, rank = _schoolbook()
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
