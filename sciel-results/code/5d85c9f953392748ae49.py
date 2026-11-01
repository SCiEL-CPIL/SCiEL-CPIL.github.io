# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use vectorized NumPy local search minimizing error count with smooth surrogate.
    3. Multiple random restarts with different sparse initializations.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Active tensor: T[a,b,c] = 1 if (a,b,c) is an active triple
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def local_search_np(U, V, W, r, max_iters=200000, time_limit=3.0):
        """
        Vectorized local search minimizing error count.
        U, V, W are numpy arrays of shape (nm, r), (mp, r), (pn, r)
        """
        start_ls = time.time()
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Compute sums: S[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # Use einsum for efficiency
        S = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Error: E = S - T
        E = S - T
        curr_err = int(np.sum(E != 0))
        best_err = curr_err
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        if curr_err == 0:
            return U, V, W, 0
        
        deltas = np.array([-2, -1, 1, 2], dtype=int)
        
        for it in range(max_iters):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            mat_idx = random.randint(0, 2)
            
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                col = random.randint(0, r - 1)
                d = random.choice(deltas)
                new_val = U[row, col] + d
                if abs(new_val) > 15:
                    continue
                
                # Delta contribution: d * V[:, col] * W[:, col] for all (b,c)
                delta_S = d * np.outer(V[:, col], W[:, col])  # shape (mp, pn)
                
                # Current error for affected slice
                old_E = E[row, :, :]
                new_E = old_E + delta_S
                
                old_err_slice = np.sum(old_E != 0)
                new_err_slice = np.sum(new_E != 0)
                
                # Smooth surrogate: sum of squared errors
                old_sq = np.sum(old_E ** 2)
                new_sq = np.sum(new_E ** 2)
                
                improvement = old_err_slice - new_err_slice
                sq_improvement = old_sq - new_sq
                
                # Accept if error count improves, or if count same but sq error improves
                if improvement > 0 or (improvement == 0 and sq_improvement > 0):
                    U[row, col] = new_val
                    E[row, :, :] = new_E
                    S[row, :, :] += delta_S
                    curr_err += -improvement  # curr_err decreases by improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                # else: revert (do nothing)
                
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                col = random.randint(0, r - 1)
                d = random.choice(deltas)
                new_val = V[row, col] + d
                if abs(new_val) > 15:
                    continue
                
                # Delta contribution: U[:, col] * d * W[:, col] for all (a,c)
                delta_S = d * np.outer(U[:, col], W[:, col])  # shape (nm, pn)
                
                # E is (nm, mp, pn), affected slice is E[:, row, :]
                old_E = E[:, row, :].copy()
                new_E = old_E + delta_S
                
                old_err_slice = np.sum(old_E != 0)
                new_err_slice = np.sum(new_E != 0)
                
                old_sq = np.sum(old_E ** 2)
                new_sq = np.sum(new_E ** 2)
                
                improvement = old_err_slice - new_err_slice
                sq_improvement = old_sq - new_sq
                
                if improvement > 0 or (improvement == 0 and sq_improvement > 0):
                    V[row, col] = new_val
                    E[:, row, :] = new_E
                    S[:, row, :] += delta_S
                    curr_err += -improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, r - 1)
                d = random.choice(deltas)
                new_val = W[row, col] + d
                if abs(new_val) > 15:
                    continue
                
                # Delta contribution: U[:, col] * V[:, col] * d for all (a,b)
                delta_S = d * np.outer(U[:, col], V[:, col])  # shape (nm, mp)
                
                # E is (nm, mp, pn), affected slice is E[:, :, row]
                old_E = E[:, :, row].copy()
                new_E = old_E + delta_S
                
                old_err_slice = np.sum(old_E != 0)
                new_err_slice = np.sum(new_E != 0)
                
                old_sq = np.sum(old_E ** 2)
                new_sq = np.sum(new_E ** 2)
                
                improvement = old_err_slice - new_err_slice
                sq_improvement = old_sq - new_sq
                
                if improvement > 0 or (improvement == 0 and sq_improvement > 0):
                    W[row, col] = new_val
                    E[:, :, row] = new_E
                    S[:, :, row] += delta_S
                    curr_err += -improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
        
        return best_U, best_V, best_W, best_err
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank(r, max_attempts=200):
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                return None
            
            # Sparse random initialization
            U = np.zeros((nm, r), dtype=int)
            V = np.zeros((mp, r), dtype=int)
            W = np.zeros((pn, r), dtype=int)
            
            for a in range(nm):
                k = random.randint(1, min(3, r))
                cols = random.sample(range(r), k)
                for c in cols:
                    U[a, c] = random.choice([-1, 1])
            for b in range(mp):
                k = random.randint(1, min(3, r))
                cols = random.sample(range(r), k)
                for c in cols:
                    V[b, c] = random.choice([-1, 1])
            for c in range(pn):
                k = random.randint(1, min(3, r))
                cols = random.sample(range(r), k)
                for c2 in cols:
                    W[c, c2] = random.choice([-1, 1])
            
            # Local search with adaptive time limit
            remaining = deadline - time.time() - 2
            if remaining < 0.5:
                return None
            time_per_attempt = min(1.5, remaining / max(1, max_attempts - attempt))
            U, V, W, err = local_search_np(U, V, W, r, max_iters=50000, time_limit=time_per_attempt)
            
            if err == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from low to high
    for r in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
