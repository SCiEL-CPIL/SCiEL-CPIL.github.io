import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145
    
    def verify(rank, U, V, W):
        """Verify exact decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best = make_schoolbook()
    best_rank = n * m * p
    
    def get_error_count(rank, U, V, W):
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        count += 1
        return count
    
    def get_smooth_error(rank, U, V, W):
        """Sum of squared errors"""
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    total += (s - 1.0) ** 2
        return total
    
    def local_search_smooth(rank, U, V, W, seed, max_steps=200):
        """Local search using smooth error as objective, then verify with exact count"""
        rng = random.Random(seed)
        U_ = [row[:] for row in U]
        V_ = [row[:] for row in V]
        W_ = [row[:] for row in W]
        
        cur_smooth = get_smooth_error(rank, U_, V_, W_)
        cur_err = get_error_count(rank, U_, V_, W_)
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        
        best_smooth = cur_smooth
        best_err = cur_err
        no_improve = 0
        
        # Candidate values for each entry - include half-integers
        candidate_vals = [0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25, 1.5, -1.5]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if cur_err == 0:
                break
            
            # Choose a random factor and entry to modify
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_[idx][c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    U_[idx][c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    U_[idx][c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        U_[idx][c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_[idx][c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    V_[idx][c] = val
                    new_smooth = get_smooth_error(rank, V_, U_, W_)  # wrong order, fix below
                    # Actually need to pass in correct order
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    V_[idx][c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        V_[idx][c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_[idx][c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    W_[idx][c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    W_[idx][c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        W_[idx][c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 60:
                break
        
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        return None
    
    def structured_init(rank, rng):
        """Initialize with structure based on matrix multiplication patterns"""
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.5:
                        U[i*m+j][c] = rng.choice([1, -1, 0.5, -0.5])
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.5:
                        V[j*p+k][c] = rng.choice([1, -1, 0.5, -0.5])
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.5:
                        W[k*n+i][c] = rng.choice([1, -1, 0.5, -0.5])
        
        return U, V, W
    
    def strassen_based_init(rank, rng):
        """
        Use Strassen's algorithm structure for the 4x4x4 part (rank 7 per block of 2x2x2),
        and schoolbook for the remaining column.
        
        For 4x4 * 4x5:
        - Split the 5 columns into 4 + 1
        - For the 4x4x4 part, use Strassen-like decomposition
        - For the 4x4x1 part, use schoolbook (16 terms)
        
        Total: 7 (Strassen for 2x2x2 blocks, but 4x4 = 4 blocks of 2x2, so 4*7=28? No...)
        
        Actually, Strassen for 2x2x2 has rank 7. For 4x4x4, we can tile:
        4x4x4 = 8 blocks of 2x2x2, each rank 7, total 56. Plus 16 for the last column = 72.
        That's not better than 80.
        
        Better: Use a known 4x4x4 decomposition. The minimal rank for 4x4x4 is known to be <= 49
        (from Strassen's 7 for 2x2x2, tiling gives 7*8=56, but there are better).
        
        Let me use a different approach: start with Strassen for 2x2x2 blocks.
        """
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # For 4x4x4 part (columns 0-3), use Strassen tiling
        # 4x4 matrix = 4 blocks of 2x2
        # Strassen for 2x2: M1=A11*(B12-B22), M2=(A11+A12)*B22, M3=(A21+A22)*B11,
        # M4=A22*(B21-B11), M5=(A11+A22)*(B11+B22), M6=(A11-A21)*(B11+B21), M7=(A12-A22)*(B12+B22)
        # C11=M5+M4-M2-M7, C12=M1+M2, C21=M3+M4, C22=M1-M3+M5+M7
        
        # For 4x4x4, tile into 2x2 blocks of 2x2 matrices
        # There are 4 row-blocks (2x2) and 4 col-blocks (2x2), so 16 sub-products
        # Each sub-product uses 7 Strassen terms
        # Total: 16*7 = 112? No, that's worse.
        
        # Actually, for 4x4x4, the known best is around 49-52.
        # Let me just use a random initialization and rely on local search.
        
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.4:
                        U[i*m+j][c] = rng.choice([1, -1, 0.5, -0.5, 0])
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.4:
                        V[j*p+k][c] = rng.choice([1, -1, 0.5, -0.5, 0])
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.4:
                        W[k*n+i][c] = rng.choice([1, -1, 0.5, -0.5, 0])
        
        return U, V, W
    
    def numpy_smooth_error(rank, U, V, W):
        """Faster smooth error using numpy"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k]) * W[k*n+i]  # wrong, need element-wise
                    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    total += (s - 1.0) ** 2
        return total
    
    # Try ranks from 49 upward (known lower bound for 4x4x4 is 49, plus some for the extra column)
    # For 4x4x5, the rank is at least max(4*4, 4*5, 4*5) = 20, but practically much higher.
    # Known: rank(4,4,4) <= 49, so rank(4,4,5) <= 49 + 16 = 65 (schoolbook for last col)
    # Or better: use a joint decomposition.
    
    for rank in range(49, 66):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        num_restarts = 30 if rank <= 55 else 20
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)
            
            # Initialize with structured random values
            U, V, W = structured_init(rank, rng)
            
            result = local_search_smooth(rank, U, V, W, seed + 7777, max_steps=150)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break
    
    return best
