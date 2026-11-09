import numpy as np
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    rank = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _split_p_search(n, m, p, time_limit=20.0):
    """
    Try to find a decomposition by splitting the p dimension into two parts.
    For each split p1 + p2 = p, we solve two subproblems independently and combine.
    The rank is rank1 + rank2.
    """
    start = time.time()
    best_rank = n * m * p
    best_decomp = None
    
    # Try all splits p1 from 1 to p-1
    for p1 in range(1, p):
        p2 = p - p1
        
        # Subproblem 1: n x m times m x p1
        # Subproblem 2: n x m times m x p2
        
        # We can use random search for each subproblem
        # But for small sizes, we can just use schoolbook if random fails
        # Or try random search with a time limit
        
        time_per_sub = (time_limit - (time.time() - start)) / 2.0
        
        # Try to find rank for subproblem 1
        rank1 = n * m * p1  # schoolbook
        decomp1 = None
        
        # Try random search for subproblem 1
        rng = np.random.RandomState(123 + p1)
        sub_start = time.time()
        for r1 in range(n*m*p1, max(n*m, m*p1, n*p1) - 1, -1):
            if time.time() - sub_start > time_per_sub / 2.0:
                break
            
            found = False
            for attempt in range(20):
                U1 = rng.randn(n*m, r1)
                V1 = rng.randn(m*p1, r1)
                
                W1 = np.zeros((p1*n, r1))
                valid = True
                
                for ki in range(p1*n):
                    k = ki // n
                    i = ki % n
                    A = np.zeros((m, r1))
                    for j in range(m):
                        A[j, :] = U1[i*m + j, :] * V1[j*p1 + k, :]
                    
                    if r1 >= m:
                        w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-6:
                            valid = False
                            break
                        W1[ki, :] = w
                    else:
                        valid = False
                        break
                
                if valid and _verify(W1.tolist(), V1.tolist(), U1.tolist(), n, m, p1):
                    decomp1 = (U1.tolist(), V1.tolist(), W1.tolist())
                    rank1 = r1
                    found = True
                    break
            
            if found:
                break
        
        # Try to find rank for subproblem 2
        rank2 = n * m * p2  # schoolbook
        decomp2 = None
        
        rng = np.random.RandomState(456 + p2)
        sub_start = time.time()
        for r2 in range(n*m*p2, max(n*m, m*p2, n*p2) - 1, -1):
            if time.time() - sub_start > time_per_sub / 2.0:
                break
            
            found = False
            for attempt in range(20):
                U2 = rng.randn(n*m, r2)
                V2 = rng.randn(m*p2, r2)
                
                W2 = np.zeros((p2*n, r2))
                valid = True
                
                for ki in range(p2*n):
                    k = ki // n
                    i = ki % n
                    A = np.zeros((m, r2))
                    for j in range(m):
                        A[j, :] = U2[i*m + j, :] * V2[j*p2 + k, :]
                    
                    if r2 >= m:
                        w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-6:
                            valid = False
                            break
                        W2[ki, :] = w
                    else:
                        valid = False
                        break
                
                if valid and _verify(W2.tolist(), V2.tolist(), U2.tolist(), n, m, p2):
                    decomp2 = (U2.tolist(), V2.tolist(), W2.tolist())
                    rank2 = r2
                    found = True
                    break
            
            if found:
                break
        
        # Combine decompositions
        total_rank = rank1 + rank2
        if total_rank < best_rank:
            # Build combined U, V, W
            U = [[0] * total_rank for _ in range(n*m)]
            V = [[0] * total_rank for _ in range(m*p)]
            W = [[0] * total_rank for _ in range(p*n)]
            
            # Fill in subproblem 1
            for row in range(n*m):
                for c in range(rank1):
                    U[row][c] = decomp1[0][row][c]
            for row in range(m*p1):
                for c in range(rank1):
                    V[row][c] = decomp1[1][row][c]
            for row in range(p1*n):
                for c in range(rank1):
                    W[row][c] = decomp1[2][row][c]
            
            # Fill in subproblem 2 (offset by rank1)
            for row in range(n*m):
                for c in range(rank2):
                    U[row][rank1 + c] = decomp2[0][row][c]
            for row in range(m*p2):
                for c in range(rank2):
                    V[row + m*p1][rank1 + c] = decomp2[1][row][c]
            for row in range(p2*n):
                for c in range(rank2):
                    W[row + p1*n][rank1 + c] = decomp2[2][row][c]
            
            best_rank = total_rank
            best_decomp = {"U": U, "V": V, "W": W}
        
        if time.time() - start > time_limit:
            break
    
    return best_decomp, best_rank

def _try_rank(r, n, m, p, time_limit=30.0, seed=0):
    """Try to find a decomposition of rank r. Returns (U,V,W) or None."""
    rng = np.random.RandomState(seed)
    start = time.time()
    
    num_ki = p * n
    num_ij = n * m
    num_jk = m * p
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        U = rng.randn(num_ij, r)
        V = rng.randn(num_jk, r)
        
        W = np.zeros((num_ki, r))
        valid = True
        
        for ki in range(num_ki):
            k = ki // n
            i = ki % n
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            
            if r >= m:
                w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                if len(residuals) > 0:
                    if residuals[0] > 1e-6:
                        valid = False
                        break
                W[ki, :] = w
            else:
                valid = False
                break
        
        if not valid:
            continue
        
        if _verify(W.tolist(), V.tolist(), U.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try split-p approach first (more structured, likely to find better rank)
    decomp, rank = _split_p_search(n, m, p, time_limit=40.0)
    if decomp is not None and rank < best_rank:
        best = decomp
        best_rank = rank
    
    # Try direct random search for even lower ranks
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 20.0:
        # Try ranks from best_rank - 1 down to 20
        for r in range(best_rank - 1, 19, -1):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            time_per_rank = min(10.0, (DEADLINE - 10 - (time.time() - start_time)) / 5.0)
            
            result = _try_rank(r, n, m, p, time_limit=time_per_rank, seed=r)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 5:
        return _schoolbook()
    
    return best
