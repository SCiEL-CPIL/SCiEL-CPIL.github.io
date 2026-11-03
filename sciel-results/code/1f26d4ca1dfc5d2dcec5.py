import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def _validate_numpy(U, V, W, n, m, p):
    """
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    
    rank = U.shape[1]
    
    # Reshape to 3D tensors
    # U[i, j, r]
    U_3d = U.reshape(n, m, rank)
    # V[j, k, r]
    V_3d = V.reshape(m, p, rank)
    # W[k, i, r]
    W_3d = W.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # Let's compute the tensor product contraction
    
    # Create an array T of shape (n, m, p) initialized to 0
    # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # This is a bit tricky to vectorize simply without loops over i,j,k or r.
    # But n,m,p are small.
    # We can loop over r and add outer products?
    # Or loop over i,j,k and sum over r.
    
    # Since rank might be up to 72, and n*m*p = 72, 
    # looping over 72*72 = 5184 operations is fast.
    
    # Let's use einsum or explicit loops if numpy is too complex for the specific index mapping.
    # U_3d[i,j,r], V_3d[j,k,r], W_3d[k,i,r]
    
    # For each r, we have a rank-1 contribution to the tensor.
    # Contribution to T[i,j,k] is U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's just loop over i, j, k and check.
    # To speed up, we can pre-compute for each r the values.
    
    # Actually, let's just use the python loop but with numpy arrays for the inner sum.
    # Or better, construct the full tensor T using broadcasting.
    
    # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # We can reshape W to (n, p, rank) ? No, W is (p, n, rank).
    # We need W[k,i,r]. So W_3d is (p, n, rank).
    # We want to align indices: i, j, k.
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r
    
    # Let's permute W to (i, k, r)? No, W[k,i,r]. 
    # We want T[i,j,k].
    # Sum over r.
    
    # T = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d.transpose(1,0,2))
    # Wait, W_3d is (p, n, rank) i.e. (k, i, r).
    # We need the term W[k,i,r].
    # In einsum, we specify indices.
    # U_3d: i j r
    # V_3d: j k r
    # W_3d: k i r
    # Result: i j k
    
    # Let's try:
    try:
        T = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d) # Incorrect, W is k,i,r not k,r
        # W_3d shape is (p, n, rank) -> (k, i, r)
        # We need to access W[k,i,r].
        # einsum string for W: 'kir'
        # So: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
        
    # Check if T is all ones
    if not np.all(T == 1):
        return False
    return True

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    return _validate_numpy(U, V, W, n, m, p)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _search_332(rank_limit, time_limit):
    """Search for a rank <= rank_limit decomposition for 3x3x2."""
    n, m, p = 3, 3, 2
    start_time = time.time()
    values = [-3, -2, -1, 0, 1, 2, 3]
    
    # Random search
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        U = [[random.choice(values) for _ in range(rank_limit)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank_limit)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank_limit)] for _ in range(p * n)]
        
        if _validate(U, V, W, n, m, p):
            return U, V, W
            
    return None

def _construct_from_332(U_332, V_332, W_332):
    """
    Construct 3x3x8 decomposition from 3x3x2 decomposition.
    n=3, m=3, p=8.
    Split p into 4 blocks of size 2.
    """
    n, m, p = 3, 3, 8
    rank_332 = len(U_332[0])
    total_rank = 4 * rank_332
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # U is shared across all blocks?
    # U[i*m+j, r] is the same for all k blocks.
    # So we just repeat the columns of U_332 4 times.
    for i in range(n * m):
        for r in range(rank_332):
            for block in range(4):
                U[i][block * rank_332 + r] = U_332[i][r]
                
    # V depends on j and k.
    # For block b (k in [2b, 2b+1]), V[j*p+k, r] corresponds to V_332[j*2 + (k%2), r]
    for j in range(m):
        for k in range(p):
            block = k // 2
            k_local = k % 2
            for r in range(rank_332):
                V[j * p + k][block * rank_332 + r] = V_332[j * 2 + k_local][r]
                
    # W depends on k and i.
    # For block b, W[k*n+i, r] corresponds to W_332[(k%2)*3 + i, r]
    for k in range(p):
        block = k // 2
        k_local = k % 2
        for i in range(n):
            for r in range(rank_332):
                W[k * n + i][block * rank_332 + r] = W_332[k_local * 3 + i][r]
                
    return {"U": U, "V": V, "W": W}

def _search_338_direct(rank_limit, time_limit):
    """Direct search for 3x3x8 with rank <= rank_limit."""
    n, m, p = 3, 3, 8
    start_time = time.time()
    values = [-2, -1, 0, 1, 2]
    
    for _ in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        U = [[random.choice(values) for _ in range(rank_limit)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank_limit)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank_limit)] for _ in range(p * n)]
        
        if _validate(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115 # Safety margin
    
    # Fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Strategy 1: Find a better 3x3x2 decomposition and tile it.
    # Schoolbook 3x3x2 rank is 18. Tiled rank 72.
    # Try to find rank 17, 16, 15...
    for r_332 in range(17, 13, -1):
        if time.time() > deadline - 20:
            break
        result_332 = _search_332(r_332, time_limit=10)
        if result_332:
            U, V, W = result_332
            decomp = _construct_from_332(U, V, W)
            if _validate(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                return decomp
                
    # Strategy 2: Direct search for 3x3x8 with rank < 72
    # Try ranks 71, 70, 69...
    for r_338 in range(71, 60, -1):
        if time.time() > deadline - 10:
            break
        result_338 = _search_338_direct(r_338, time_limit=5)
        if result_338:
            U, V, W = result_338
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    return schoolbook
