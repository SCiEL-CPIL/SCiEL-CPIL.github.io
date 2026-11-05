import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _construct_rank13_block():
    """
    Find a rank-13 decomposition for 2x4x2 matrix multiplication.
    Returns U, V, W arrays of shapes (8, 13), (8, 13), (4, 13) or None.
    """
    n, m, p = 2, 4, 2
    rank = 13
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # ALS iteration
    for it in range(1000):
        # Solve for U
        # U[i, j, r] * sum_k V[j, k, r] W[k, i, r] = 1
        # For each r, we have a linear system? No, it's coupled.
        # Actually, for fixed V, W, the equation is linear in U.
        # sum_r U[i, j, r] * (V[j, :, r] dot W[:, i, r]) = 1
        # Let M_r[i, j] = V[j, :, r] dot W[:, i, r]
        # Then sum_r U[:, :, r] * M_r = 1
        # This is a linear system for U.
        
        # We can solve for each r? No, U is a matrix of size (n*m, rank).
        # We can reshape U to (n, m, rank).
        # The equation is:
        # einsum('ijr,jkr,kin->ijk', U, V, W) = T
        
        # For fixed V, W, we can write:
        # U[i, j, r] = sum_k T[i, j, k] * W[k, i, r] * V[j, k, r] / (sum_k V[j, k, r] * W[k, i, r]) ?
        # No, this is not correct because the sum is over r.
        
        # Correct ALS step:
        # We want to minimize || T - sum_r u_r v_r^T w_r ||^2
        # This is a standard CP decomposition problem.
        # We can use the update rule:
        # U[:, :, r] = T * (V[:, :, r] W[:, :, r]) / (V[:, :, r] W[:, :, r])
        # where * denotes element-wise multiplication and / is element-wise division.
        # But T is 3D, V and W are 3D.
        
        # Let's use the standard ALS update for CP:
        # U[i, j, r] = sum_k T[i, j, k] V[j, k, r] W[k, i, r] / sum_k V[j, k, r] W[k, i, r]
        
        # Compute denominator
        # Denom[i, j, r] = sum_k V[j, k, r] W[k, i, r]
        # V is (m, p, rank) -> (4, 2, 13)
        # W is (p, n, rank) -> (2, 2, 13)
        # We need to align indices.
        
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # We want Denom[i, j, r] = sum_k V3[j, k, r] * W3[k, i, r]
        # This is a tensor contraction.
        # We can do this for each r.
        Denom = np.zeros((n, m, rank))
        Num = np.zeros((n, m, rank))
        for r in range(rank):
            # V3[:, :, r] is (m, p)
            # W3[:, :, r] is (p, n)
            # We want sum_k V3[j, k, r] W3[k, i, r]
            # This is (V3[:, :, r].T @ W3[:, :, r])?
            # V3[j, k] * W3[k, i] -> sum_k
            # Result[i, j]
            M_r = V3[:, :, r].T @ W3[:, :, r] # (m, p) @ (p, n) -> (m, n)
            # M_r[j, i] = sum_k V3[j, k] W3[k, i]
            # We need Denom[i, j, r] = M_r[j, i]
            Denom[:, :, r] = M_r.T
            # Num[i, j, r] = sum_k T[i, j, k] V3[j, k, r] W3[k, i, r]
            # T is (n, m, p)
            # T[i, j, k] * V3[j, k, r] * W3[k, i, r]
            # Sum over k
            # We can compute this as:
            # Temp[i, j] = sum_k T[i, j, k] V3[j, k, r] W3[k, i, r]
            # This is not a simple matrix product.
            # Let's loop over k? p=2, so it's small.
            Num_r = np.zeros((n, m))
            for k in range(p):
                Num_r += T[:, :, k] * V3[:, k, r][np.newaxis, :] * W3[k, :, r][:, np.newaxis]
            Num[:, :, r] = Num_r
            
        # Update U
        # U[i, j, r] = Num[i, j, r] / Denom[i, j, r]
        # Handle division by zero
        U3 = Num / np.where(Denom == 0, 1e-12, Denom)
        U = U3.reshape(n * m, rank)
        
        # Similarly for V and W
        # V[j, k, r] = sum_i T[i, j, k] U[i, j, r] W[k, i, r] / sum_i U[i, j, r] W[k, i, r]
        U3 = U.reshape(n, m, rank)
        # Denom_V[j, k, r] = sum_i U3[i, j, r] W3[k, i, r]
        Denom_V = np.zeros((m, p, rank))
        Num_V = np.zeros((m, p, rank))
        for r in range(rank):
            # U3[:, :, r] is (n, m)
            # W3[:, :, r] is (p, n)
            # We want sum_i U3[i, j, r] W3[k, i, r]
            # This is (U3[:, :, r] @ W3[:, :, r].T)?
            # U3[i, j] * W3[k, i] -> sum_i
            # Result[j, k]
            M_r = U3[:, :, r] @ W3[:, :, r].T # (n, m) @ (n, p) -> (m, p)
            # M_r[j, k] = sum_i U3[i, j] W3[k, i]
            Denom_V[:, :, r] = M_r
            # Num_V[j, k, r] = sum_i T[i, j, k] U3[i, j, r] W3[k, i, r]
            Num_V_r = np.zeros((m, p))
            for i in range(n):
                Num_V_r += T[i, :, :] * U3[i, :, r][:, np.newaxis, :] * W3[:, i, r][np.newaxis, :]
            Num_V[:, :, r] = Num_V_r
            
        V3_new = Num_V / np.where(Denom_V == 0, 1e-12, Denom_V)
        V = V3_new.reshape(m * p, rank)
        
        # W[k, i, r] = sum_j T[i, j, k] U3[i, j, r] V3_new[j, k, r] / sum_j U3[i, j, r] V3_new[j, k, r]
        V3 = V.reshape(m, p, rank)
        Denom_W = np.zeros((p, n, rank))
        Num_W = np.zeros((p, n, rank))
        for r in range(rank):
            # U3[:, :, r] is (n, m)
            # V3[:, :, r] is (m, p)
            # We want sum_j U3[i, j, r] V3[j, k, r]
            # This is (U3[:, :, r] @ V3[:, :, r])?
            # U3[i, j] * V3[j, k] -> sum_j
            # Result[i, k]
            M_r = U3[:, :, r] @ V3[:, :, r] # (n, m) @ (m, p) -> (n, p)
            # M_r[i, k] = sum_j U3[i, j] V3[j, k]
            # We need Denom_W[k, i, r] = M_r[i, k]
            Denom_W[:, :, r] = M_r.T
            # Num_W[k, i, r] = sum_j T[i, j, k] U3[i, j, r] V3[j, k, r]
            Num_W_r = np.zeros((p, n))
            for j in range(m):
                Num_W_r += T[:, j, :] * U3[:, j, r][:, np.newaxis] * V3[j, :, r][np.newaxis, :]
            # Num_W_r[k, i] = sum_j T[i, j, k] ...
            # Wait, T is (n, m, p). T[i, j, k].
            # Num_W_r[k, i] = sum_j T[i, j, k] U3[i, j] V3[j, k]
            # My loop:
            # Num_W_r += T[:, j, :] * ...
            # T[:, j, :] is (n, p).
            # U3[:, j, r] is (n,).
            # V3[j, :, r] is (p,).
            # Product is (n, p).
            # Sum over j gives (n, p).
            # So Num_W_r is (n, p).
            # We need Denom_W[k, i, r] which is (p, n).
            # So Num_W[:, :, r] should be Num_W_r.T
            Num_W[:, :, r] = Num_W_r.T
            
        W3_new = Num_W / np.where(Denom_W == 0, 1e-12, Denom_W)
        W = W3_new.reshape(p * n, rank)
        
        # Check error
        T_est = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3_new)
        err = np.max(np.abs(T_est - T))
        if err < 1e-9:
            return U, V, W

    return None

def _construct_rank52():
    """
    Construct a rank-52 decomposition using 4 rank-13 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 52
    
    # Get the rank-13 block
    U_block, V_block, W_block = _construct_rank13_block()
    if U_block is None:
        return None
        
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    block_rank = 13
    for l in range(4):
        # l corresponds to k in [2l, 2l+1]
        # U is same for all blocks
        U[:, l*block_rank : (l+1)*block_rank] = U_block
        
        # V depends on l
        # V_block is for 4x2 multiplication.
        # For block l, the k indices are 2l, 2l+1.
        # V_block[j, k, r] corresponds to j in 0..3, k in 0..1.
        # In the full V, k is 0..7.
        # So V[j, 2l+k, r] = V_block[j, k, r]
        V_block_3d = V_block.reshape(m, 2, block_rank)
        for j in range(m):
            for k in range(2):
                V[j * p + (2*l + k), l*block_rank : (l+1)*block_rank] = V_block_3d[j, k, :]
                
        # W depends on l
        # W_block is for 2x2 output.
        # W_block[k, i, r] corresponds to k in 0..1, i in 0..1.
        # In the full W, k is 0..7.
        # W[2l+k, i, r] = W_block[k, i, r]
        W_block_3d = W_block.reshape(2, n, block_rank)
        for k in range(2):
            for i in range(n):
                W[(2*l + k) * n + i, l*block_rank : (l+1)*block_rank] = W_block_3d[k, i, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _matmul_check(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)

def _construct_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    # Try rank-52
    try:
        if time.time() < deadline - 10:
            cand = _construct_rank52()
            if cand is not None:
                ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
                if ok:
                    return cand
    except Exception:
        pass

    # Fallback to rank-56
    cand = _construct_rank56()
    return cand
