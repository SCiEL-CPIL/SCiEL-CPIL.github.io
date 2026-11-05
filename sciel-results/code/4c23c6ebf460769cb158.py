"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

def _make_rank55():
    """Return the known rank-55 decomposition using Strassen on 5 column pairs."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0, idx] += 1; U[4, idx] += 1
        V[k0, idx] += 1; V[10 + k1, idx] += 1
        W[2*k0, idx] += 1; W[2*k1+1, idx] += 1
        idx += 1
        
        # M2: (A10 + A11) * B0,k0
        U[3, idx] += 1; U[4, idx] += 1
        V[k0, idx] += 1
        W[2*k0+1, idx] += 1; W[2*k1+1, idx] -= 1
        idx += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0, idx] += 1
        V[k1, idx] += 1; V[10+k1, idx] -= 1
        W[2*k1, idx] += 1; W[2*k1+1, idx] += 1
        idx += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[4, idx] += 1
        V[10+k0, idx] += 1; V[k0, idx] -= 1
        W[2*k0, idx] += 1; W[2*k0+1, idx] += 1
        idx += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0, idx] += 1; U[1, idx] += 1
        V[10+k1, idx] += 1
        W[2*k0, idx] -= 1; W[2*k1, idx] += 1
        idx += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[3, idx] += 1; U[0, idx] -= 1
        V[k0, idx] += 1; V[k1, idx] += 1
        W[2*k1+1, idx] += 1
        idx += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[1, idx] += 1; U[4, idx] -= 1
        V[10+k0, idx] += 1; V[10+k1, idx] += 1
        W[2*k0, idx] += 1
        idx += 1
        
        # Schoolbook for 3rd row of A: A02*B2,k and A12*B2,k
        for k in [k0, k1]:
            U[2, idx] += 1
            V[20+k, idx] += 1
            W[2*k, idx] += 1
            idx += 1
            U[5, idx] += 1
            V[20+k, idx] += 1
            W[2*k+1, idx] += 1
            idx += 1
    
    return U, V, W

def _verify_decomp(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """Verify that sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1 for all valid i,j,k."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = np.dot(U[u_row, :] * V[v_row, :], W[w_row, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_lower_rank(target_rank, max_iter=200, deadline=None):
    """
    Try to find a rank-<target_rank> decomposition by starting from rank-55
    and attempting to eliminate one column via perturbation.
    """
    n, m, p = 2, 3, 10
    U55, V55, W55 = _make_rank55()
    
    # Start with first target_rank columns
    U = U55[:, :target_rank].copy()
    V = V55[:, :target_rank].copy()
    W = W55[:, :target_rank].copy()
    
    # If already valid, return
    if _verify_decomp(U, V, W):
        return U, V, W
    
    # Try to fix by adjusting the last few columns
    # Use a simple iterative approach: for each column that's "off", try to compensate
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute error tensor
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    val = np.dot(U[u_row, :] * V[v_row, :], W[w_row, :])
                    errors.append((i, j, k, val - 1.0))
        
        err = max(abs(e[3]) for e in errors)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            return best_U, best_V, best_W
        
        # Perturbation: add small random adjustments to random entries
        for _ in range(3):
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = np.random.randint(0, n*m)
                col = np.random.randint(0, target_rank)
                U[row, col] += np.random.uniform(-0.1, 0.1)
            elif mat == 'V':
                row = np.random.randint(0, m*p)
                col = np.random.randint(0, target_rank)
                V[row, col] += np.random.uniform(-0.1, 0.1)
            else:
                row = np.random.randint(0, p*n)
                col = np.random.randint(0, target_rank)
                W[row, col] += np.random.uniform(-0.1, 0.1)
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x10.
    Start with rank-55, then search for lower rank with time budget.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, get the known rank-55 solution
    U55, V55, W55 = _make_rank55()
    
    # Verify the rank-55 solution
    if not _verify_decomp(U55, V55, W55):
        raise RuntimeError("Rank-55 solution failed verification!")
    
    best_result = (U55, V55, W55)
    best_rank = 55
    
    # Try to find a lower rank
    for target_rank in range(54, 45, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_lower_rank(target_rank, max_iter=50, deadline=deadline - 15)
        if result is not None:
            U, V, W = result
            if _verify_decomp(U, V, W):
                best_result = (U, V, W)
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return {'U': best_result[0].tolist(), 'V': best_result[1].tolist(), 'W': best_result[2].tolist()}
