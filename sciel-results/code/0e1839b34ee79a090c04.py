import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    # Check if entries are integers or half-integers
    # and if the decomposition is exact
    n, m, p = DIMENSIONS
    # Flatten U, V, W to check types
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2)):
            return False
    
    # Reconstruct tensor
    # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r]
    # We only care about the non-zero parts of the true tensor
    # True tensor T_true[a, b, c] is 1 if a=(i,j), b=(j,k), c=(i,k) with i,i',j,j',k,k' matching appropriately.
    # Specifically, T_true[(i,j), (j,k), (i,k)] = 1.
    # All other entries are 0.
    
    # Construct T_true
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Construct T_approx
    # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check exactness
    if not np.allclose(T_true, T_approx):
        return False
        
    return True

def _als_decompose(target_rank, time_limit):
    n, m, p = DIMENSIONS
    dims = [n*m, m*p, p*n]
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(dims[0], target_rank)
        V = np.random.randn(dims[1], target_rank)
        W = np.random.randn(dims[2], target_rank)
        
        # Normalize to help convergence
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        for it in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            # U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / (sum_c W[c,r]^2)
            # Actually, standard ALS:
            # U = (T x2 V x3 W) / (V^T V * W^T W)
            
            # Compute T x2 V
            # (T x2 V)[a, c, r] = sum_b T[a,b,c] V[b,r]
            TV = np.einsum('abc,br->acr', T_true, V)
            # Compute (T x2 V) x3 W
            # (TV x3 W)[a, r, s] = sum_c TV[a,c,r] W[c,s]
            TVW = np.einsum('acr,cs->ars', TV, W)
            
            # Denominator for U: sum_c W[c,r]^2
            # This is diagonal in r? No, for each r, sum over c of W[c,r]^2.
            # But in CP decomposition, the update is:
            # U[:, r] = (sum_{b,c} T[a,b,c] V[b,r] W[c,r]) / (sum_c W[c,r]^2 * sum_b V[b,r]^2)?
            # Wait, the standard ALS update for U is:
            # U[:, r] = (T x2 V[:, r] x3 W[:, r]) / (||V[:, r]||^2 ||W[:, r]||^2)
            
            # Let's do it per column r to be safe and clear
            for r in range(target_rank):
                # Numerator: sum_{b,c} T[a,b,c] V[b,r] W[c,r]
                # This is (T x2 V[:, r]) x3 W[:, r]
                # T x2 V[:, r] is a matrix of size (n*m, p*n)
                TV_r = np.einsum('abc,b->ac', T_true, V[:, r])
                Num = np.einsum('ac,c->a', TV_r, W[:, r])
                
                # Denominator: ||V[:, r]||^2 * ||W[:, r]||^2
                Den = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
                
                if Den > 1e-10:
                    U[:, r] = Num / Den
                else:
                    U[:, r] = 0.0
            
            # Similarly for V and W
            for r in range(target_rank):
                TW_r = np.einsum('abc,c->ab', T_true, W[:, r])
                Num = np.einsum('ab,b->a', TW_r, U[:, r])
                Den = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
                if Den > 1e-10:
                    V[:, r] = Num / Den
                else:
                    V[:, r] = 0.0
                    
            for r in range(target_rank):
                TU_r = np.einsum('abc,a->bc', T_true, U[:, r])
                Num = np.einsum('bc,b->c', TU_r, V[:, r])
                Den = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
                if Den > 1e-10:
                    W[:, r] = Num / Den
                else:
                    W[:, r] = 0.0
            
            # Check convergence / exactness
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_true, T_approx):
                # Snap to integers/half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                if _verify(U_snapped, V_snapped, W_snapped):
                    return {"U": U_snapped, "V": V_snapped, "W": W_snapped}

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a lower rank decomposition
    # Schoolbook is 54.
    # Try ranks 43, 44, 45, 46
    for rank in [43, 44, 45, 46]:
        if time.time() > deadline - 10:
            break
        res = _als_decompose(rank, 20)
        if res:
            return res
            
    # Fallback to schoolbook
    return _schoolbook()
