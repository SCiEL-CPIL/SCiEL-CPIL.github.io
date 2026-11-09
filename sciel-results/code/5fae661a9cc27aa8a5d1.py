import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Target tensor T[u, v, w] = 1 if (u, v, w) corresponds to a term in matrix mult
    # u = i*m + j, v = j*p + k, w = k*n + i
    # Note: The tensor is sparse. Only specific (u,v,w) triples are non-zero.
    target_support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_support.add((u, v, w))
    
    # Precompute indices for faster access
    target_u, target_v, target_w = [], [], []
    for u, v, w in target_support:
        target_u.append(u)
        target_v.append(v)
        target_w.append(w)
    target_u = np.array(target_u)
    target_v = np.array(target_v)
    target_w = np.array(target_w)
    
    # All possible indices
    all_u = np.arange(nm)
    all_v = np.arange(mp)
    all_w = np.arange(np_)
    
    def error(U, V, W):
        # U: (nm, r), V: (mp, r), W: (np, r)
        r = U.shape[1]
        # Compute sum over c of U[u,c] V[v,c] W[w,c] for all u,v,w
        # This is a tensor contraction.
        # We can do this by iterating over c and accumulating into a 3D array, 
        # but 3D array is nm*mp*np = 1764 floats. That's small.
        T = np.zeros((nm, mp, np_))
        for c in range(r):
            # Outer product of U[:,c], V[:,c], W[:,c]
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Calculate squared error
        # Target is 1 for (u,v,w) in support, 0 otherwise
        err = 0.0
        # Iterate over all u, v, w
        # T is (nm, mp, np_)
        # Target is 1 at target_u, target_v, target_w
        # err = sum(T^2) - 2 * sum(T[target]) + num_targets
        # sum(T^2)
        sum_t_sq = np.sum(T * T)
        # sum(T[target])
        sum_t_target = 0.0
        for idx in range(len(target_u)):
            sum_t_target += T[target_u[idx], target_v[idx], target_w[idx]]
        
        num_targets = len(target_u)
        err = sum_t_sq - 2 * sum_t_target + num_targets
        return err

    def is_valid(U, V, W, rank):
        # Exact check with tolerance
        # For speed, we can use the error function and check if it's close to 0
        # But floating point errors might accumulate.
        # Let's do a direct check on the support and non-support.
        # For u,v,w in support: sum should be 1
        # For u,v,w not in support: sum should be 0
        
        # To be fast, we compute T as above.
        r = U.shape[1]
        T = np.zeros((nm, mp, np_))
        for c in range(r):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Check support
        for idx in range(len(target_u)):
            val = T[target_u[idx], target_v[idx], target_w[idx]]
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check non-support (sample or full check?)
        # Full check is 1764 iterations, which is fast.
        for u in range(nm):
            for v in range(mp):
                for w in range(np_):
                    if (u, v, w) in target_support:
                        continue
                    val = T[u, v, w]
                    if abs(val) > 1e-9:
                        return False
        return True

    def solve_rank(r, values, max_attempts=5000):
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        attempts = 0
        while time.time() - START_TIME < TIME_LIMIT:
            attempts += 1
            if attempts > max_attempts:
                break
            
            # Random initialization
            U = np.random.choice(values, size=(nm, r))
            V = np.random.choice(values, size=(mp, r))
            W = np.random.choice(values, size=(np_, r))
            
            current_err = error(U, V, W)
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if current_err < 1e-9:
                # Verify exactness
                if is_valid(U, V, W, r):
                    return U, V, W
            
            # Local search
            stuck_counter = 0
            while time.time() - START_TIME < TIME_LIMIT:
                if stuck_counter > 100:
                    break
                
                # Pick random matrix, row, col
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    dims = nm
                elif mat_idx == 1:
                    mat = V
                    dims = mp
                else:
                    mat = W
                    dims = np_
                
                idx = np.random.choice(dims)
                col = np.random.choice(r)
                
                old_val = mat[idx, col]
                # Try a new value from the set
                new_val = np.random.choice(values)
                
                if new_val == old_val:
                    continue
                    
                mat[idx, col] = new_val
                new_err = error(U, V, W)
                
                if new_err < current_err:
                    current_err = new_err
                    stuck_counter = 0
                    if current_err < 1e-9:
                        if is_valid(U, V, W, r):
                            return U, V, W
                else:
                    mat[idx, col] = old_val
                    stuck_counter += 1
                    
        return best_U, best_V, best_W

    # Fallback: Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find a better rank
    # Start from lower bound. For 2x3x7, rank is at least 14.
    # Let's try from 14 to 41.
    for r in range(14, rank_sb):
        if time.time() - START_TIME > TIME_LIMIT * 0.8:
            break
            
        # Try integer values first
        res = solve_rank(r, [0, 1, -1], max_attempts=2000)
        if res and res[0] is not None:
            U, V, W = res
            # Convert to list of lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Try half-integers
        res = solve_rank(r, [0, 0.5, 1, -0.5, -1], max_attempts=2000)
        if res and res[0] is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no better rank found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
