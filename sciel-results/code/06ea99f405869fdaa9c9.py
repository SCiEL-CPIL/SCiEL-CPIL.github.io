"""Matrix multiplication 3x3 x 3x5: find rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 140  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def _build_tensor():
    """Build the 3x3 x 3x5 matrix multiplication tensor T[a,b,c].
    a = i*m+j (A index), b = j'*p+k (B index), c = k'*n+i' (C index)
    T[a,b,c] = 1 if i=i', j=j', k=k', else 0.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _schoolbook():
    """Return the schoolbook decomposition (rank 45)."""
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
    return U, V, W

def _verify(U, V, W, T, tol=1e-8):
    """Verify U⊗V⊗W sums to T."""
    T_recon = np.zeros_like(T)
    for c in range(U.shape[1]):
        T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T_recon - T)) < tol

def _try_greedy_peel(T, max_rank, rng, time_limit):
    """Greedily peel rank-1 components from T.
    At each step, find the best rank-1 u⊗v⊗w to subtract.
    """
    T_work = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    start_time = time.time()
    
    for step in range(max_rank):
        if time.time() - start_time > time_limit:
            return None
        
        best_score = np.inf
        best_u = None
        best_v = None
        best_w = None
        
        n_trials = 200
        for trial in range(n_trials):
            if time.time() - start_time > time_limit:
                return None
            
            # Random initialization with structure
            u = rng.standard_normal(T.shape[0])
            v = rng.standard_normal(T.shape[1])
            
            # Project to get w
            w = np.einsum('abc,a,b->c', T_work, u, v)
            nu2 = np.dot(u, u)
            nv2 = np.dot(v, v)
            if nu2 < 1e-12 or nv2 < 1e-12:
                continue
            w = w / (nu2 * nv2)
            
            # Score: residual norm after subtracting u⊗v⊗w
            inner = np.einsum('abc,a,b,c->', T_work, u, v, w)
            norm_T2 = np.dot(T_work.ravel(), T_work.ravel())
            norm_r1_2 = nu2 * nv2 * np.dot(w, w)
            score = norm_T2 - 2 * inner + norm_r1_2
            
            if score < best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None:
            return None
        
        T_work -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.dot(T_work.ravel(), T_work.ravel()) < 1e-10:
            break
    
    if np.dot(T_work.ravel(), T_work.ravel()) > 1e-8:
        return None
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    return U, V, W

def _to_lists(U, V, W):
    """Convert to lists, rounding to nearest 0.5 if close."""
    def to_list(A):
        result = []
        for row in A:
            r = []
            for val in row:
                hr = round(val * 2) / 2
                if abs(val - hr) < 1e-6:
                    r.append(hr)
                else:
                    r.append(float(val))
            result.append(r)
        return result
    return to_list(U), to_list(V), to_list(W)

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    start_time = time.time()
    
    # Schoolbook fallback
    U, V, W = _schoolbook()
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Try greedy peeling for various target ranks
    rng = np.random.RandomState(42)
    
    for target_rank in range(20, best_rank):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        found = False
        for restart in range(30):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            time_left = DEADLINE - 15 - (time.time() - start_time)
            time_per = min(2.0, time_left / max(1, 30 - restart))
            
            result = _try_greedy_peel(T, target_rank, rng, time_per)
            if result is not None:
                U2, V2, W2 = result
                if _verify(U2, V2, W2, T):
                    best_rank = target_rank
                    best_result = (U2, V2, W2)
                    found = True
                    break
        
        if found:
            continue  # Try even lower
        else:
            # Couldn't find this rank, but keep trying lower ranks with more restarts
            pass
    
    # Try with different seeds
    if time.time() - start_time < DEADLINE - 20:
        for seed in [123, 456, 789, 101, 202, 303, 404, 505]:
            if time.time() - start_time > DEADLINE - 15:
                break
            rng2 = np.random.RandomState(seed)
            for target_rank in range(20, best_rank):
                if time.time() - start_time > DEADLINE - 15:
                    break
                for restart in range(15):
                    if time.time() - start_time > DEADLINE - 15:
                        break
                    time_left = DEADLINE - 15 - (time.time() - start_time)
                    time_per = min(1.5, time_left / max(1, 15 - restart))
                    result = _try_greedy_peel(T, target_rank, rng2, time_per)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, T):
                            best_rank = target_rank
                            best_result = (U2, V2, W2)
                            break
                if best_rank == target_rank:
                    break
    
    U, V, W = best_result
    U_list, V_list, W_list = _to_lists(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
