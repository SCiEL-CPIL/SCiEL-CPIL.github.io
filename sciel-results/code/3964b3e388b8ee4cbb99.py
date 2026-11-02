"""Find a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication."""
import time
import random
import numpy as np

N, M, P = 2, 2, 8
NM, MP, PN = N * M, M * P, P * N  # 4, 16, 16

def _build_valid_tensor():
    """Return the 4x16x16 tensor with 1s at valid (i,j,k) positions."""
    T = np.zeros((NM, MP, PN), dtype=np.int8)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1
    return T

_VALID_TENSOR = _build_valid_tensor()

def verify(U, V, W):
    """Verify exact decomposition. U:(4,rank), V:(16,rank), W:(16,rank)."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    if U.shape[0] != NM or V.shape[0] != MP or W.shape[0] != PN:
        return False
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    if rank == 0:
        return False
    # Compute T = sum_r U[:,r] * V[:,r] * W[:,r]
    T = np.zeros((NM, MP, PN), dtype=np.int64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.array_equal(T, _VALID_TENSOR)

def schoolbook():
    """Rank-32 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((NM, rank), dtype=np.int64)
    V = np.zeros((MP, rank), dtype=np.int64)
    W = np.zeros((PN, rank), dtype=np.int64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def strassen_228():
    """
    Rank-28 decomposition: 4 independent 2x2*2x2 Strassen blocks.
    """
    # Standard Strassen 2x2*2x2 terms: (U_row, V_row, W_row) each length 4
    # U indices: (i,j) -> i*2+j
    # V indices: (j,k) -> j*2+k
    # W indices: (k,i) -> k*2+i
    strassen_terms = [
        # (U, V, W)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    rank = 28
    U = np.zeros((NM, rank), dtype=np.int64)
    V = np.zeros((MP, rank), dtype=np.int64)
    W = np.zeros((PN, rank), dtype=np.int64)
    
    for block in range(4):
        # block handles k in [2*block, 2*block+1]
        for t, (u, v, w) in enumerate(strassen_terms):
            col = block * 7 + t
            # U: only depends on (i,j), same for all blocks
            for i in range(N):
                for j in range(M):
                    U[i * M + j, col] = u[i * 2 + j]
            # V: depends on (j, k_local)
            for j in range(M):
                for kl in range(2):
                    kg = 2 * block + kl
                    V[j * P + kg, col] = v[j * 2 + kl]
            # W: depends on (k_local, i)
            for i in range(N):
                for kl in range(2):
                    kg = 2 * block + kl
                    W[kg * N + i, col] = w[kl * 2 + i]
    
    return U.tolist(), V.tolist(), W.tolist()

def _rank_search(target_rank, time_budget=60.0, seed=42):
    """
    Search for a decomposition of target_rank using random init + local refinement.
    Uses numpy for speed. Returns (U_list, V_list, W_list) or None.
    """
    deadline = time.time() + time_budget
    rng = np.random.RandomState(seed)
    
    best_err = np.inf
    best_U = best_V = best_W = None
    
    # Precompute valid tensor
    Tv = _VALID_TENSOR
    
    max_attempts = 30
    max_iters_per_attempt = 200
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Initialize with small random integers in {-1, 0, 1}
        U = rng.randint(-1, 2, size=(NM, target_rank)).astype(np.int64)
        V = rng.randint(-1, 2, size=(MP, target_rank)).astype(np.int64)
        W = rng.randint(-1, 2, size=(PN, target_rank)).astype(np.int64)
        
        for it in range(max_iters_per_attempt):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            T = np.zeros((NM, MP, PN), dtype=np.int64)
            for r in range(target_rank):
                T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.sum((T - Tv) ** 2)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err == 0:
                break
            
            # Coordinate descent: pick a random variable and perturb
            var = rng.randint(0, 3)
            if var == 0:
                a = rng.randint(0, NM)
                r = rng.randint(0, target_rank)
                U[a, r] += rng.randint(-1, 2)
            elif var == 1:
                b = rng.randint(0, MP)
                r = rng.randint(0, target_rank)
                V[b, r] += rng.randint(-1, 2)
            elif var == 2:
                c = rng.randint(0, PN)
                r = rng.randint(0, target_rank)
                W[c, r] += rng.randint(-1, 2)
            else:
                # Try a bigger perturbation: swap two columns
                r1 = rng.randint(0, target_rank)
                r2 = rng.randint(0, target_rank)
                if r1 != r2:
                    U[:, r1], U[:, r2] = U[:, r2].copy(), U[:, r1].copy()
                    V[:, r1], V[:, r2] = V[:, r2].copy(), V[:, r1].copy()
                    W[:, r1], W[:, r2] = W[:, r2].copy(), W[:, r1].copy()
        
        if best_err == 0:
            break
    
    if best_err == 0 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _structured_rank_27_search(time_budget=45.0, seed=123):
    """
    Search for rank-27 by taking the rank-28 Strassen decomposition and
    trying to merge/remove one term, or perturb and refine.
    """
    deadline = time.time() + time_budget
    rng = np.random.RandomState(seed)
    
    # Start from the rank-28 Strassen
    U28, V28, W28 = strassen_228()
    U28 = np.array(U28, dtype=np.int64)
    V28 = np.array(V28, dtype=np.int64)
    W28 = np.array(W28, dtype=np.int64)
    
    Tv = _VALID_TENSOR
    
    best_err = np.inf
    best_U = best_V = best_W = None
    
    # Strategy: try to find a rank-27 by perturbing rank-28 and doing local search
    # First, try removing one term and compensating with others
    for try_remove in range(28):
        if time.time() > deadline:
            break
        
        # Remove column try_remove, keep 27 columns
        mask = [i for i in range(28) if i != try_remove]
        U = U28[:, mask].copy()
        V = V28[:, mask].copy()
        W = W28[:, mask].copy()
        
        # Now do local refinement with 27 columns
        for it in range(100):
            if time.time() > deadline:
                break
            
            T = np.zeros((NM, MP, PN), dtype=np.int64)
            for r in range(27):
                T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.sum((T - Tv) ** 2)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err == 0:
                break
            
            # Perturb
            var = rng.randint(0, 3)
            if var == 0:
                a = rng.randint(0, NM)
                r = rng.randint(0, 27)
                U[a, r] += rng.randint(-1, 2)
            elif var == 1:
                b = rng.randint(0, MP)
                r = rng.randint(0, 27)
                V[b, r] += rng.randint(-1, 2)
            elif var == 2:
                c = rng.randint(0, PN)
                r = rng.randint(0, 27)
                W[c, r] += rng.randint(-1, 2)
            else:
                r1 = rng.randint(0, 27)
                r2 = rng.randint(0, 27)
                if r1 != r2:
                    U[:, r1], U[:, r2] = U[:, r2].copy(), U[:, r1].copy()
                    V[:, r1], V[:, r2] = V[:, r2].copy(), V[:, r1].copy()
                    W[:, r1], W[:, r2] = W[:, r2].copy(), W[:, r1].copy()
        
        if best_err == 0:
            break
    
    if best_err == 0 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # 1. Try rank-28 Strassen
    try:
        U, V, W = strassen_228()
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # 2. Try rank-27 structured search
    if time.time() < deadline:
        try:
            result = _structured_rank_27_search(time_budget=min(30.0, max(0.0, deadline - time.time())))
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # 3. Try rank-27 random search
    if time.time() < deadline:
        try:
            result = _rank_search(27, time_budget=min(30.0, max(0.0, deadline - time.time())), seed=7)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # 4. Try rank-28 random search (in case Strassen had a bug)
    if time.time() < deadline:
        try:
            result = _rank_search(28, time_budget=min(20.0, max(0.0, deadline - time.time())), seed=99)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # 5. Fallback to schoolbook (rank 32)
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
