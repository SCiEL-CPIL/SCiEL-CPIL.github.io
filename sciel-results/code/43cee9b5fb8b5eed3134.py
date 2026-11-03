import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n * m, m * p, p * n

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit

    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Baseline: rank-17 solution
    U17, V17, W17 = _build_rank17()

    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17

    # Try to find rank-16 solution
    if time.time() < deadline - 30:
        r16 = _search_rank16(T, U17, V17, W17, deadline)
        if r16 is not None:
            best_rank = 16
            best_U, best_V, best_W = r16

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_rank17():
    n, m, p = 2, 3, 3
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 17
    U = np.zeros((dim_U, rank), dtype=np.int64)
    V = np.zeros((dim_V, rank), dtype=np.int64)
    W = np.zeros((dim_W, rank), dtype=np.int64)

    # Strassen for A[0:2,0:2] * B[0:2,0:2]
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)*B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11*(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1

    # M6 = (A10-A00)*(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook: A[:,2]*B[2,:] -> C[:,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Schoolbook: A[:,0:2]*B[:,2] -> C[:,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1

    # Schoolbook: A[:,2]*B[2,2] -> C[:,2]
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W

def _search_rank16(T, U17, V17, W17, deadline):
    """
    Search for rank-16 by fixing the first 7 terms to Strassen
    and optimizing the remaining 9 terms.
    """
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 16
    num_strassen = 7
    num_free = rank - num_strassen # 9 terms

    # Extract Strassen part (first 7 columns of rank-17 solution)
    U_strassen = U17[:, :num_strassen].copy()
    V_strassen = V17[:, :num_strassen].copy()
    W_strassen = W17[:, :num_strassen].copy()

    # The target for the free part is T - T_strassen
    # T_strassen is the sum of rank-1 terms from Strassen
    T_strassen = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for r in range(num_strassen):
        T_strassen += np.einsum('a,b,c->abc', U_strassen[:, r], V_strassen[:, r], W_strassen[:, r])
    
    T_free = T - T_strassen

    # Initialize free part with schoolbook-like terms for the remaining indices
    # The remaining indices in T are those involving j=2 or k=2
    # We can just start with zeros and let the search find them,
    # or initialize with the schoolbook terms from rank-17 that are not Strassen.
    
    # Let's initialize with the schoolbook terms from rank 17 (cols 7-16)
    # But we only have 9 free slots, and rank 17 has 10 schoolbook terms.
    # We need to merge or drop one.
    
    U_free_init = np.zeros((dim_U, num_free), dtype=np.float64)
    V_free_init = np.zeros((dim_V, num_free), dtype=np.float64)
    W_free_init = np.zeros((dim_W, num_free), dtype=np.float64)
    
    # Copy first 9 schoolbook terms
    for r in range(num_free):
        U_free_init[:, r] = U17[:, 7+r].astype(np.float64)
        V_free_init[:, r] = V17[:, 7+r].astype(np.float64)
        W_free_init[:, r] = W17[:, 7+r].astype(np.float64)
        
    # The 10th schoolbook term (index 16 in rank 17) is dropped.
    # This creates an initial error.

    def compute_error(U_f, V_f, W_f):
        S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for r in range(num_free):
            S += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
        return np.sum((S - T_free) ** 2)

    current_U = U_free_init.copy()
    current_V = V_free_init.copy()
    current_W = W_free_init.copy()
    current_err = compute_error(current_U, current_V, current_W)
    
    # Local Search with Integer Steps
    np.random.seed(42)
    steps = 0
    max_steps = 200000
    temperature = 1.0
    
    while time.time() < deadline - 10 and steps < max_steps:
        steps += 1
        
        # Randomly select a free term and a matrix
        r = np.random.randint(num_free)
        mat = np.random.choice(3)
        
        if mat == 0:
            idx = np.random.randint(dim_U)
            old_val = current_U[idx, r]
            delta = np.random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            # Constrain to small integers
            if abs(new_val) > 3: continue
            current_U[idx, r] = new_val
            new_err = compute_error(current_U, current_V, current_W)
            if new_err < current_err:
                current_err = new_err
            else:
                current_U[idx, r] = old_val
        elif mat == 1:
            idx = np.random.randint(dim_V)
            old_val = current_V[idx, r]
            delta = np.random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            if abs(new_val) > 3: continue
            current_V[idx, r] = new_val
            new_err = compute_error(current_U, current_V, current_W)
            if new_err < current_err:
                current_err = new_err
            else:
                current_V[idx, r] = old_val
        else:
            idx = np.random.randint(dim_W)
            old_val = current_W[idx, r]
            delta = np.random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            if abs(new_val) > 3: continue
            current_W[idx, r] = new_val
            new_err = compute_error(current_U, current_V, current_W)
            if new_err < current_err:
                current_err = new_err
            else:
                current_W[idx, r] = old_val
                
        # Check for exact solution
        if current_err < 1e-9:
            # Verify with integers
            U_int = np.round(np.hstack([U_strassen, current_U])).astype(np.int64)
            V_int = np.round(np.hstack([V_strassen, current_V])).astype(np.int64)
            W_int = np.round(np.hstack([W_strassen, current_W])).astype(np.int64)
            
            # Final check
            S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
            for r in range(rank):
                S += np.einsum('a,b,c->abc', U_int[:, r], V_int[:, r], W_int[:, r])
            if np.allclose(S, T, atol=1e-9):
                return U_int, V_int, W_int
                
        # Occasional restart if stuck
        if steps % 5000 == 0:
            # Try a random restart for the free part
            current_U = np.random.randint(-1, 2, size=(dim_U, num_free)).astype(np.float64)
            current_V = np.random.randint(-1, 2, size=(dim_V, num_free)).astype(np.float64)
            current_W = np.random.randint(-1, 2, size=(dim_W, num_free)).astype(np.float64)
            current_err = compute_error(current_U, current_V, current_W)

    return None
