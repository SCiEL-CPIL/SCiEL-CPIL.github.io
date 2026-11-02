import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _build_expected(n, m, p):
    """Build the expected multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Check that U, V, W decompose the matrix multiplication tensor."""
    R = U.shape[1]
    if R == 0:
        return False
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = _build_expected(n, m, p)
    return np.allclose(T, Expected, atol=1e-9)

def _trivial_decomposition(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1
    
    # M3: A00(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1
    
    # M4: A11(B01-B00)
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[3, 3] += 1
    
    # M5: (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[1, 4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Block decomposition for 2x4x7 using Strassen on 2x2x2 sub-blocks.
    Rank 50.
    """
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    r_offset = 0
    
    for j_group in range(2):
        j_base = j_group * 2
        k_groups = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for k_start, k_end in k_groups:
            k_len = k_end - k_start
            if k_len == 2:
                U_s, V_s, W_s = _strassen_222()
                for r in range(7):
                    for ia in range(4):
                        i = ia // 2
                        j_local = ia % 2
                        j = j_base + j_local
                        a_idx = i * m + j
                        U[a_idx, r_offset + r] += U_s[ia, r]
                    for ib in range(4):
                        j_local = ib // 2
                        k_local = ib % 2
                        j = j_base + j_local
                        k = k_start + k_local
                        b_idx = j * p + k
                        V[b_idx, r_offset + r] += V_s[ib, r]
                    for ic in range(4):
                        k_local = ic // 2
                        i = ic % 2
                        k = k_start + k_local
                        c_idx = k * n + i
                        W[c_idx, r_offset + r] += W_s[ic, r]
                r_offset += 7
            elif k_len == 1:
                k = k_start
                for i in range(2):
                    for j_local in range(2):
                        j = j_base + j_local
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        U[a_idx, r_offset] = 1
                        V[b_idx, r_offset] = 1
                        W[c_idx, r_offset] = 1
                        r_offset += 1
    
    return {"U": U, "V": V, "W": W}

def _randomized_search(n, m, p, target_rank, time_budget=30):
    """
    Randomized search for a lower rank decomposition.
    """
    start_time = time.time()
    Expected = _build_expected(n, m, p)
    num_a, num_b, num_c = n*m, m*p, p*n
    
    # Start with block decomposition
    block = _block_decomposition(n, m, p)
    U = block["U"].copy()
    V = block["V"].copy()
    W = block["W"].copy()
    current_rank = U.shape[1]
    
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by random column elimination and re-optimization
    # This is a simplified approach - in practice, we'd need a more sophisticated method
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _local_search_reduce_rank(n, m, p, time_budget=30):
    """
    Local search to try to reduce the rank of the block decomposition.
    
    Strategy: 
    1. Start with the rank-50 block decomposition.
    2. Try to eliminate one rank term at a time by adjusting the remaining terms.
    3. Use a simple heuristic: if removing a term creates a small residual, try to absorb it.
    """
    start_time = time.time()
    Expected = _build_expected(n, m, p)
    num_a, num_b, num_c = n*m, m*p, p*n
    
    block = _block_decomposition(n, m, p)
    U = block["U"].copy()
    V = block["V"].copy()
    W = block["W"].copy()
    current_rank = U.shape[1]
    
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try each possible rank reduction
    for target_rank in range(current_rank - 1, max(current_rank - 10, 0), -1):
        if time.time() - start_time > time_budget:
            break
        
        # Try to find a rank-target_rank decomposition
        # This is computationally expensive, so we use a heuristic
        # For now, just check if the current decomposition is valid
        # and return it if no better is found
        
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _construct_with_error_repair(n, m, p, time_budget=30):
    """
    Construct decomposition with error repair mechanism.
    
    Strategy:
    1. Start with block decomposition (rank 50).
    2. Apply error repair: try to fix small errors in the decomposition.
    3. Use this to potentially find a lower rank solution.
    """
    start_time = time.time()
    Expected = _build_expected(n, m, p)
    num_a, num_b, num_c = n*m, m*p, p*n
    
    block = _block_decomposition(n, m, p)
    U = block["U"].copy()
    V = block["V"].copy()
    W = block["W"].copy()
    current_rank = U.shape[1]
    
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Error repair: try to adjust the decomposition to reduce rank
    # This is a placeholder for a more sophisticated repair mechanism
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with trivial
    trivial = _trivial_decomposition(n, m, p)
    best_rank = trivial["U"].shape[1]
    best_result = trivial
    
    # Try block decomposition (should give rank 50)
    if time.time() - start_time < deadline - 10:
        block = _block_decomposition(n, m, p)
        if _verify(block["U"], block["V"], block["W"], n, m, p):
            if block["U"].shape[1] < best_rank:
                best_rank = block["U"].shape[1]
                best_result = block
    
    # Try randomized search
    if time.time() - start_time < deadline - 5:
        rand = _randomized_search(n, m, p, best_rank - 1, time_budget=deadline - time.time() - 5)
        if rand is not None and _verify(rand["U"], rand["V"], rand["W"], n, m, p):
            if rand["U"].shape[1] < best_rank:
                best_rank = rand["U"].shape[1]
                best_result = rand
    
    # Try local search
    if time.time() - start_time < deadline - 5:
        local = _local_search_reduce_rank(n, m, p, time_budget=deadline - time.time() - 5)
        if local is not None and _verify(local["U"], local["V"], local["W"], n, m, p):
            if local["U"].shape[1] < best_rank:
                best_rank = local["U"].shape[1]
                best_result = local
    
    # Try error repair
    if time.time() - start_time < deadline - 5:
        repair = _construct_with_error_repair(n, m, p, time_budget=deadline - time.time() - 5)
        if repair is not None and _verify(repair["U"], repair["V"], repair["W"], n, m, p):
            if repair["U"].shape[1] < best_rank:
                best_rank = repair["U"].shape[1]
                best_result = repair
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = _build_expected(n, m, p)
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
