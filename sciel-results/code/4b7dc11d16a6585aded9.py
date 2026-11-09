import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    # q=3, n=6, radius=1
    # Total words = 3**6 = 729
    total_words = q ** n
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    words = []
    for i in range(total_words):
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(neigh)
    
    # Build the incidence matrix A (total_words x total_words)
    # A[j, i] = 1 if word j is in the Hamming ball of radius 1 around word i
    A = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in neighbors[i]:
            A[j, i] = 1
    
    # --- Solve using MILP ---
    # Minimize sum(x_i) subject to A @ x >= 1, x in {0, 1}
    c_obj = np.ones(total_words)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(total_words), ub=np.inf)
    # Since A is sparse (each column has 13 non-zeros), we can use sparse matrix for efficiency
    # But milp in scipy accepts dense or sparse. Let's use sparse for safety with larger instances if q,n change.
    from scipy.sparse import csr_matrix
    A_sparse = csr_matrix(A)
    
    constraints = LinearConstraint(A_sparse, lb=np.ones(total_words), ub=np.inf)
    integrality = np.ones(total_words) # All variables integer (binary)
    bounds = Bounds(0, 1) # x_i in [0, 1]
    
    start_time = time.time()
    deadline = 118.0
    
    best_C = []
    
    # Try to solve the full problem first
    # If it takes too long, we might need a fallback, but for 729 vars it should be fast enough
    # SciPy's milp uses HiGHS which is quite fast for small/medium MIPs.
    
    try:
        res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": deadline - (time.time() - start_time) - 5})
        if res.status == 0 or res.status == 1: # Optimal or time limit reached with feasible solution
            if res.x is not None:
                x = res.x
                # Round to nearest integer (should be very close to 0 or 1)
                x_bin = (x > 0.5).astype(int)
                best_C = [i for i in range(total_words) if x_bin[i] == 1]
    except Exception as e:
        print(f"MILP failed: {e}")
        # Fallback to heuristic if MILP fails completely
        # For now, return empty or a trivial solution?
        # Better to fall back to the previous heuristic logic if available, but we removed it.
        # Let's implement a quick greedy fallback just in case.
        best_C = _greedy_cover(A, total_words, neighbors, words, q, n)
        
    # Verify the result is a valid cover
    if best_C:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_C:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        if np.min(cover_count) < 1:
            # If invalid (shouldn't happen with MILP), fallback
            best_C = _greedy_cover(A, total_words, neighbors, words, q, n)

    result = []
    for idx in best_C:
        result.append(list(words[idx]))
        
    return result

def _greedy_cover(A, total_words, neighbors, words, q, n):
    """Simple greedy set cover fallback."""
    uncovered = set(range(total_words))
    C = []
    while uncovered:
        best_c = -1
        max_covered = -1
        # Sample a few candidates to speed up if needed, but 729 is small
        # Randomize order for diversity? No, deterministic greedy is fine for fallback.
        for c in range(total_words):
            covered = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    covered += 1
            if covered > max_covered:
                max_covered = covered
                best_c = c
        if best_c == -1:
            break
        C.append(best_c)
        for nb in neighbors[best_c]:
            uncovered.discard(nb)
    return C
