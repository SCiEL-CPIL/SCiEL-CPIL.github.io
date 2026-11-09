"""Greedy covering-code search with parameterized heuristic and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_distance_matrix(words, q, n, radius):
    """
    Build a boolean matrix A where A[i, j] is True if word i is within
    Hamming distance radius of word j.
    words: list of lists
    """
    N = len(words)
    words_arr = np.array(words, dtype=np.int8)
    diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    A = (dists <= radius).astype(np.int8)
    return A


def _solve_milp(words, q, n, radius, time_limit=100):
    """
    Solve the set covering problem using MILP.
    Minimize sum(x_j) subject to sum_j(A[i, j] * x_j) >= 1 for all i.
    """
    N = len(words)
    A = _build_distance_matrix(words, q, n, radius)
    
    c = np.ones(N)
    cons = LinearConstraint(A, lb=1, ub=np.inf)
    integrality = np.ones(N)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        
        if res.status == 0:
            x = res.x.astype(int)
            code = [words[j] for j in range(N) if x[j] == 1]
            return code
        else:
            return None
    except Exception:
        return None


def _greedy_fallback(words, q, n, radius):
    """
    Fast greedy heuristic to produce a valid (though possibly suboptimal) code.
    """
    N = len(words)
    words_arr = np.array(words, dtype=np.int8)
    covered = np.zeros(N, dtype=bool)
    code = []
    
    diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    valid = (dists <= radius)
    
    while not covered.all():
        un_cov_mask = ~covered
        new_counts = (valid & un_cov_mask[np.newaxis, :]).sum(axis=1)
        
        best_idx = np.argmax(new_counts)
        if new_counts[best_idx] == 0:
            break
            
        code.append(list(words[best_idx]))
        covered |= valid[best_idx, :]
        
    return code


def _local_search_polish(code, words, q, n, radius, time_limit=30):
    """
    Polish a valid code by removing redundant codewords and trying swaps.
    Returns a potentially smaller valid code.
    """
    start = time.time()
    N = len(words)
    words_arr = np.array(words, dtype=np.int8)
    code_indices = []
    
    # Map code words to their indices in words
    word_to_idx = {}
    for i, w in enumerate(words):
        word_to_idx[tuple(w)] = i
    
    for cw in code:
        idx = word_to_idx.get(tuple(cw))
        if idx is not None:
            code_indices.append(idx)
    
    # Build coverage matrix: cover[i, j] = True if word i is covered by codeword j
    code_arr = np.array([words[i] for i in code_indices], dtype=np.int8)
    diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    cover = (dists <= radius)
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for ci in range(len(code_indices)):
            if (time.time() - start) > time_limit:
                break
            # Check if removing code_indices[ci] still covers everything
            remaining_cover = cover.copy()
            remaining_cover[:, ci] = False
            if remaining_cover.any(axis=1).all():
                # Safe to remove
                code_indices.pop(ci)
                # Rebuild cover matrix
                code_arr = np.array([words[i] for i in code_indices], dtype=np.int8)
                diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
                dists = diff.sum(axis=2)
                cover = (dists <= radius)
                improved = True
                break
    
    # Phase 2: Try swapping one codeword for another to enable further deletions
    for _ in range(50):  # Limited iterations
        if (time.time() - start) > time_limit:
            break
        
        best_swap = None
        best_size = len(code_indices)
        
        for ci in range(len(code_indices)):
            if (time.time() - start) > time_limit:
                break
            for new_idx in range(N):
                if new_idx in code_indices:
                    continue
                
                # Try replacing code_indices[ci] with new_idx
                trial_indices = code_indices.copy()
                trial_indices[ci] = new_idx
                trial_arr = np.array([words[i] for i in trial_indices], dtype=np.int8)
                diff = words_arr[:, np.newaxis, :] != trial_arr[np.newaxis, :, :]
                dists = diff.sum(axis=2)
                trial_cover = (dists <= radius)
                
                # Check if valid
                if not trial_cover.any(axis=1).all():
                    continue
                
                # Try to remove more codewords from this trial
                trial_indices_copy = trial_indices.copy()
                improved_trial = True
                while improved_trial and (time.time() - start) < time_limit:
                    improved_trial = False
                    for ri in range(len(trial_indices_copy)):
                        if (time.time() - start) > time_limit:
                            break
                        rc = trial_cover.copy()
                        rc[:, ri] = False
                        if rc.any(axis=1).all():
                            trial_indices_copy.pop(ri)
                            # Rebuild
                            ta = np.array([words[i] for i in trial_indices_copy], dtype=np.int8)
                            d = words_arr[:, np.newaxis, :] != ta[np.newaxis, :, :]
                            dc = (d.sum(axis=2) <= radius)
                            trial_cover = dc
                            improved_trial = True
                            break
                
                if len(trial_indices_copy) < best_size:
                    best_size = len(trial_indices_copy)
                    best_swap = (ci, new_idx, trial_indices_copy)
        
        if best_swap is not None:
            ci, new_idx, new_code_indices = best_swap
            code_indices = new_code_indices
            # Rebuild cover
            code_arr = np.array([words[i] for i in code_indices], dtype=np.int8)
            diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
            dists = diff.sum(axis=2)
            cover = (dists <= radius)
        else:
            break
    
    return [words[i] for i in code_indices]


def _verify_code(code, q, n, radius):
    """Verify that every word is within radius of some codeword."""
    total = q ** n
    if total > 50000:
        rng = np.random.default_rng(42)
        for _ in range(10000):
            w = list(rng.integers(0, q, size=n))
            found = False
            for cw in code:
                if _hamming(w, cw) <= radius:
                    found = True
                    break
            if not found:
                return False
        return True
    else:
        words_list = [_idx_to_word(i, n, q) for i in range(total)]
        words_arr = np.array(words_list, dtype=np.int8)
        code_arr = np.array(code, dtype=np.int8)
        
        diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
        dists = diff.sum(axis=2)
        covered = (dists <= radius).any(axis=1)
        return covered.all()


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        return [[0] * n]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    
    best_code = None
    best_size = float('inf')
    
    # 1. Try MILP
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 15  # Reserve 15s for polish and verification
    
    if remaining > 10:
        try:
            code = _solve_milp(words_list, q, n, radius, time_limit=remaining)
            if code is not None and len(code) > 0:
                if _verify_code(code, q, n, radius):
                    best_code = code
                    best_size = len(code)
        except Exception:
            pass
    
    # 2. Try greedy if MILP didn't give a good result or failed
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 15
    
    if remaining > 5:
        try:
            code = _greedy_fallback(words_list, q, n, radius)
            if _verify_code(code, q, n, radius):
                if best_code is None or len(code) < best_size:
                    best_code = code
                    best_size = len(code)
        except Exception:
            pass
    
    # 3. Polish the best code found so far
    if best_code is not None:
        elapsed = time.time() - start_time
        remaining = _DEADLINE - elapsed - 5
        
        if remaining > 5:
            try:
                polished = _local_search_polish(best_code, words_list, q, n, radius, time_limit=remaining)
                if _verify_code(polished, q, n, radius) and len(polished) < best_size:
                    best_code = polished
                    best_size = len(polished)
            except Exception:
                pass
    
    # 4. Fallback to Tetracode construction if we still don't have a code
    if best_code is None:
        try:
            code = _direct_sum_tetracode(q, n, radius)
            if _verify_code(code, q, n, radius):
                best_code = code
        except Exception:
            pass
    
    # 5. Last resort
    if best_code is None:
        best_code = words_list
    
    return best_code

# EVOLVE-BLOCK-END
