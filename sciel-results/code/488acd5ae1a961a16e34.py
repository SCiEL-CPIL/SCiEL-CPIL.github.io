import math
import time
import random
from itertools import combinations

def score(U):
    """Compute the score for a finite set U containing 0."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _fast_score(U):
    """Optimized score calculation using sets."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_with_fft(U):
    """Use FFT to compute sumset and diffset sizes for larger sets."""
    import numpy as np
    
    U_list = sorted(set(U))
    max_u = max(U_list)
    n = len(U_list)
    
    if n < 10 or max_u > 100000:
        return _fast_score(U_list)
    
    # Create indicator array
    size = 2 * max_u + 1
    if size > 200000:
        return _fast_score(U_list)
    
    arr = np.zeros(size, dtype=np.float64)
    for u in U_list:
        arr[u] = 1.0
    
    # FFT for sumset
    nfft = 1
    while nfft < 2 * size - 1:
        nfft *= 2
    fft_arr = np.fft.fft(arr, n=nfft)
    sum_fft = fft_arr * fft_arr
    sum_result = np.fft.ifft(sum_fft).real
    sumset_count = int(np.sum(sum_result > 0.5))
    
    # For diffset, use convolution with reversed array
    arr_rev = arr[::-1]
    fft_rev = np.fft.fft(arr_rev, n=nfft)
    diff_fft = fft_arr * fft_rev
    diff_result = np.fft.ifft(diff_fft).real
    diffset_count = int(np.sum(diff_result > 0.5))
    
    if sumset_count == 0:
        return 0.0
    
    ratio = diffset_count / sumset_count
    if ratio <= 1.0:
        return 1.0
    
    val = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _construct_golomb_ruler(n):
    """Construct a Golomb ruler of length n using greedy method."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    ruler = [0]
    used_diffs = set()
    
    for _ in range(n - 1):
        best_pos = None
        best_score = -1
        # Try positions from 1 to some upper bound
        max_try = len(ruler) * 10 + 10
        for pos in range(1, max_try + 1):
            new_diffs = set()
            valid = True
            for r in ruler:
                d = abs(pos - r)
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                # Prefer smaller positions to keep max small
                best_pos = pos
                break
        
        if best_pos is None:
            # Fallback: try larger positions
            for pos in range(max_try + 1, max_try * 3 + 1):
                new_diffs = set()
                valid = True
                for r in ruler:
                    d = abs(pos - r)
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    best_pos = pos
                    break
        
        if best_pos is None:
            break
        
        ruler.append(best_pos)
        used_diffs.update(abs(best_pos - r) for r in ruler[:-1])
    
    ruler.sort()
    return ruler

def _construct_powers(U_base, k):
    """Create sets based on powers structure."""
    candidates = []
    
    # Powers of 2
    for i in range(2, min(k + 1, 15)):
        U = [2**j for j in range(i + 1)]
        candidates.append(U)
    
    # 2^k - 1
    for i in range(2, min(k + 1, 15)):
        U = [2**j - 1 for j in range(i + 1)]
        if 0 in U:
            candidates.append(U)
    
    # Mixed: small AP + large gap
    for base_n in range(3, min(k + 1, 8)):
        for gap in [base_n, base_n * 2, base_n * 3, base_n * 5]:
            U = list(range(base_n)) + [gap]
            if len(set(U)) == len(U):
                candidates.append(sorted(U))
    
    return candidates

def _local_search(U, max_iters=50, deadline=None):
    """Improve set U via local search with multiple strategies."""
    U = sorted(set(U))
    best_score = _fast_score(U)
    best_U = U[:]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        max_u = max(best_U) if best_U else 0
        n = len(best_U)
        
        # Strategy 1: Move one element
        for i in range(1, n):
            old_val = best_U[i]
            # Try nearby values and some strategic ones
            candidates = set()
            for delta in range(-5, 6):
                v = old_val + delta
                if v >= 1:
                    candidates.add(v)
            candidates.add(old_val * 2)
            candidates.add(old_val // 2)
            candidates.add(max_u - old_val)
            candidates.add(max_u + 1)
            
            for new_val in sorted(candidates):
                if new_val == old_val or new_val < 1:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = sorted(set(new_U))
                    improved = True
                    break
            if improved:
                break
        
        # Strategy 2: Add an element
        if not improved:
            upper = max_u + max(10, n * 2)
            for _ in range(5):
                x = random.randint(1, upper)
                if x not in best_U:
                    new_U = best_U + [x]
                    s = _fast_score(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = sorted(set(new_U))
                        improved = True
                        break
        
        # Strategy 3: Remove an element
        if not improved and n > 2:
            for _ in range(3):
                idx = random.randint(1, n - 1)
                new_U = best_U[:]
                new_U.pop(idx)
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = sorted(set(new_U))
                    improved = True
                    break
        
        if not improved:
            break
    
    return sorted(set(best_U)), best_score

def _hill_climbing(U, max_iters=100, deadline=None):
    """Hill climbing with diverse operations."""
    current_set = sorted(set(U))
    current_score = _fast_score(current_set)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        max_u = max(current_set) if current_set else 0
        n = len(current_set)
        upper_bound = max_u * 2 + 20
        
        # Try several random operations
        for _ in range(10):
            if deadline and time.time() > deadline:
                break
            
            op = random.choice(['add', 'remove', 'replace'])
            
            if op == 'add' and n < 15:
                x = random.randint(1, upper_bound)
                if x not in current_set:
                    new_set = current_set + [x]
                    s = _fast_score(new_set)
                    if s > current_score + 1e-9:
                        current_set = sorted(set(new_set))
                        current_score = s
                        improved = True
                        break
                        
            elif op == 'remove' and n > 2:
                idx = random.randint(1, n - 1)
                new_set = current_set[:]
                new_set.pop(idx)
                s = _fast_score(new_set)
                if s > current_score + 1e-9:
                    current_set = sorted(set(new_set))
                    current_score = s
                    improved = True
                    break
                    
            elif op == 'replace' and n > 1:
                idx = random.randint(1, n - 1)
                new_val = random.randint(1, upper_bound)
                if new_val not in current_set:
                    new_set = current_set[:]
                    new_set[idx] = new_val
                    if len(set(new_set)) == len(new_set):
                        s = _fast_score(new_set)
                        if s > current_score + 1e-9:
                            current_set = sorted(set(new_set))
                            current_score = s
                            improved = True
                            break
        
        if not improved:
            break
    
    return sorted(set(current_set)), current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    random.seed(12345)
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # 1. Try known good structures
    known_structures = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 3, 7],
        [0, 1, 4, 10],
        [0, 2, 3, 7],
        [0, 1, 5, 6],
        [0, 1, 4, 9],
        [0, 2, 7, 8],
    ]
    
    for U in known_structures:
        if time.time() > deadline:
            break
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # 2. Generate Golomb rulers of various sizes
    for n in range(3, 10):
        if time.time() > deadline:
            break
        gr = _construct_golomb_ruler(n)
        if len(gr) == n:
            s = _fast_score(gr)
            if s > best_score:
                best_score = s
                best_set = gr[:]
    
    # 3. Generate power-based structures
    power_candidates = _construct_powers(None, 12)
    for U in power_candidates:
        if time.time() > deadline:
            break
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # 4. Random initial sets
    for _ in range(30):
        if time.time() > deadline:
            break
        n = random.randint(3, 8)
        max_val = random.randint(10, 100)
        U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # 5. Local search on best
    if time.time() < deadline - 20:
        best_set, best_score = _local_search(best_set, max_iters=50, deadline=deadline)
    
    # 6. Hill climbing for further improvement
    if time.time() < deadline - 10:
        best_set, best_score = _hill_climbing(best_set, max_iters=100, deadline=deadline)
    
    # 7. Try FFT-based scoring for potentially larger sets
    if time.time() < deadline - 5:
        # Try some larger structured sets with FFT scoring
        for n in range(8, 15):
            if time.time() > deadline - 5:
                break
            gr = _construct_golomb_ruler(n)
            if len(gr) >= n - 1:
                s = _score_with_fft(gr)
                if s > best_score:
                    best_score = s
                    best_set = gr[:]
    
    # 8. Final polish with local search
    if time.time() < deadline - 2:
        best_set, best_score = _local_search(best_set, max_iters=20, deadline=deadline)
    
    return sorted(set(best_set))
