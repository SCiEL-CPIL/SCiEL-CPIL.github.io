import numpy as np
import time

N = 50

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return l2_sq / autoconv_max

def _is_sidon(S):
    """Check if S is a Sidon set (all pairwise sums i+j for i<=j are distinct)."""
    sums = set()
    for i_idx, i in enumerate(S):
        for j_idx, j in enumerate(S):
            if j_idx >= i_idx:
                sm = i + j
                if sm in sums:
                    return False
                sums.add(sm)
    return True

def _find_sidon_set(n, max_size=12, n_restarts=500):
    """
    Find a large Sidon set in [0, n-1] using randomized greedy with multiple strategies.
    """
    best_S = []
    best_size = 0
    
    for restart in range(n_restarts):
        S = []
        sums = set()
        
        # Different ordering strategies
        if restart % 4 == 0:
            candidates = list(range(n))
            np.random.shuffle(candidates)
        elif restart % 4 == 1:
            # Prefer spread out elements
            mid = n // 2
            left = list(range(0, mid))
            right = list(range(mid, n))
            np.random.shuffle(left)
            np.random.shuffle(right)
            candidates = []
            li, ri = 0, 0
            while li < len(left) or ri < len(right):
                if ri >= len(right) or (li < len(left) and np.random.random() < 0.5):
                    candidates.append(left[li])
                    li += 1
                else:
                    candidates.append(right[ri])
                    ri += 1
        elif restart % 4 == 2:
            candidates = list(range(n))
            np.random.shuffle(candidates)
        else:
            candidates = list(range(n))
            np.random.shuffle(candidates)
        
        for i in candidates:
            # Check if i can be added to S
            valid = True
            new_sums = set()
            
            for j in S:
                s = i + j
                if s in sums:
                    valid = False
                    break
                new_sums.add(s)
            
            if not valid:
                continue
            
            s_self = 2 * i
            if s_self in sums or s_self in new_sums:
                continue
            
            S.append(i)
            sums.update(new_sums)
            sums.add(s_self)
            
            if len(S) > best_size:
                best_S = S[:]
                best_size = len(S)
                if best_size >= max_size:
                    break
        
        if best_size >= max_size:
            break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Start with uniform
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Find a large Sidon set
    sidon_set = _find_sidon_set(N, max_size=12, n_restarts=500)
    
    if len(sidon_set) > 0:
        h_sparse = np.zeros(N)
        h_sparse[sidon_set] = 1.0 / len(sidon_set)
        score_sparse = _score_raw(h_sparse)
        
        if score_sparse > best_score:
            best_score = score_sparse
            best_h = h_sparse.copy()
    
    # 2. Try to find even larger Sidon sets
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        
        s = _find_sidon_set(N, max_size=12, n_restarts=200)
        if len(s) > len(sidon_set):
            sidon_set = s
            h_sparse = np.zeros(N)
            h_sparse[s] = 1.0 / len(s)
            score_sparse = _score_raw(h_sparse)
            
            if score_sparse > best_score:
                best_score = score_sparse
                best_h = h_sparse.copy()
    
    # 3. Try adding elements to the Sidon set
    if time.time() < deadline - 5:
        s = list(sidon_set)
        for candidate in range(N):
            if candidate in s:
                continue
            test_s = s + [candidate]
            if _is_sidon(test_s):
                h_test = np.zeros(N)
                h_test[test_s] = 1.0 / len(test_s)
                score_test = _score_raw(h_test)
                if score_test > best_score:
                    best_score = score_test
                    best_h = h_test.copy()
                    sidon_set = test_s
                    s = list(sidon_set)
    
    return list(best_h)
