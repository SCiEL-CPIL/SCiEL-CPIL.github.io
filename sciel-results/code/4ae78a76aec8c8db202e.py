"""Rewritten program for integer sum-product extremal sets."""
import time
import random
from itertools import combinations

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for a set of distinct positive integers."""
    A = list(A)
    sumset = set()
    prodset = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def _generate_2smooth_candidates():
    """Generate sets of 20 distinct 2-smooth numbers (of form 2^a * 3^b) <= 10000."""
    vals = set()
    a = 0
    while 2**a <= 10000:
        b = 0
        v = 2**a
        while v <= 10000:
            vals.add(v)
            b += 1
            v = 2**a * 3**b
        a += 1
    vals = sorted(vals)
    return vals

def _generate_3smooth_candidates():
    """Generate sets of 20 distinct 3-smooth numbers (of form 2^a * 3^b * 5^c) <= 10000."""
    vals = set()
    a = 0
    while 2**a <= 10000:
        b = 0
        v2 = 2**a
        while v2 <= 10000:
            c = 0
            v = v2
            while v <= 10000:
                vals.add(v)
                c += 1
                v = v2 * 5**c
            b += 1
            v2 = 2**a * 3**b
        a += 1
    vals = sorted(vals)
    return vals

def _local_search(A, max_iters=5000, timeout=55.0):
    """Local search: try replacing elements with nearby values to reduce score."""
    A = sorted(A)
    current_score = _calc_score(A)
    start = time.time()
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time.time() - start > timeout:
            break
        improved = False
        for i in range(len(A)):
            if time.time() - start > timeout:
                break
            old_val = A[i]
            # Try nearby values
            for delta in range(1, 50):
                new_val = old_val + delta
                if new_val > 10000:
                    break
                if new_val in A:
                    continue
                A[i] = new_val
                new_score = _calc_score(A)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                A[i] = old_val
            # Also try smaller values
            for delta in range(1, 50):
                new_val = old_val - delta
                if new_val < 1:
                    break
                if new_val in A:
                    continue
                A[i] = new_val
                new_score = _calc_score(A)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                A[i] = old_val
        iters += 1
    return sorted(A), current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Initial candidates
    candidates = []
    
    # 1. Consecutive integers 1..20
    candidates.append(list(range(1, 21)))
    
    # 2. Powers of 2
    candidates.append([2**i for i in range(20)])
    
    # 3. Small 2-smooth numbers
    smooth2 = _generate_2smooth_candidates()
    if len(smooth2) >= 20:
        candidates.append(smooth2[:20])
        # Also try a different subset
        candidates.append(smooth2[:10] + smooth2[-10:])
    
    # 4. Small 3-smooth numbers
    smooth3 = _generate_3smooth_candidates()
    if len(smooth3) >= 20:
        candidates.append(smooth3[:20])
        candidates.append(smooth3[:10] + smooth3[-10:])
    
    # 5. Geometric with ratio ~1.5
    geo = [int(round(1.5**i)) for i in range(20)]
    # Ensure distinct and <= 10000
    geo = sorted(set(geo))
    if len(geo) >= 20:
        candidates.append(geo[:20])
    
    # 6. Mix of small numbers and powers
    mix1 = list(range(1, 11)) + [2**i for i in range(3, 13)]
    mix1 = sorted(set(x for x in mix1 if x <= 10000))
    if len(mix1) >= 20:
        candidates.append(mix1[:20])
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        # Ensure validity: 20 distinct positive integers <= 10000
        cand = sorted(set(x for x in cand if 1 <= x <= 10000))
        if len(cand) < 20:
            continue
        score = _calc_score(cand)
        if score < best_score:
            best_score = score
            best_A = list(cand)
    
    # Local search on the best candidate
    if best_A is not None:
        if time.time() < deadline - 5.0:
            best_A, best_score = _local_search(best_A, max_iters=10000, timeout=deadline - time.time() - 2.0)
    
    # Try a few random restarts if time permits
    if best_A is not None and time.time() < deadline - 10.0:
        random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            # Generate a random set
            A_rand = random.sample(range(1, 10001), 20)
            A_rand = sorted(A_rand)
            score_rand = _calc_score(A_rand)
            if score_rand < best_score:
                best_score = score_rand
                best_A = list(A_rand)
                # Local search
                if time.time() < deadline - 5.0:
                    best_A, best_score = _local_search(best_A, max_iters=2000, timeout=deadline - time.time() - 3.0)
    
    if best_A is None:
        # Fallback
        best_A = list(range(1, 21))
    
    return best_A

# EVOLVE-BLOCK-END
