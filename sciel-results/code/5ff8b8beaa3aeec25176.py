"""Construct finite set U of nonnegative integers containing 0 to maximize
the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1) on C6."""
import time
import math
from itertools import combinations

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset and difference set sizes
    # For efficiency with small sets, use sets
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_count = len(sumset)
    diff_count = len(diffset)
    
    if sum_count == 0:
        return 0.0
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return 1.0  # The log term would be <= 0, so score is at most 1
    
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term

def _exhaustive_search_small(n_max, max_val, deadline):
    """Exhaustive search over small sets of size up to n_max with elements up to max_val."""
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Start with known good small sets
    candidates = [
        [0, 1, 2, 5],
        [0, 1, 3],
        [0, 1, 2],
        [0, 1, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 4, 5],
        [0, 1, 2, 4],
        [0, 1, 2, 3, 6],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 5, 6],
        [0, 1, 2, 4, 6],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 4, 7],
        [0, 1, 2, 3, 5, 6],
        [0, 1, 2, 4, 5, 7],
        [0, 1, 3, 4, 5, 6],
    ]
    
    for cand in candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Try to improve via local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current = list(best_set)
        # Try adding elements
        for new_elem in range(0, max_val + 1):
            if new_elem in current:
                continue
            candidate = current + [new_elem]
            candidate.sort()
            s = _score_set(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                improved = True
                break
        # Try removing elements (except 0)
        if not improved:
            for i in range(1, len(best_set)):
                candidate = best_set[:i] + best_set[i+1:]
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    improved = True
                    break
        # Try replacing elements
        if not improved:
            for i in range(1, len(best_set)):
                for new_elem in range(0, max_val + 1):
                    if new_elem in best_set:
                        continue
                    candidate = list(best_set)
                    candidate[i] = new_elem
                    candidate.sort()
                    s = _score_set(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = candidate
                        improved = True
                        break
                if improved:
                    break
    
    return best_set, best_score

def _construct_via_multiplicative_structure(max_size, deadline):
    """
    Try constructing sets based on multiplicative structures in finite fields.
    The idea: elements in a multiplicative subgroup of F_p have structured
    sum and difference sets. We look for small representatives.
    """
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Try sets of the form {0} union {a^k mod p for k=0..m-1} mapped to integers
    # Actually, let's try a different approach: use the structure of
    # sets where U+U and U-U have known sizes.
    
    # For a set U, |U+U| >= 2|U|-1 and |U-U| >= 2|U|-1.
    # The ratio |U-U|/|U+U| is maximized when U-U is large and U+U is small.
    # Arithmetic progressions: U = {0, 1, ..., n-1}, |U+U| = 2n-1, |U-U| = 2n-1, ratio = 1.
    # Sets with more "spread": try U = {0, 1, 3, 6, 10, ...} (triangular numbers)
    # or U = {0, 1, 2, 4, 8, ...} (powers of 2)
    
    # Powers of 2: U = {0, 1, 2, 4, 8, ..., 2^(k-1)}
    # |U+U| = k*(k+1)/2 + ... actually for powers of 2, sums are unique in binary
    # So |U+U| = number of distinct sums = C(k+1, 2) + 1? No.
    # For U = {0, 1, 2, 4, ..., 2^(k-1)}, sums a+b where a,b in U.
    # Each sum is unique because of binary representation (carry-free for distinct bits,
    # but 2^i + 2^i = 2^(i+1) causes carries).
    # Actually this gets complicated. Let's just compute.
    
    for k in range(2, 12):
        U = [2**i for i in range(k)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Try other structured sets
    structured_sets = []
    # Triangular numbers
    for k in range(2, 15):
        U = [i*(i+1)//2 for i in range(k)]
        structured_sets.append(U)
    # Fibonacci-like
    for k in range(2, 15):
        U = [0]
        a, b = 1, 1
        for i in range(k-1):
            U.append(a)
            a, b = b, a+b
        structured_sets.append(U)
    
    for U in structured_sets:
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    return best_set, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Try known good small sets and local search
    if time.time() < deadline - 10:
        set1, score1 = _exhaustive_search_small(8, 30, deadline - 10)
        if score1 > best_score:
            best_score = score1
            best_set = set1
    
    # Phase 2: Try structured constructions
    if time.time() < deadline - 5:
        set2, score2 = _construct_via_multiplicative_structure(12, deadline - 5)
        if score2 > best_score:
            best_score = score2
            best_set = set2
    
    # Phase 3: More aggressive local search from the best found
    if time.time() < deadline - 2:
        current = list(best_set)
        improved = True
        max_val = max(current) + 20
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding
            for new_elem in range(0, max_val + 1):
                if new_elem in current:
                    continue
                candidate = current + [new_elem]
                candidate.sort()
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    current = list(candidate)
                    max_val = max(current) + 10
                    improved = True
                    break
            # Try replacing
            if not improved:
                for i in range(1, len(current)):
                    for new_elem in range(0, max_val + 1):
                        if new_elem in current:
                            continue
                        candidate = list(current)
                        candidate[i] = new_elem
                        candidate.sort()
                        s = _score_set(candidate)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = candidate
                            current = list(candidate)
                            improved = True
                            break
                    if improved:
                        break
            # Try removing
            if not improved:
                for i in range(1, len(current)):
                    candidate = current[:i] + current[i+1:]
                    s = _score_set(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = candidate
                        current = list(candidate)
                        improved = True
                        break
    
    return best_set
