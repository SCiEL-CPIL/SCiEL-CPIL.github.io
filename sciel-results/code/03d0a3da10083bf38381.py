import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    def is_golomb(u):
        """Check if u is a Golomb ruler (all pairwise differences distinct)."""
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def local_search(base_u, time_limit):
        """Local search: try adding, removing, or swapping elements while maintaining Golomb property."""
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        improved = True
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score_local
            
            # Try adding a new element
            current_max = max(best_u)
            n = len(best_u)
            search_hi = current_max + n * n * 2
            for new_mark in range(current_max + 1, search_hi + 1):
                candidate = sorted(best_u + [new_mark])
                if not is_golomb(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                if (time.time() - start_time) > time_limit:
                    break
            
            # Try removing an element (if size > 3)
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try swapping: replace one element with another
            if best_u and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # don't swap 0
                    old_val = best_u[i]
                    # Try replacing with nearby values
                    for delta in range(1, max(10, len(best_u) * 3)):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0:
                                continue
                            if new_val in best_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            if not is_golomb(candidate):
                                continue
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                        if (time.time() - start_time) > time_limit:
                            break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score_local = best_candidate_score
                improved = True
        
        return best_u, best_score_local

    # Known verified Golomb rulers (optimal or near-optimal)
    known_golomb = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        [0, 1, 4, 10, 12, 17, 20, 22, 24],
        [0, 1, 4, 10, 12, 17, 20, 22, 24, 26],
    ]
    
    # First, evaluate known Golomb rulers
    for gu in known_golomb:
        try_set(gu)
    
    # Run local search from multiple starting points
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 6],
    ]
    
    for sp in start_points:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        # Allocate time proportionally
        time_for_this = min(remaining * 0.25, 30)
        if time_for_this < 5:
            time_for_this = max(5, remaining * 0.5)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            try_set(ext_u)
    
    # Exhaustive search for small sets to ensure we don't miss anything
    max_n = 6
    max_m = 30
    
    for n in range(2, max_n + 1):
        if time.time() > deadline - 15:
            break
        # Choose n-1 elements from 1..max_m
        for combo in combinations(range(1, max_m + 1), n - 1):
            if time.time() > deadline - 15:
                break
            cand = [0] + list(combo)
            try_set(cand)
    
    return list(best_set)
