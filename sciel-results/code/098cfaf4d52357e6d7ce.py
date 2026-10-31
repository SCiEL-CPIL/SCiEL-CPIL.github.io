import math
import time

def construct():
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Known verified Golomb rulers
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
    ]
    
    # Try to extend n=6 ruler via local search
    def try_extend(base_u, max_n, time_limit=20):
        start_time = time.time()
        best_u = list(base_u)
        best_score = compute_score(best_u)
        
        for target_n in range(len(base_u) + 1, max_n + 1):
            if time.time() - start_time > time_limit:
                break
            improved = True
            while improved:
                improved = False
                if time.time() - start_time > time_limit:
                    break
                current_max = max(best_u)
                best_candidate = None
                best_candidate_score = best_score
                search_hi = current_max + target_n * target_n
                for new_mark in range(current_max + 1, search_hi + 1):
                    candidate = sorted(best_u + [new_mark])
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if time.time() - start_time > time_limit:
                        break
                if best_candidate is not None:
                    best_u = best_candidate
                    best_score = best_candidate_score
                    improved = True
        return best_u, best_score
    
    ext_u, _ = try_extend([0, 1, 4, 10, 12, 17], 10, time_limit=20)
    if ext_u and len(ext_u) > 6:
        candidates.append(ext_u)
    
    # Greedy Golomb ruler construction
    def greedy_golomb(n, time_limit=15):
        start_time = time.time()
        u = [0]
        for i in range(1, n):
            if time.time() - start_time > time_limit:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 20:
                    break
        return u
    
    greedy_u = greedy_golomb(10, time_limit=15)
    if greedy_u and len(greedy_u) >= 6:
        candidates.append(greedy_u)
    
    # Find best
    best_set = [0, 1, 3]
    best_score = 0.0
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u
    
    return best_set
