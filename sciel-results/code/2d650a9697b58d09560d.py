import math
import time
import numpy as np

def construct():
    DEADLINE = 140  # seconds, leave 10s safety margin
    _t0 = time.time()
    
    def elapsed():
        return time.time() - _t0
    
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        arr = np.array(u, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        """Fast Golomb check using sorted differences."""
        u = sorted(u)
        n = len(u)
        if n < 3:
            return True
        # Collect all pairwise differences and check for duplicates
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(u[j] - u[i])
        diffs.sort()
        for k in range(1, len(diffs)):
            if diffs[k] == diffs[k-1]:
                return False
        return True
    
    def is_golomb_fast(u, existing_diffs):
        """Check if adding u to a set with known diffs keeps it Golomb.
        existing_diffs should be a set of all pairwise diffs of the original set.
        Returns (True, updated_diffs) or (False, None)."""
        new_diffs = set(existing_diffs)
        for x in u:
            for d in existing_diffs:
                if d in new_diffs:
                    return False, None
                new_diffs.add(d)
        # Check diffs among new elements
        nu = len(u)
        for i in range(nu):
            for j in range(i+1, nu):
                d = abs(u[j] - u[i])
                if d in new_diffs:
                    return False, None
                new_diffs.add(d)
        return True, new_diffs
    
    # Known verified Golomb rulers (optimal or near-optimal)
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        # Additional diverse starting points
        [0, 2, 7, 15, 22, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 37],
        [0, 3, 7, 12, 20, 29, 38],
    ]
    
    # Local search with faster Golomb checking and better time management
    def local_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score = compute_score(best_u)
        improved = True
        
        # Precompute diffs for fast checking
        def get_diffs(u):
            u = sorted(u)
            diffs = set()
            for i in range(len(u)):
                for j in range(i+1, len(u)):
                    diffs.add(u[j] - u[i])
            return diffs
        
        current_diffs = get_diffs(best_u)
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score
            best_candidate_diffs = None
            
            # Try adding a new element
            current_max = max(best_u)
            n = len(best_u)
            search_hi = current_max + max(15, n * n)
            for new_mark in range(current_max + 1, search_hi + 1):
                # Fast check: new mark must not create duplicate diffs
                valid = True
                for d in current_diffs:
                    if d in current_diffs:  # always true, but we need to check new diffs
                        pass
                # Check if new_mark - x creates a duplicate diff for any x in best_u
                # and if new_mark - x == y - z for existing diffs
                new_diffs_needed = set()
                for x in best_u:
                    new_diffs_needed.add(new_mark - x)
                # Check intersection with existing diffs
                if new_diffs_needed & current_diffs:
                    continue
                # Check if any two new diffs are equal (they won't be since new_mark is unique)
                # But check if new_mark - x1 == new_mark - x2 which implies x1==x2, impossible
                # So just check the union
                candidate_diffs = current_diffs | new_diffs_needed
                # Also need to check that new diffs don't conflict with each other
                # (they can't since new_mark - x is distinct for distinct x)
                candidate = sorted(best_u + [new_mark])
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                    best_candidate_diffs = candidate_diffs
                if (time.time() - start_time) > time_limit:
                    break
            
            # Try removing an element (if size > 3)
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    removed = best_u[i]
                    # Remove diffs involving removed element
                    new_diffs = set()
                    for d in current_diffs:
                        # Keep d if it doesn't involve removed element
                        # We need to check: d = a-b where a,b in best_u, a>b
                        # d involves removed if removed-a or removed-b is in best_u appropriately
                        # Simpler: just recompute
                        pass
                    # Recompute is simpler and still fast for small n
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                        best_candidate_diffs = get_diffs(candidate)
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try swapping: replace one element with another
            if best_u and (time.time() - start_time) < time_limit:
                max_delta = max(15, len(best_u) * 3)
                for i in range(1, len(best_u)):  # don't swap 0
                    old_val = best_u[i]
                    # Compute diffs not involving old_val
                    other_diffs = set()
                    for d in current_diffs:
                        # Check if d involves old_val
                        involves_old = False
                        for x in best_u:
                            if x != old_val:
                                if old_val - x == d or x - old_val == d:
                                    involves_old = True
                                    break
                        if not involves_old:
                            other_diffs.add(d)
                    
                    for delta in range(1, max_delta + 1):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0 or new_val in best_u:
                                continue
                            # New diffs involving new_val
                            new_diffs = set()
                            for x in best_u:
                                if x != old_val:
                                    new_diffs.add(abs(new_val - x))
                            # Check conflicts
                            if new_diffs & other_diffs:
                                continue
                            if len(new_diffs) != len(new_diffs):  # duplicates among new
                                continue
                            # Check new_diffs internal duplicates
                            if len(new_diffs) < sum(1 for x in best_u if x != old_val):
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                                best_candidate_diffs = other_diffs | new_diffs
                        if (time.time() - start_time) > time_limit:
                            break
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try 2-swap: replace two elements with two new values (small perturbations)
            if best_u and len(best_u) >= 4 and (time.time() - start_time) < time_limit:
                n = len(best_u)
                for i in range(1, n):
                    for j in range(i + 1, n):
                        if (time.time() - start_time) > time_limit:
                            break
                        old_i, old_j = best_u[i], best_u[j]
                        for di in [-1, 1]:
                            for dj in [-1, 1]:
                                new_i, new_j = old_i + di, old_j + dj
                                if new_i < 0 or new_j < 0:
                                    continue
                                if new_i in best_u or new_j in best_u:
                                    continue
                                if new_i == new_j:
                                    continue
                                candidate = sorted([x if x != old_i else new_i for x in best_u])
                                candidate = sorted([x if x != old_j else new_j for x in candidate])
                                if len(set(candidate)) != len(candidate):
                                    continue
                                s = compute_score(candidate)
                                if s > best_candidate_score:
                                    best_candidate_score = s
                                    best_candidate = candidate
                                    best_candidate_diffs = get_diffs(candidate)
                        if (time.time() - start_time) > time_limit:
                            break
                    if (time.time() - start_time) > time_limit:
                        break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score = best_candidate_score
                if best_candidate_diffs is not None:
                    current_diffs = best_candidate_diffs
                else:
                    current_diffs = get_diffs(best_u)
                improved = True
        
        return best_u, best_score
    
    # Run local search on multiple starting points
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 9, 11],
        [0, 2, 7, 15, 22, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 37],
    ]
    
    # Sort start points by initial score (best first) to get good results early
    scored_starts = []
    for sp in start_points:
        s = compute_score(sp)
        scored_starts.append((s, sp))
    scored_starts.sort(key=lambda x: -x[0])
    
    for s, sp in scored_starts:
        remaining = DEADLINE - elapsed()
        if remaining < 5:
            break
        # Allocate time: give more to better starts
        time_for_this = min(remaining * 0.3, 30)
        if time_for_this < 5:
            time_for_this = max(5, remaining * 0.4)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            candidates.append(ext_u)
    
    # Greedy Golomb ruler construction
    def greedy_golomb(n, time_limit):
        start_time = time.time()
        u = [0]
        for i in range(1, n):
            if (time.time() - start_time) > time_limit:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 30:
                    break
        return u
    
    if DEADLINE - elapsed() > 8:
        greedy_u = greedy_golomb(10, time_limit=min(8, DEADLINE - elapsed() - 2))
        if greedy_u and len(greedy_u) >= 6:
            candidates.append(greedy_u)
    
    # Find best among all candidates
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
    
    # Final polish: do a focused local search on the best with wider range
    remaining = DEADLINE - elapsed()
    if remaining > 10 and best_score > 0:
        polish_u, polish_score = local_search(best_set, remaining - 5)
        if polish_score > best_score:
            best_set = polish_u
            best_score = polish_score
    
    return best_set
