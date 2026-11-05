"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys

N_MAX = 4096
# Max possible triple sum is 3*4096 = 12288
MAX_SUM = 3 * N_MAX

def _triple_sums_set(S):
    """Return the set of all unordered triple sums (with repetition) for sorted list S."""
    n = len(S)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _new_sums_with_x(x, S):
    """Compute all triple sums that involve x, given existing set S."""
    n = len(S)
    sums = set()
    sums.add(3 * x)
    for i in range(n):
        sums.add(2 * x + S[i])
        sums.add(x + 2 * S[i])
    for i in range(n):
        for j in range(i + 1, n):
            sums.add(x + S[i] + S[j])
    return sums

def _check_candidate_fast(x, S, existing_sums):
    """Fast check if x can be added without collision."""
    n = len(S)
    # Check (x,x,x)
    if 3*x in existing_sums:
        return False
    # Check (x,x,S[i])
    for i in range(n):
        if (2*x + S[i]) in existing_sums:
            return False
    # Check (x,S[i],S[i])
    for i in range(n):
        if (x + 2*S[i]) in existing_sums:
            return False
    # Check (x,S[i],S[j]) for i<j
    for i in range(n):
        for j in range(i+1, n):
            if (x + S[i] + S[j]) in existing_sums:
                return False
    return True

def _greedy_build(order, n_max=N_MAX, time_limit=10.0):
    """Greedy build following a given order of candidates."""
    S = []
    existing_sums = set()
    deadline = time.time() + time_limit
    for x in order:
        if time.time() > deadline:
            break
        if _check_candidate_fast(x, S, existing_sums):
            S.append(x)
            # Add new sums
            new_sums = _new_sums_with_x(x, S[:-1])
            existing_sums |= new_sums
    S.sort()
    return S

def _local_search_add(S, existing_sums, n_max=N_MAX, time_limit=20.0):
    """Try to add elements by scanning candidates and checking collisions."""
    S = list(S)
    existing_sums = set(existing_sums)
    deadline = time.time() + time_limit
    S_set = set(S)
    candidates = [x for x in range(1, n_max + 1) if x not in S_set]
    random.shuffle(candidates)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if _check_candidate_fast(x, S, existing_sums):
                S.append(x)
                S_set.add(x)
                new_sums = _new_sums_with_x(x, S[:-1])
                existing_sums |= new_sums
                improved = True
    S.sort()
    return S, existing_sums

def _local_search_swap(S, existing_sums, n_max=N_MAX, time_limit=25.0):
    """Try to swap out elements to make room for more, or add new ones."""
    S = list(S)
    existing_sums = set(existing_sums)
    deadline = time.time() + time_limit
    S_set = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        candidates = [x for x in range(1, n_max + 1) if x not in S_set]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            if _check_candidate_fast(x, S, existing_sums):
                S.append(x)
                S_set.add(x)
                new_sums = _new_sums_with_x(x, S[:-1])
                existing_sums |= new_sums
                improved = True
                break
        if not improved:
            # Try removing one element and then adding a new one
            for idx in range(len(S)):
                if time.time() > deadline:
                    break
                removed = S[idx]
                S_tmp = S[:idx] + S[idx+1:]
                S_tmp_set = set(S_tmp)
                sums_tmp = _triple_sums_set(S_tmp)
                candidates = [x for x in range(1, n_max + 1) if x not in S_tmp_set]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _check_candidate_fast(x, S_tmp, sums_tmp):
                        S_tmp.append(x)
                        S = S_tmp
                        S_set = S_tmp_set | {x}
                        new_sums = _new_sums_with_x(x, S_tmp[:-1])
                        existing_sums = sums_tmp | new_sums
                        improved = True
                        break
                if improved:
                    break
    S.sort()
    return S, existing_sums

def construct():
    start_time = time.time()
    total_time_limit = 82.0  # safety margin before 90s kill
    
    best_S = []
    best_len = 0
    
    # Phase 1: Multiple greedy builds with different orderings
    greedy_deadline = start_time + 12.0
    attempts = 0
    while time.time() < greedy_deadline:
        attempts += 1
        if attempts % 3 == 1:
            order = list(range(1, N_MAX + 1))
        elif attempts % 3 == 2:
            order = list(range(1, N_MAX + 1))
            random.shuffle(order)
        else:
            order = []
            val = 1
            while val <= N_MAX:
                order.append(val)
                val = int(val * 1.6) + 1
            remaining = [x for x in range(1, N_MAX + 1) if x not in set(order)]
            random.shuffle(remaining)
            order.extend(remaining)
        
        S = _greedy_build(order, n_max=N_MAX, time_limit=2.5)
        if len(S) > best_len:
            best_S = S
            best_len = len(S)
    
    # Phase 2: Local search - add more elements
    if time.time() < start_time + total_time_limit:
        existing_sums = _triple_sums_set(best_S)
        remaining_time = start_time + total_time_limit - time.time()
        add_time = min(remaining_time * 0.35, 18.0)
        best_S, existing_sums = _local_search_add(best_S, existing_sums, n_max=N_MAX, time_limit=add_time)
        if len(best_S) > best_len:
            best_len = len(best_S)
    
    # Phase 3: Local search - swaps
    if time.time() < start_time + total_time_limit:
        existing_sums = _triple_sums_set(best_S)
        remaining_time = start_time + total_time_limit - time.time()
        swap_time = min(remaining_time * 0.55, 35.0)
        best_S, existing_sums = _local_search_swap(best_S, existing_sums, n_max=N_MAX, time_limit=swap_time)
        if len(best_S) > best_len:
            best_len = len(best_S)
    
    # Phase 4: Final add attempt
    if time.time() < start_time + total_time_limit - 2:
        existing_sums = _triple_sums_set(best_S)
        remaining_time = start_time + total_time_limit - time.time() - 1
        if remaining_time > 2:
            best_S, existing_sums = _local_search_add(best_S, existing_sums, n_max=N_MAX, time_limit=remaining_time)
            if len(best_S) > best_len:
                best_len = len(best_S)
    
    # Ensure all elements are in [1, 4096] and distinct
    best_S = sorted(set(x for x in best_S if 1 <= x <= N_MAX))
    return best_S

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements")
    print(result)
