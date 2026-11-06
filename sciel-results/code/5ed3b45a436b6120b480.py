import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    required = [1, 2, 5, 11]
    required_set = set(required)
    best_set = set(required)
    best_size = len(best_set)
    
    def get_sums(S):
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def is_valid_addition(x, S, sums):
        for y in S:
            if (x + y) in sums:
                return False
        if (2 * x) in sums:
            return False
        return True

    def add_to_sums(x, S, sums):
        for y in S:
            sums.add(x + y)
        sums.add(2 * x)

    def greedy_fill(S, sums):
        """Greedily add as many elements as possible."""
        while True:
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            if not candidates:
                break
            x = random.choice(candidates)
            S.add(x)
            add_to_sums(x, S, sums)
        return S, sums

    def local_search_1swap(S, sums, time_limit):
        """Local search with 1-swap moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            if candidates:
                x = random.choice(candidates)
                S.add(x)
                add_to_sums(x, S, sums)
                improved = True
                continue
            # Try 1-swap
            non_fixed = [a for a in S if a not in required_set]
            if not non_fixed:
                break
            a = random.choice(non_fixed)
            b = random.randint(1, 284)
            if b in S:
                continue
            S_temp = S - {a}
            sums_temp = get_sums(S_temp)
            if is_valid_addition(b, S_temp, sums_temp):
                S.remove(a)
                S.add(b)
                sums = get_sums(S)
                improved = True
        return S, sums

    def local_search_2swap(S, sums, time_limit):
        """Local search with 2-swap moves (remove 2, add 2)."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            if candidates:
                x = random.choice(candidates)
                S.add(x)
                add_to_sums(x, S, sums)
                improved = True
                continue
            # Try 2-swap: remove 2 non-fixed elements, add 2 new elements
            non_fixed = [a for a in S if a not in required_set]
            if len(non_fixed) < 2:
                break
            a1, a2 = random.sample(non_fixed, 2)
            b1 = random.randint(1, 284)
            b2 = random.randint(1, 284)
            if b1 in S or b2 in S or b1 == b2:
                continue
            S_temp = S - {a1, a2}
            sums_temp = get_sums(S_temp)
            # Check if b1 can be added
            if not is_valid_addition(b1, S_temp, sums_temp):
                continue
            S_temp.add(b1)
            add_to_sums(b1, S_temp, sums_temp)
            # Check if b2 can be added
            if not is_valid_addition(b2, S_temp, sums_temp):
                continue
            S_temp.add(b2)
            add_to_sums(b2, S_temp, sums_temp)
            # Perform the swap
            S = S_temp
            sums = sums_temp
            improved = True
        return S, sums

    def remove_and_refill(S, sums, time_limit):
        """Remove a non-fixed element and greedily refill."""
        non_fixed = [a for a in S if a not in required_set]
        if not non_fixed:
            return S, sums
        a = random.choice(non_fixed)
        S_temp = S - {a}
        sums_temp = get_sums(S_temp)
        S_temp, sums_temp = greedy_fill(S_temp, sums_temp)
        if len(S_temp) > len(S):
            return S_temp, sums_temp
        return S, sums

    def random_greedy_with_swaps():
        S = set(required)
        sums = get_sums(S)
        
        # Phase 1: Greedy Add
        S, sums = greedy_fill(S, sums)
        
        # Phase 2: Local search with 1-swaps
        S, sums = local_search_1swap(S, sums, deadline)
        
        # Phase 3: Local search with 2-swaps
        S, sums = local_search_2swap(S, sums, deadline)
        
        # Phase 4: Remove and refill
        while time.time() < deadline:
            S_new, sums_new = remove_and_refill(S, sums, deadline)
            if len(S_new) > len(S):
                S, sums = S_new, sums_new
                # Continue local search after improvement
                S, sums = local_search_1swap(S, sums, deadline)
                S, sums = local_search_2swap(S, sums, deadline)
            else:
                break
        
        return S

    # Main Loop: Multiple Restarts
    while time.time() < deadline:
        candidate_set = random_greedy_with_swaps()
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
            if best_size == 20:
                break

    return list(best_set)
