def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80
    
    n = 256
    
    def is_valid(S_set, S_list):
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        return False
        return True
    
    def count_violations(S_set, S_list):
        m = len(S_list)
        count = 0
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        count += 1
        return count
    
    # Start with (128, 256]
    S_list = list(range(129, 257))
    S_set = set(S_list)
    
    # Remove elements to fix violations
    while True:
        if time.time() > deadline:
            break
        if is_valid(S_set, S_list):
            break
        # Find the element that's in the most violations
        # For each element, count how many violations it's part of
        from collections import defaultdict
        viol_count = defaultdict(int)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        viol_count[a] += 1
                        viol_count[b] += 1
                        viol_count[c] += 1
        # Remove the element with the most violations
        worst = max(viol_count, key=viol_count.get)
        S_list.remove(worst)
        S_set.discard(worst)
    
    # Now try to add back removed elements and add new ones
    removed = set(range(129, 257)) - S_set
    all_candidates = list(removed) + list(range(1, 129))
    all_candidates.sort(reverse=True)
    
    for a in all_candidates:
        if time.time() > deadline:
            break
        if a in S_set:
            continue
        S_list_test = S_list + [a]
        S_list_test.sort()
        S_set_test = S_set | {a}
        if is_valid(S_set_test, S_list_test):
            S_list = S_list_test
            S_set = S_set_test
    
    return S_list
