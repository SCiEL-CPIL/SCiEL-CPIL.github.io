import time

def construct():
    deadline = time.time() + 85.0
    
    def has_sum_200(bits):
        return (bits >> 200) & 1
    
    def add_element(bits, x):
        return bits | (bits << x)
    
    def is_valid(bits):
        return not has_sum_200(bits)
    
    # Start with known valid set {66, ..., 100}
    base = list(range(66, 101))
    bits = 1  # bit 0 set
    for x in base:
        bits = add_element(bits, x)
    
    assert is_valid(bits), "Base set should be valid"
    
    best_set = set(base)
    best_size = len(base)
    
    # Greedy: try to add elements in different orders
    candidates = [x for x in range(1, 66)]
    
    # Multiple greedy orderings
    orderings = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        # Random-ish deterministic orders
        list(reversed(sorted(candidates, key=lambda x: x % 7))),
        sorted(candidates, key=lambda x: (x % 5, x)),
        sorted(candidates, key=lambda x: -(x % 11)),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        cur_bits = bits
        cur_set = set(base)
        for x in order:
            if time.time() > deadline:
                break
            new_bits = add_element(cur_bits, x)
            if is_valid(new_bits):
                cur_bits = new_bits
                cur_set.add(x)
        if len(cur_set) > best_size:
            best_size = len(cur_set)
            best_set = cur_set
    
    # Also try starting from scratch with greedy
    for start_order in [sorted(range(1,101), reverse=True), sorted(range(1,101))]:
        if time.time() > deadline:
            break
        cur_bits = 1
        cur_set = set()
        for x in start_order:
            if time.time() > deadline:
                break
            new_bits = add_element(cur_bits, x)
            if is_valid(new_bits):
                cur_bits = new_bits
                cur_set.add(x)
        if len(cur_set) > best_size:
            best_size = len(cur_set)
            best_set = cur_set
    
    # Local search: try swaps to improve
    not_in_best = [x for x in range(1, 101) if x not in best_set]
    best_list = list(best_set)
    n_best = len(best_list)
    
    # 1-for-1 swaps
    for i in range(n_best):
        if time.time() > deadline:
            break
        for j in range(len(not_in_best)):
            if time.time() > deadline:
                break
            new_set = set(best_list)
            new_set.discard(best_list[i])
            new_set.add(not_in_best[j])
            if len(new_set) < best_size:
                continue
            nb = 1
            for x in sorted(new_set):
                nb = add_element(nb, x)
            if is_valid(nb):
                # Try to add more
                cur_bits = nb
                cur_set = new_set
                for y in not_in_best:
                    if y in cur_set:
                        continue
                    if time.time() > deadline:
                        break
                    new_b = add_element(cur_bits, y)
                    if is_valid(new_b):
                        cur_bits = new_b
                        cur_set.add(y)
                if len(cur_set) > best_size:
                    best_size = len(cur_set)
                    best_set = cur_set
                    not_in_best = [x for x in range(1, 101) if x not in best_set]
    
    # 2-for-1 swaps (remove 2, add 3) - limited
    if best_size < 95 and time.time() < deadline - 10:
        not_in_best = [x for x in range(1, 101) if x not in best_set]
        best_list = list(best_set)
        n_best = len(best_list)
        max_i = min(n_best, 15)
        max_k = min(len(not_in_best), 8)
        for i in range(max_i):
            if time.time() > deadline:
                break
            for j in range(i+1, max_i):
                if time.time() > deadline:
                    break
                for k in range(max_k):
                    if time.time() > deadline:
                        break
                    for l in range(k+1, max_k):
                        if time.time() > deadline:
                            break
                        for m in range(l+1, max_k):
                            if time.time() > deadline:
                                break
                            new_set = set(best_list)
                            new_set.discard(best_list[i])
                            new_set.discard(best_list[j])
                            new_set.add(not_in_best[k])
                            new_set.add(not_in_best[l])
                            new_set.add(not_in_best[m])
                            if len(new_set) <= best_size:
                                continue
                            nb = 1
                            for x in sorted(new_set):
                                nb = add_element(nb, x)
                            if is_valid(nb):
                                cur_bits = nb
                                cur_set = new_set
                                for y in not_in_best:
                                    if y in cur_set:
                                        continue
                                    if time.time() > deadline:
                                        break
                                    new_b = add_element(cur_bits, y)
                                    if is_valid(new_b):
                                        cur_bits = new_b
                                        cur_set.add(y)
                                if len(cur_set) > best_size:
                                    best_size = len(cur_set)
                                    best_set = cur_set
                                    not_in_best = [x for x in range(1, 101) if x not in best_set]
    
    return list(best_set)
