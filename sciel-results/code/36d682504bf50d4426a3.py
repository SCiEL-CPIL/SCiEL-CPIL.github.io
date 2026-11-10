import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit
    N = 1000
    
    def greedy_init(order):
        S = set()
        S_list = []
        counts = [0] * (2 * N + 1)
        
        for x in order:
            valid = True
            if counts[2 * x] >= 2:
                valid = False
            
            if valid:
                for y in S_list:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                S.add(x)
                S_list.append(x)
                for y in S_list[:-1]:
                    counts[x + y] += 1
                counts[2 * x] += 1
                
        return S, S_list, counts

    def try_insert(S, S_list, counts, x):
        if x in S:
            return True
        if counts[2 * x] >= 2:
            return False
        for y in S_list:
            if counts[x + y] >= 2:
                return False
        
        S.add(x)
        S_list.append(x)
        for y in S_list[:-1]:
            counts[x + y] += 1
        counts[2 * x] += 1
        return True

    def remove_from(S, S_list, counts, x):
        if x not in S:
            return False
        S.remove(x)
        S_list.remove(x)
        for y in S_list:
            counts[x + y] -= 1
        counts[2 * x] -= 1
        return True

    def local_search(S, S_list, counts, max_iters=50000):
        best_S = set(S)
        best_S_list = list(S_list)
        best_counts = counts[:]
        best_size = len(S)
        
        current_S = set(S)
        current_S_list = list(S_list)
        current_counts = counts[:]
        
        all_elements = list(range(1, N + 1))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            if random.random() < 0.4:
                candidates = [x for x in all_elements if x not in current_S]
                if candidates:
                    x = random.choice(candidates)
                    if try_insert(current_S, current_S_list, current_counts, x):
                        if len(current_S) > best_size:
                            best_S = set(current_S)
                            best_S_list = list(current_S_list)
                            best_counts = current_counts[:]
                            best_size = len(current_S)
            else:
                if not current_S:
                    continue
                x = random.choice(current_S_list)
                remove_from(current_S, current_S_list, current_counts, x)
                
                candidates = [x2 for x2 in all_elements if x2 not in current_S]
                if candidates:
                    random.shuffle(candidates)
                    found = False
                    for x2 in candidates[:10]:
                        if try_insert(current_S, current_S_list, current_counts, x2):
                            found = True
                            if len(current_S) > best_size:
                                best_S = set(current_S)
                                best_S_list = list(current_S_list)
                                best_counts = current_counts[:]
                                best_size = len(current_S)
                            break
                    if not found:
                        try_insert(current_S, current_S_list, current_counts, x)
                        
        return best_S, best_S_list, best_counts

    random.seed(42)
    best_S_overall = set()
    best_size_overall = 0
    
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: random.random()),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        S, S_list, counts = greedy_init(ordering)
        best_S, best_S_list, best_counts = local_search(S, S_list, counts, max_iters=30000)
        
        if len(best_S) > best_size_overall:
            best_size_overall = len(best_S)
            best_S_overall = best_S

    if time.time() < deadline - 5:
        final_S = set(best_S_overall)
        final_S_list = list(best_S_overall)
        final_counts = [0] * (2 * N + 1)
        for x in final_S_list:
            for y in final_S_list:
                if y < x:
                    continue
                final_counts[x + y] += 1
        
        all_elements = list(range(1, N + 1))
        for x in all_elements:
            if x in final_S:
                continue
            if try_insert(final_S, final_S_list, final_counts, x):
                if len(final_S) > best_size_overall:
                    best_size_overall = len(final_S)
                    best_S_overall = final_S

    return list(best_S_overall)
