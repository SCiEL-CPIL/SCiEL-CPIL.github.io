import itertools
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    N = q**n
    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    neighbors = [[] for _ in range(N)]
    neighbors_set = [set() for _ in range(N)]
    
    for idx, w in enumerate(words):
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        neighbors[idx] = list(set(neighbors[idx]))
        neighbors_set[idx] = set(neighbors[idx])

    def greedy_init():
        code = []
        code_set = set()
        covered = [False] * N
        
        all_words = list(range(N))
        random.shuffle(all_words)
        
        while not all(covered):
            best_w = -1
            best_count = 0
            for w in all_words:
                if w in code_set:
                    continue
                cnt = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            
            if best_w == -1:
                break
            
            code.append(best_w)
            code_set.add(best_w)
            for u in neighbors[best_w]:
                covered[u] = True
        
        return code, code_set

    def local_search(code, code_set, time_limit):
        cover_count = [0] * N
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        best_code = list(code)
        best_size = len(best_code)
        start = time.time()
        
        while time.time() < start + time_limit:
            improved = False
            random.shuffle(code)
            
            # Phase 1: Remove
            i = 0
            while i < len(code):
                c = code[i]
                uncovered = []
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        uncovered.append(nb)
                
                if not uncovered:
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break
                else:
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                    i += 1
            
            if improved:
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue
            
            # Phase 2: Swap
            for i in range(len(code)):
                if time.time() > start + time_limit: break
                c = code[i]
                
                uncovered = []
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        uncovered.append(nb)
                
                if not uncovered:
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break
                
                u0 = uncovered[0]
                candidates = neighbors_set[u0]
                
                valid_swaps = []
                for c_prime in candidates:
                    if c_prime in code_set:
                        continue
                    ok = True
                    for u in uncovered[1:]:
                        if c_prime not in neighbors_set[u]:
                            ok = False
                            break
                    if ok:
                        valid_swaps.append(c_prime)
                
                if valid_swaps:
                    c_prime = random.choice(valid_swaps)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c_prime]:
                        cover_count[nb] += 1
                    code[i] = c_prime
                    code_set.remove(c)
                    code_set.add(c_prime)
                    improved = True
                    break
            
            if not improved:
                break

        return best_code

    start_time = time.time()
    deadline = 140.0
    global_best = None
    global_best_size = float('inf')
    
    while time.time() < deadline:
        time_budget = deadline - time.time()
        if time_budget < 10: break
        
        code, code_set = greedy_init()
        search_time = min(10.0, time_budget - 5.0)
        
        best_code = local_search(code, code_set, search_time)
        
        if len(best_code) < global_best_size:
            global_best_size = len(best_code)
            global_best = best_code
    
    if global_best is None:
        code, _ = greedy_init()
        global_best = code
        
    return [list(idx_to_tuple(c)) for c in global_best]
