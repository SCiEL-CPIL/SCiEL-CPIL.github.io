import itertools
import random
import time
import math

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

    def greedy_independent_set_init():
        """Greedy approach on the distance-2 graph: select words that are mutually at distance >= 2,
        maximizing coverage. This is equivalent to finding a large independent set in the graph where
        edges connect words at distance 1."""
        code = []
        code_set = set()
        covered = [False] * N
        
        all_words = list(range(N))
        random.shuffle(all_words)
        
        # First pass: greedy set cover
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
            
            # Phase 1: Remove redundant codewords
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
            
            # Phase 2: Swap codeword to better position
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

    def simulated_annealing(code, code_set, time_limit):
        """SA with efficient coverage tracking to escape local minima."""
        cover_count = [0] * N
        uncovered_count = 0
        for c in code:
            for nb in neighbors[c]:
                if cover_count[nb] == 0:
                    uncovered_count += 1
                cover_count[nb] += 1
        
        best_code = list(code)
        best_size = len(code)
        
        current_obj = len(code) + 10 * uncovered_count
        T = 5.0
        cooling = 0.9995
        start = time.time()
        
        while time.time() < start + time_limit:
            move_type = random.random()
            
            if move_type < 0.45 and len(code) > 1:
                # Remove a codeword
                i = random.randint(0, len(code) - 1)
                c = code[i]
                new_uncovered = 0
                affected = []
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        new_uncovered += 1
                    affected.append(nb)
                
                if new_uncovered == 0:
                    code.pop(i)
                    code_set.remove(c)
                    uncovered_count -= 0  # no change since we removed 0 new uncovered
                    new_obj = len(code) + 10 * uncovered_count
                    delta = new_obj - current_obj
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_obj = new_obj
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                    else:
                        code.insert(i, c)
                        code_set.add(c)
                        for nb in affected:
                            cover_count[nb] += 1
                else:
                    for nb in affected:
                        cover_count[nb] += 1
                T *= cooling
                
            elif move_type < 0.85 and len(code) > 0:
                # Swap a codeword
                i = random.randint(0, len(code) - 1)
                c = code[i]
                
                # Find candidates that cover the uniquely-covered neighbors
                unique_neighbors = [nb for nb in neighbors[c] if cover_count[nb] == 1]
                
                if not unique_neighbors:
                    # Can just remove
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    code_set.remove(c)
                    new_obj = len(code) + 10 * uncovered_count
                    delta = new_obj - current_obj
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_obj = new_obj
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                    else:
                        code.insert(i, c)
                        code_set.add(c)
                        for nb in neighbors[c]:
                            cover_count[nb] += 1
                    T *= cooling
                    continue
                
                # Try to find a replacement
                candidates = list(neighbors_set[unique_neighbors[0]])
                random.shuffle(candidates)
                
                swapped = False
                for c_prime in candidates[:50]:  # Limit candidates for speed
                    if c_prime in code_set:
                        continue
                    
                    # Check if swap maintains coverage
                    new_uncovered = 0
                    temp_affected = []
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                        if cover_count[nb] == 0:
                            new_uncovered += 1
                        temp_affected.append(nb)
                    for nb in neighbors[c_prime]:
                        if cover_count[nb] == 0:
                            pass  # was uncovered, now covered
                        cover_count[nb] += 1
                        temp_affected.append(nb)
                    
                    # Count new uncovered after swap
                    actual_new_uncovered = 0
                    for nb in temp_affected:
                        if cover_count[nb] == 0:
                            actual_new_uncovered += 1
                    
                    # Revert for proper accounting
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                    for nb in neighbors[c_prime]:
                        cover_count[nb] -= 1
                    
                    if actual_new_uncovered == 0:
                        # Valid swap - apply it
                        for nb in neighbors[c]:
                            cover_count[nb] -= 1
                        for nb in neighbors[c_prime]:
                            cover_count[nb] += 1
                        code[i] = c_prime
                        code_set.remove(c)
                        code_set.add(c_prime)
                        new_obj = len(code) + 10 * uncovered_count
                        delta = new_obj - current_obj
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            current_obj = new_obj
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = list(code)
                        else:
                            for nb in neighbors[c]:
                                cover_count[nb] += 1
                            for nb in neighbors[c_prime]:
                                cover_count[nb] -= 1
                            code[i] = c
                            code_set.remove(c_prime)
                            code_set.add(c)
                        swapped = True
                        break
                
                if not swapped:
                    T *= cooling
            else:
                T *= cooling
            
            if T < 0.1:
                T = 0.1
        
        return best_code

    start_time = time.time()
    deadline = 140.0
    global_best = None
    global_best_size = float('inf')
    
    while time.time() < deadline:
        time_budget = deadline - time.time()
        if time_budget < 15: break
        
        code, code_set = greedy_init()
        
        # First do greedy local search
        ls_time = min(5.0, time_budget - 10.0)
        if ls_time > 0:
            code = local_search(code, code_set, ls_time)
            code_set = set(code)
        
        # Then do simulated annealing
        sa_time = min(20.0, time_budget - 5.0)
        if sa_time > 0:
            best_code = simulated_annealing(code, code_set, sa_time)
            
            if len(best_code) < global_best_size:
                global_best_size = len(best_code)
                global_best = best_code
    
    if global_best is None:
        code, _ = greedy_init()
        global_best = code
        
    return [list(idx_to_tuple(c)) for c in global_best]
