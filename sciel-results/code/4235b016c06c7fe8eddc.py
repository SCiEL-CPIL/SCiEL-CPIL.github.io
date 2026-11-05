import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflict graph using bitsets for speed
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = 0
            a, b = i, j
            while b:
                a, b = b, a % b
            g = a
            u_prime = i // g
            v_prime = j // g
            if g % (u_prime + v_prime) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Bitset for all vertices
    all_mask = 0
    for v in vertices:
        all_mask |= (1 << v)
    
    def can_add(v, s_mask):
        """Check if vertex v can be added to set represented by s_mask."""
        return (conflict_mask[v] & s_mask) == 0
    
    def greedy_by_order(order, s_mask=0):
        result_mask = s_mask
        for v in order:
            if (result_mask >> v) & 1:
                continue
            if can_add(v, result_mask):
                result_mask |= (1 << v)
        return result_mask
    
    def mask_to_set(mask):
        return {v for v in vertices if (mask >> v) & 1}
    
    def popcount(mask):
        return bin(mask).count('1')
    
    # Greedy construction
    best_mask = 0
    
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    order_odd_first = sorted(vertices, key=lambda x: (x % 2 == 0, degrees[x]))
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))
    # New: sort by degree/value ratio
    order_ratio = sorted(vertices, key=lambda x: degrees[x] / x)
    order_ratio_desc = sorted(vertices, key=lambda x: -degrees[x] / x)
    
    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_odd),
        greedy_by_order(order_even),
        greedy_by_order(order_odd_first),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
        greedy_by_order(order_ratio),
        greedy_by_order(order_ratio_desc),
    ]
    
    # More random shuffles for diverse starts
    for _ in range(30):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))
    
    for c in candidates:
        if popcount(c) > popcount(best_mask):
            best_mask = c
    
    # Local search: add vertices, then try 1-for-2 and 2-for-3 swaps
    def local_search(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try adding each excluded vertex
            excluded = [v for v in vertices if not (s_mask >> v) & 1]
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s_mask):
                    s_mask |= (1 << v)
                    improved = True
            
            # Phase 2: Try 1-for-2 swaps (remove one, add two non-conflicting)
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u)
                    addable = []
                    for v in vertices:
                        if v != u and not (s_mask >> v) & 1 and can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 25)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if not ((conflict_mask[v1] >> v2) & 1):
                                    s_mask = temp_mask | (1 << v1) | (1 << v2)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
            
            # Phase 3: Try 2-for-3 swaps (remove two, add three non-conflicting)
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                limit_pairs = min(len(s_list), 30)
                for i in range(limit_pairs):
                    if time.time() >= time_limit:
                        break
                    u1 = s_list[i]
                    for j in range(i + 1, limit_pairs):
                        if time.time() >= time_limit:
                            break
                        u2 = s_list[j]
                        temp_mask = s_mask & ~(1 << u1) & ~(1 << u2)
                        addable = []
                        for v in vertices:
                            if v != u1 and v != u2 and not (s_mask >> v) & 1 and can_add(v, temp_mask):
                                addable.append(v)
                        if len(addable) >= 3:
                            random.shuffle(addable)
                            found = False
                            alimit = min(len(addable), 15)
                            for a in range(alimit):
                                for b in range(a + 1, alimit):
                                    for c in range(b + 1, alimit):
                                        v1, v2, v3 = addable[a], addable[b], addable[c]
                                        if (not ((conflict_mask[v1] >> v2) & 1) and 
                                            not ((conflict_mask[v1] >> v3) & 1) and 
                                            not ((conflict_mask[v2] >> v3) & 1)):
                                            s_mask = temp_mask | (1 << v1) | (1 << v2) | (1 << v3)
                                            improved = True
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
        return s_mask
    
    # Run local search from best candidate
    time_limit = deadline - 15.0
    best_mask = local_search(best_mask, time_limit)
    
    # Simulated annealing phase
    def simulated_annealing(s_mask, time_limit):
        current_mask = s_mask
        current_size = popcount(current_mask)
        best_mask_sa = current_mask
        best_size_sa = current_size
        
        t0 = 100.0
        t_end = 0.05
        cooling = 0.998
        temp = t0
        no_improve = 0
        
        while time.time() < time_limit and temp > t_end:
            move_type = random.random()
            
            if move_type < 0.35:
                # Try adding a random excluded vertex
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not excluded:
                    temp *= cooling
                    continue
                v = random.choice(excluded)
                if can_add(v, current_mask):
                    new_mask = current_mask | (1 << v)
                    new_size = current_size + 1
                    delta = 1
                else:
                    conflicts = [u for u in vertices if (current_mask >> u) & 1 and ((conflict_mask[v] >> u) & 1)]
                    if not conflicts:
                        temp *= cooling
                        continue
                    u = random.choice(conflicts)
                    temp_mask = current_mask & ~(1 << u)
                    if can_add(v, temp_mask):
                        new_mask = temp_mask | (1 << v)
                        new_size = current_size
                        delta = 0
                    else:
                        temp *= cooling
                        continue
            elif move_type < 0.55:
                # Try removing a random included vertex
                included = [v for v in vertices if (current_mask >> v) & 1]
                if not included:
                    temp *= cooling
                    continue
                v = random.choice(included)
                new_mask = current_mask & ~(1 << v)
                new_size = current_size - 1
                delta = -1
            elif move_type < 0.80:
                # Try a swap: remove one, add one
                included = [v for v in vertices if (current_mask >> v) & 1]
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not included or not excluded:
                    temp *= cooling
                    continue
                u = random.choice(included)
                v = random.choice(excluded)
                temp_mask = current_mask & ~(1 << u)
                if can_add(v, temp_mask):
                    new_mask = temp_mask | (1 << v)
                    new_size = current_size
                    delta = 0
                else:
                    temp *= cooling
                    continue
            else:
                # Try 1-for-2 swap: remove one, add two
                included = [v for v in vertices if (current_mask >> v) & 1]
                excluded = [v for v in vertices if not (current_mask >> v) & 1]
                if not included or len(excluded) < 2:
                    temp *= cooling
                    continue
                u = random.choice(included)
                temp_mask = current_mask & ~(1 << u)
                addable = [v for v in excluded if can_add(v, temp_mask)]
                if len(addable) >= 2:
                    random.shuffle(addable)
                    found = False
                    limit = min(len(addable), 10)
                    for i in range(limit):
                        for j in range(i + 1, limit):
                            v1, v2 = addable[i], addable[j]
                            if not ((conflict_mask[v1] >> v2) & 1):
                                new_mask = temp_mask | (1 << v1) | (1 << v2)
                                new_size = current_size + 1
                                delta = 1
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        temp *= cooling
                        continue
                else:
                    temp *= cooling
                    continue
            
            # Accept/reject
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size_sa:
                    best_mask_sa = current_mask
                    best_size_sa = current_size
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if random.random() < math.exp(delta / temp):
                    current_mask = new_mask
                    current_size = new_size
                no_improve += 1
            
            temp *= cooling
            
            if no_improve > 300:
                break
        
        return best_mask_sa
    
    # Run simulated annealing
    sa_time_limit = deadline - 8.0
    best_mask = simulated_annealing(best_mask, sa_time_limit)
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        best_mask = local_search(best_mask, deadline - 1.0)
    
    # Final validation
    result_list = sorted(mask_to_set(best_mask))
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2))
    
    return result_list
# EVOLVE-BLOCK-END
