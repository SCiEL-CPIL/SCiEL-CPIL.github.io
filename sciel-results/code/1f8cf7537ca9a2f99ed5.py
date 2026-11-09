import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    # Build conflict graph: conflict[i][j] = True if (i*j) % (i+j) == 0
    # For efficiency, for each a, find all b where (a+b) | a*b
    
    # conflict[a] = set of b values that conflict with a
    conflict = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find all divisors d of a_sq where d > a
        # b = d - a, need 1 <= b <= N, b != a
        # d = a + b, so d > a means b > 0
        # d | a_sq
        for d in range(a + 1, a_sq + 1):
            if a_sq % d == 0:
                b = d - a
                if 1 <= b <= N and b != a:
                    conflict[a].add(b)
                    conflict[b].add(a)
    
    # Now find maximum independent set using greedy + local search
    # Greedy: sort by degree (ascending), pick vertices with no conflicts to already-picked
    
    def greedy_independent_set(order):
        chosen = set()
        for v in order:
            can_add = True
            for u in chosen:
                if v in conflict[u]:
                    can_add = False
                    break
            if can_add:
                chosen.add(v)
        return chosen
    
    # Try multiple orderings
    all_nodes = list(range(1, N + 1))
    
    # Initial: sort by degree ascending
    degree = [len(conflict[i]) for i in range(N + 1)]
    order_by_degree = sorted(all_nodes, key=lambda x: degree[x])
    
    best_set = greedy_independent_set(order_by_degree)
    best_size = len(best_set)
    
    # Also try random orderings
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        random_order = all_nodes[:]
        random.shuffle(random_order)
        s = greedy_independent_set(random_order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to swap elements to improve
    # For each element in best_set, try removing it and adding a non-conflicting element
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_list = list(best_set)
        for i, v in enumerate(best_list):
            # Try removing v
            temp_set = set(best_list)
            temp_set.remove(v)
            # Try adding each candidate not in temp_set
            for cand in all_nodes:
                if cand in temp_set:
                    continue
                # Check if cand conflicts with any in temp_set
                can_add = True
                for u in temp_set:
                    if cand in conflict[u]:
                        can_add = False
                        break
                if can_add:
                    new_set = temp_set | {cand}
                    # This is same size, but might allow future improvements
                    # Actually for size improvement, we need to add without removing
                    # Let's try pure addition first
                    pass
            
            # Try pure addition: can we add any node to best_set?
            for cand in all_nodes:
                if cand in best_set:
                    continue
                can_add = True
                for u in best_set:
                    if cand in conflict[u]:
                        can_add = False
                        break
                if can_add:
                    best_set.add(cand)
                    best_size += 1
                    improved = True
                    break
            if improved:
                break
    
    # More aggressive local search: 2-opt style swaps
    if time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            # Pick a random element from best_set and a random element from complement
            best_list = list(best_set)
            comp = [x for x in all_nodes if x not in best_set]
            if not comp:
                break
            v = random.choice(best_list)
            cand = random.choice(comp)
            
            # Check if we can swap: remove v, add cand
            temp_set = set(best_list)
            temp_set.remove(v)
            can_add = True
            for u in temp_set:
                if cand in conflict[u]:
                    can_add = False
                    break
            if can_add:
                # This swap maintains size but may enable future additions
                # Check if after swap, we can add more
                new_set = temp_set | {cand}
                # Try to add more
                added = False
                for extra in all_nodes:
                    if extra in new_set:
                        continue
                    can_add_extra = True
                    for u in new_set:
                        if extra in conflict[u]:
                            can_add_extra = False
                            break
                    if can_add_extra:
                        new_set.add(extra)
                        added = True
                        break
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
    
    # Final greedy pass to see if we can add anything
    for cand in all_nodes:
        if cand in best_set:
            continue
        can_add = True
        for u in best_set:
            if cand in conflict[u]:
                can_add = False
                break
        if can_add:
            best_set.add(cand)
            best_size += 1
    
    return list(best_set)
