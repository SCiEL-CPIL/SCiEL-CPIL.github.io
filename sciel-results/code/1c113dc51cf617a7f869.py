import time
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if time.time() > deadline:
                    break
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return numbers
    
    T = len(bad_triples)
    if T == 0:
        return numbers
    
    # Build conflict graph: two elements are in conflict if they appear together in a bad triple
    conflict = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)
    
    # Greedy vertex cover to get initial solution
    counts = [0] * (n + 1)
    for (a, b, c) in bad_triples:
        counts[a] += 1
        counts[b] += 1
        counts[c] += 1
    
    active = [True] * T
    active_count = T
    removed = [False] * (n + 1)
    
    while active_count > 0:
        if time.time() > deadline:
            break
        
        best_x = -1
        best_cnt = -1
        for x in range(1, n + 1):
            if not removed[x] and counts[x] > best_cnt:
                best_cnt = counts[x]
                best_x = x
        
        if best_x == -1 or best_cnt <= 0:
            break
        
        removed[best_x] = True
        
        for i in range(T):
            if not active[i]:
                continue
            a, b, c = bad_triples[i]
            if a == best_x or b == best_x or c == best_x:
                active[i] = False
                active_count -= 1
                for x in (a, b, c):
                    if x != best_x:
                        counts[x] -= 1
    
    result_set = set(x for x in range(1, n + 1) if not removed[x])
    
    def is_valid_set(s):
        s_list = sorted(s)
        len_s = len(s_list)
        for i in range(len_s):
            for j in range(i + 1, len_s):
                a = s_list[i]
                b = s_list[j]
                lab = lcm_table[a][b]
                for k in range(j + 1, len_s):
                    c = s_list[k]
                    if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                        return False
        return True
    
    # Try adding back each removed element
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if removed[x]:
            can_add = True
            for y in result_set:
                lxy = lcm_table[x][y]
                for z in result_set:
                    if z <= y:
                        continue
                    if lcm_table[x][z] == lxy and lcm_table[y][z] == lxy:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                result_set.add(x)
                removed[x] = False
    
    # Branch and bound on conflict graph for Maximum Independent Set
    # We want to maximize |S| such that no two elements in S are in conflict
    # This is equivalent to: find max subset of [1..n] with no bad triples
    
    def max_independent_set_bnb(candidates, current_size, best_size, best_set):
        """Branch and bound to find maximum independent set in conflict graph."""
        if time.time() > deadline:
            return best_size, best_set
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = set()
            return best_size, best_set
        
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best_size:
            return best_size, best_set
        
        # Pick vertex with max degree in induced subgraph on candidates
        cand_list = list(candidates)
        deg = {}
        for v in cand_list:
            deg[v] = len(conflict[v] & candidates)
        
        # Sort by degree descending
        cand_list.sort(key=lambda v: -deg[v])
        v = cand_list[0]
        
        # Branch 1: include v
        new_candidates = candidates - {v} - conflict[v]
        new_best_size, new_best_set = max_independent_set_bnb(
            new_candidates, current_size + 1, best_size, best_set
        )
        best_size = new_best_size
        best_set = new_best_set
        
        # Branch 2: exclude v
        new_candidates = candidates - {v}
        new_best_size, new_best_set = max_independent_set_bnb(
            new_candidates, current_size, best_size, best_set
        )
        best_size = new_best_size
        best_set = new_best_set
        
        return best_size, best_set
    
    # Use B&B starting from current solution
    if time.time() < deadline - 10:  # Leave 10s for final polish
        all_elements = set(range(1, n + 1))
        best_size, best_set = max_independent_set_bnb(all_elements, 0, len(result_set), set())
        
        if len(best_set) > len(result_set):
            result_set = best_set
    
    # Final validation
    result_list = sorted(result_set)
    if not result_list:
        return [1]
    
    if not is_valid_set(result_set):
        result_list = sorted(x for x in range(1, n + 1) if not removed[x])
        if not result_list:
            return [1]
    
    return result_list
