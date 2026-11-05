import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute conflict adjacency as bitmasks
    adj = [0] * N
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) > N:
                adj[i-1] |= (1 << (j-1))
                adj[j-1] |= (1 << (i-1))
    
    # Precompute degree for each node
    degree = [adj[i].bit_count() for i in range(N)]
    
    # Greedy independent set with different orderings
    def greedy_is(order):
        avail = (1 << N) - 1
        result = set()
        for v in order:
            if avail & (1 << v):
                result.add(v + 1)
                avail &= ~(1 << v)
                avail &= ~adj[v]
        return result
    
    # Generate multiple orderings and take best
    def get_orderings():
        orderings = []
        # By degree ascending
        orderings.append(sorted(range(N), key=lambda x: degree[x]))
        # By value ascending  
        orderings.append(list(range(N)))
        # By value descending
        orderings.append(list(range(N-1, -1, -1)))
        # Random-ish: sort by (degree, value)
        orderings.append(sorted(range(N), key=lambda x: (degree[x], x)))
        # Sort by (value, degree)
        orderings.append(sorted(range(N), key=lambda x: (x, degree[x])))
        return orderings
    
    best_set = set()
    best_size = 0
    
    for order in get_orderings():
        s = greedy_is(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: 1-swap and 2-add improvements
    def is_valid_addition(current_set, candidate):
        c = candidate - 1
        for x in current_set:
            if adj[c] & (1 << (x-1)):
                return False
        return True
    
    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # 1-swap: remove one, add two
            for a in list(s):
                for b in range(1, N+1):
                    if b in s:
                        continue
                    if not is_valid_addition(s - {a}, b):
                        continue
                    # Try to add another after removing a and adding b
                    temp = s - {a} | {b}
                    for c in range(1, N+1):
                        if c in temp:
                            continue
                        if not is_valid_addition(temp, c):
                            continue
                        new_s = temp | {c}
                        if len(new_s) > len(s):
                            s = new_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # 2-add: add two without removing
            for a in range(1, N+1):
                if a in s:
                    continue
                if not is_valid_addition(s, a):
                    continue
                temp = s | {a}
                for b in range(1, N+1):
                    if b in temp:
                        continue
                    if not is_valid_addition(temp, b):
                        continue
                    new_s = temp | {b}
                    if len(new_s) > len(s):
                        s = new_s
                        improved = True
                        break
                if improved:
                    break
        return s
    
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    if time.time() > deadline - 5:
        return list(best_set)
    
    # Branch and bound with improved coloring
    def greedy_color_bound(available):
        if not available:
            return [], 0
        
        # Get vertices and sort by degree descending
        verts = []
        m = available
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        verts.sort(key=lambda v: (adj[v] & available).bit_count(), reverse=True)
        
        color = {}
        next_color = 0
        
        for v in verts:
            used_colors = set()
            nb_mask = adj[v] & available
            m2 = nb_mask
            while m2:
                lsb = m2 & -m2
                u = lsb.bit_length() - 1
                m2 ^= lsb
                if u in color:
                    used_colors.add(color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c > next_color:
                next_color = c
        
        res = [(v, color[v]) for v in verts]
        res.sort(key=lambda x: x[1], reverse=True)
        return res, next_color + 1
    
    def solve(available, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 2:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        if current_size + available.bit_count() <= best_size:
            return
        
        colored_verts, ub = greedy_color_bound(available)
        if current_size + ub <= best_size:
            return
        
        for v, c in colored_verts:
            if current_size + c <= best_size:
                break
            
            bit_v = 1 << v
            if not (available & bit_v):
                continue
            
            new_avail = available & ~adj[v] & ~bit_v
            solve(new_avail, current_size + 1, current_set | {v + 1})
            
            available &= ~bit_v
    
    all_nodes = (1 << N) - 1
    solve(all_nodes, 0, best_set)
    
    return list(best_set)
