import time, random

def construct():
    n = 15
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]

    def is_valid(adj):
        # Fast check: For each pair (s,t) with same degree, ensure no path of length 3
        deg = [0]*n
        for i in range(n):
            deg[i] = len(adj[i])
        
        # Group vertices by degree
        from collections import defaultdict
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)
            
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                s = vertices[i]
                for j in range(i+1, len(vertices)):
                    t = vertices[j]
                    # Check if there is a path s-u-v-t
                    # s -> u -> v -> t
                    # u in N(s), v in N(u) \ {s}, t in N(v) \ {u}
                    found = False
                    for u in adj[s]:
                        if u == t: continue # Path length 1, not 3
                        for v in adj[u]:
                            if v == s or v == u: continue
                            if t in adj[v] and t != u and t != s:
                                # Ensure s,u,v,t are distinct
                                if s != u and u != v and v != t and s != v and s != t and u != t:
                                    found = True
                                    break
                        if found: break
                    if found:
                        return False
        return True

    # Baseline K7,8
    best_adj = [set() for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            best_adj[i].add(j)
            best_adj[j].add(i)
    best_score = sum(len(s) for s in best_adj) // 2
    
    rng = random.Random(42)
    
    def simulate_annealing(start_adj, t_max, t_min, steps):
        current_adj = [set(s) for s in start_adj]
        current_score = sum(len(s) for s in current_adj) // 2
        
        if not is_valid(current_adj):
            return current_adj, current_score
            
        best_local_adj = [set(s) for s in current_adj]
        best_local_score = current_score
        
        T = t_max
        if steps <= 0:
            return best_local_adj, best_local_score
        cooling = (t_min / t_max) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            move_type = rng.random()
            u1, v1, u2, v2 = -1, -1, -1, -1
            is_add = False
            is_remove = False
            is_swap = False
            
            if move_type < 0.4: # Add
                # Pick random pair
                while True:
                    u1 = rng.randint(0, n-1)
                    v1 = rng.randint(0, n-1)
                    if u1 == v1: continue
                    if u1 > v1: u1, v1 = v1, u1
                    if v1 not in current_adj[u1]:
                        break
                is_add = True
            elif move_type < 0.7: # Remove
                while True:
                    u2 = rng.randint(0, n-1)
                    v2 = rng.randint(0, n-1)
                    if u2 == v2: continue
                    if u2 > v2: u2, v2 = v2, u2
                    if v2 in current_adj[u2]:
                        break
                is_remove = True
            else: # Swap
                # Pick random missing and random existing
                while True:
                    u1 = rng.randint(0, n-1)
                    v1 = rng.randint(0, n-1)
                    if u1 == v1: continue
                    if u1 > v1: u1, v1 = v1, u1
                    if v1 not in current_adj[u1]:
                        break
                while True:
                    u2 = rng.randint(0, n-1)
                    v2 = rng.randint(0, n-1)
                    if u2 == v2: continue
                    if u2 > v2: u2, v2 = v2, u2
                    if v2 in current_adj[u2]:
                        break
                is_swap = True

            # Apply move
            if is_add:
                current_adj[u1].add(v1)
                current_adj[v1].add(u1)
                delta = 1
            elif is_remove:
                current_adj[u2].discard(v2)
                current_adj[v2].discard(u2)
                delta = -1
            else:
                current_adj[u1].add(v1)
                current_adj[v1].add(u1)
                current_adj[u2].discard(v2)
                current_adj[v2].discard(u2)
                delta = 0
            
            if is_valid(current_adj):
                new_score = current_score + delta
                if delta > 0 or (delta == 0 and rng.random() < 0.5) or (delta < 0 and rng.random() < pow(2.718, delta / T)):
                    current_score = new_score
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_adj = [set(s) for s in current_adj]
                else:
                    # Reject, rollback
                    if is_add:
                        current_adj[u1].discard(v1)
                        current_adj[v1].discard(u1)
                    elif is_remove:
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
                    else:
                        current_adj[u1].discard(v1)
                        current_adj[v1].discard(u1)
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
            else:
                # Invalid, rollback
                if is_add:
                    current_adj[u1].discard(v1)
                    current_adj[v1].discard(u1)
                elif is_remove:
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                else:
                    current_adj[u1].discard(v1)
                    current_adj[v1].discard(u1)
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
            
            T *= cooling
        
        return best_local_adj, best_local_score

    restarts = 0
    while time.time() < deadline:
        restarts += 1
        if restarts % 5 == 0:
            start = [set() for _ in range(n)]
            for (u, v) in all_edges:
                if rng.random() < 0.6:
                    start[u].add(v)
                    start[v].add(u)
        else:
            start = [set(s) for s in best_adj]
            
        adj, sc = simulate_annealing(start, 10.0, 0.01, 500) # Reduced steps to allow more restarts
        if sc > best_score:
            best_score = sc
            best_adj = adj
            
    # Final polish: try adding all missing edges
    for (u, v) in all_edges:
        if v not in best_adj[u]:
            best_adj[u].add(v)
            best_adj[v].add(u)
            if is_valid(best_adj):
                best_score += 1
            else:
                best_adj[u].discard(v)
                best_adj[v].discard(u)
                
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if v > u:
                edges.append([int(u), int(v)])
                
    return edges
