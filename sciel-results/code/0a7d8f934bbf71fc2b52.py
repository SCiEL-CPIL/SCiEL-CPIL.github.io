import time, random

def construct():
    n = 15
    t0 = time.time()
    deadline = t0 + 80.0

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]

    def is_valid(adj):
        deg = [0]*n
        for i in range(n):
            deg[i] = len(adj[i])
        for u in range(n):
            au = adj[u]
            for v in au:
                if v < u: continue
                av = adj[v]
                for s in au:
                    if s == v: continue
                    ds = deg[s]
                    for t in av:
                        if t == u or t == s: continue
                        if deg[t] == ds:
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
        # Deep copy
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
            # Prepare move
            u1, v1, u2, v2 = -1, -1, -1, -1
            is_add = False
            is_remove = False
            is_swap = False
            
            if move_type < 0.4: # Add
                missing = [(u, v) for (u, v) in all_edges if v not in current_adj[u]]
                if not missing: continue
                u1, v1 = rng.choice(missing)
                is_add = True
            elif move_type < 0.7: # Remove
                existing = [(u, v) for (u, v) in all_edges if v in current_adj[u]]
                if not existing: continue
                u2, v2 = rng.choice(existing)
                is_remove = True
            else: # Swap
                missing = [(u, v) for (u, v) in all_edges if v not in current_adj[u]]
                existing = [(u, v) for (u, v) in all_edges if v in current_adj[u]]
                if not missing or not existing: continue
                u1, v1 = rng.choice(missing)
                u2, v2 = rng.choice(existing)
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
                delta = 0 # Score change is 0 for swap? 
                # Wait, swap removes 1 and adds 1, so score is same.
                # But validity might change.
                # If we only accept valid moves, and score doesn't change, 
                # we should accept with probability 1 if it helps escape?
                # In valid-only SA, we accept if valid.
                # If score is same, we accept to explore.
                # But we need to track score correctly.
                # current_score is not updated yet.
                # Let's just check validity.
            
            if is_valid(current_adj):
                new_score = current_score + delta
                # For swap, delta is 0, so new_score == current_score
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
            
        adj, sc = simulate_annealing(start, 10.0, 0.01, 1000)
        if sc > best_score:
            best_score = sc
            best_adj = adj
            
    # Final polish
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
