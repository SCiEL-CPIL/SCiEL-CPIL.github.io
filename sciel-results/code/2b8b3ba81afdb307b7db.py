import random
import time

def construct():
    def is_valid(n, edges):
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)): continue
                for w in range(v + 1, n):
                    if not (adj_mask[u] & (1 << w)) or not (adj_mask[v] & (1 << w)): continue
                    common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    mask_x = ((1 << n) - 1) ^ ((1 << (w + 1)) - 1)
                    if common & mask_x:
                        return False
        
        # Check I6 (Independent set of size 6)
        # This is equivalent to K6 in the complement.
        # We check if there are 6 vertices with no edges between them.
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] & (1 << v): continue
                for w in range(v + 1, n):
                    if adj_mask[u] & (1 << w) or adj_mask[v] & (1 << w): continue
                    for x in range(w + 1, n):
                        if adj_mask[u] & (1 << x) or adj_mask[v] & (1 << x) or adj_mask[w] & (1 << x): continue
                        for y in range(x + 1, n):
                            if adj_mask[u] & (1 << y) or adj_mask[v] & (1 << y) or adj_mask[w] & (1 << y) or adj_mask[x] & (1 << y): continue
                            for z in range(y + 1, n):
                                if adj_mask[u] & (1 << z) or adj_mask[v] & (1 << z) or adj_mask[w] & (1 << z) or adj_mask[x] & (1 << z) or adj_mask[y] & (1 << z): continue
                                return False
        return True

    def count_violations(n, edges):
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)): continue
                for w in range(v + 1, n):
                    if not (adj_mask[u] & (1 << w)) or not (adj_mask[v] & (1 << w)): continue
                    common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    mask_x = ((1 << n) - 1) ^ ((1 << (w + 1)) - 1)
                    k4_count += bin(common & mask_x).count('1')
        
        i6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] & (1 << v): continue
                for w in range(v + 1, n):
                    if adj_mask[u] & (1 << w) or adj_mask[v] & (1 << w): continue
                    for x in range(w + 1, n):
                        if adj_mask[u] & (1 << x) or adj_mask[v] & (1 << x) or adj_mask[w] & (1 << x): continue
                        for y in range(x + 1, n):
                            if adj_mask[u] & (1 << y) or adj_mask[v] & (1 << y) or adj_mask[w] & (1 << y) or adj_mask[x] & (1 << y): continue
                            for z in range(y + 1, n):
                                if adj_mask[u] & (1 << z) or adj_mask[v] & (1 << z) or adj_mask[w] & (1 << z) or adj_mask[x] & (1 << z) or adj_mask[y] & (1 << z): continue
                                i6_count += 1
        return k4_count + i6_count

    def local_search(n, initial_edges, time_limit):
        edges = set(tuple(e) for e in initial_edges)
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        start_time = time.time()
        best_edges = edges
        best_score = count_violations(n, list(edges))
        
        while time.time() - start_time < time_limit:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            new_edges = set(edges)
            if (u, v) in new_edges:
                new_edges.remove((u, v))
            else:
                new_edges.add((u, v))
            
            # Calculate score difference? Full recount is slow.
            # For simplicity, just recount if time allows, or use delta.
            # Given n=35, recount is ~10ms. 120s / 0.01s = 12000 iterations.
            score = count_violations(n, list(new_edges))
            if score < best_score:
                best_score = score
                best_edges = new_edges
                if best_score == 0:
                    return list(best_edges)
            else:
                # Accept with some probability (Simulated Annealing)
                # Or just reject to keep it simple (Hill Climbing)
                # Hill climbing might get stuck.
                if random.random() < 0.1:
                    best_edges = new_edges
                    best_score = score
        
        return list(best_edges)

    # 1. Try Paley graphs
    best_candidate = None
    for q in [29, 17, 13]:
        if q > 40: continue
        squares = {x * x % q for x in range(1, q)}
        edges = [[i, j] for i in range(q) for j in range(i + 1, q) if (j - i) % q in squares]
        if is_valid(q, edges):
            best_candidate = {'n': q, 'edges': edges}
            break # Since we iterate descending, first valid is best
    
    # 2. Try to find a larger graph (up to 35) via local search
    # We know 35 is possible. Let's try to find it.
    # Start with a random graph of size 35? Or extend best_candidate?
    # Extending is hard. Random restart is better.
    target_n = 35
    start_time = time.time()
    found_35 = None
    
    # Try a few random starts
    for _ in range(5):
        if time.time() - start_time > 100: break
        random_edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    random_edges.append([i, j])
        
        # Run local search for a short time
        result_edges = local_search(target_n, random_edges, 10)
        if is_valid(target_n, result_edges):
            found_35 = {'n': target_n, 'edges': result_edges}
            break
            
    if found_35:
        return found_35
        
    if best_candidate:
        return best_candidate
        
    # Fallback
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in {x * x % 17 for x in range(1, 17)}]}
