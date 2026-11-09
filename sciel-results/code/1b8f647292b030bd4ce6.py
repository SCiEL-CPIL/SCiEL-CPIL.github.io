import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_violations(adj):
        degs = [len(adj[i]) for i in range(n)]
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
        
        violations = []
        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i+1, len(group)):
                    t = group[j]
                    # Check if path s-u-v-t exists
                    found = False
                    # u in N(s) \ {t}, v in N(t) \ {s}, (u,v) in E, u != v
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[t]:
                            if v == s:
                                continue
                            if u == v:
                                continue
                            if v in adj[u]:
                                violations.append((s, u, v, t))
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        return violations

    def is_valid(adj):
        return len(get_violations(adj)) == 0

    def try_add_edge(adj, u, v):
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        if is_valid(adj):
            return True
        else:
            adj[u].discard(v)
            adj[v].discard(u)
            return False

    def try_remove_edge(adj, u, v):
        if v not in adj[u]:
            return False
        adj[u].discard(v)
        adj[v].discard(u)
        if is_valid(adj):
            return True
        else:
            adj[u].add(v)
            adj[v].add(u)
            return False

    best_edge_count = 0
    best_adj = None
    
    # Run multiple trials
    for trial in range(500):
        if time.time() > deadline - 5:
            break
            
        random.seed(trial)
        # Start with a dense random graph
        p = random.uniform(0.6, 0.9)
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                
        # Fix phase: remove edges or change degrees to fix violations
        # We prioritize adding edges if possible, but initially we might need to remove
        # to get to a valid state.
        
        # Actually, if we start dense, we likely have many violations.
        # Let's try to fix by removing edges from violating paths first,
        # then add back later.
        
        max_fix_steps = 50
        for _ in range(max_fix_steps):
            if time.time() > deadline - 5:
                break
            violations = get_violations(adj)
            if not violations:
                break
            
            # Pick a random violation
            s, u, v, t = random.choice(violations)
            
            # Try to fix by removing an edge from the path
            # This guarantees breaking this specific path.
            # It might create new violations, but we loop.
            path_edges = [(s, u), (u, v), (v, t)]
            # Shuffle to avoid bias
            random.shuffle(path_edges)
            
            removed = False
            for pe in path_edges:
                if try_remove_edge(adj, pe[0], pe[1]):
                    removed = True
                    break
            
            if not removed:
                # If we couldn't remove (maybe already removed? shouldn't happen if valid logic)
                # Try changing degree of s or t
                # Try adding edge to s
                non_neighbors = [x for x in range(n) if x != s and x not in adj[s]]
                if non_neighbors:
                    target = random.choice(non_neighbors)
                    if try_add_edge(adj, s, target):
                        removed = True # 'fixed' flag
                if not removed:
                    non_neighbors = [x for x in range(n) if x != t and x not in adj[t]]
                    if non_neighbors:
                        target = random.choice(non_neighbors)
                        if try_add_edge(adj, t, target):
                            removed = True

        if is_valid(adj):
            # Fill phase: try to add edges to maximize count
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                # Randomize edge order for diversity
                edges_to_try = all_edges[:]
                random.shuffle(edges_to_try)
                for u, v in edges_to_try:
                    if time.time() > deadline - 5:
                        break
                    if v not in adj[u]:
                        if try_add_edge(adj, u, v):
                            changed = True
                            
            final_count = sum(len(adj[i]) for i in range(n)) // 2
            if final_count > best_edge_count:
                best_edge_count = final_count
                best_adj = [set(adj[i]) for i in range(n)]

    if best_adj is None:
        # Fallback: empty graph
        best_adj = [set() for _ in range(n)]
        
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
                
    return edges
