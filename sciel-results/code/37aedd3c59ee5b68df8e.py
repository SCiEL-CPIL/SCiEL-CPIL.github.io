import time
import random

_N = 15
_DEADLINE = 50.0 # Safety margin

def _is_valid(adj):
    n = _N
    deg = [len(adj[i]) for i in range(n)]
    # Check all 3-paths s-u-v-t
    for u in range(n):
        for v in adj[u]:
            if v <= u: continue # Unnecessary for path check, but good for iteration
            # Path s-u-v-t
            # s in adj[u], s != v
            # t in adj[v], t != u
            # s != t
            # deg[s] != deg[t]
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u or t == s: continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _get_violations(adj):
    n = _N
    deg = [len(adj[i]) for i in range(n)]
    violations = []
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u or t == s: continue
                    if deg[s] == deg[t]:
                        violations.append((s, u, v, t))
    return violations

def construct():
    start_time = time.time()
    n = _N
    # Initialize K_n
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
            
    best_edges = None
    best_count = -1
    
    # Greedy Removal
    # We might run this multiple times with different random seeds
    for seed in range(10):
        if time.time() - start_time > _DEADLINE:
            break
        random.seed(seed)
        # Copy adj
        curr_adj = [set(s) for s in adj]
        
        # Remove edges until valid
        while True:
            if time.time() - start_time > _DEADLINE:
                break
            violations = _get_violations(curr_adj)
            if not violations:
                break
            
            # Count edge frequency in violations
            edge_counts = {}
            for s, u, v, t in violations:
                # Edges in path: (s,u), (u,v), (v,t)
                edges_in_path = [(s,u), (u,v), (v,t)]
                for e in edges_in_path:
                    key = (min(e), max(e))
                    edge_counts[key] = edge_counts.get(key, 0) + 1
            
            # Pick edge with max count
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            e_to_remove = random.choice(candidates)
            u, v = e_to_remove
            curr_adj[u].remove(v)
            curr_adj[v].remove(u)
            
        if time.time() - start_time > _DEADLINE:
            break
            
        # Try to add edges back
        # Iterate through all missing edges
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > _DEADLINE:
                break
            for i in range(n):
                for j in range(i+1, n):
                    if j not in curr_adj[i]:
                        # Try adding
                        curr_adj[i].add(j)
                        curr_adj[j].add(i)
                        if _is_valid(curr_adj):
                            improved = True
                        else:
                            curr_adj[i].remove(j)
                            curr_adj[j].remove(i)
                            
        # Record best
        count = sum(len(s) for s in curr_adj) // 2
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                for j in curr_adj[i]:
                    if i < j:
                        best_edges.append([i, j])
                        
    if best_edges is None:
        # Fallback: Empty graph or Star
        # Star is valid?
        # Path s-u-v-t.
        # In star, center 0, leaves 1..14.
        # Path 1-0-2-? No, 2 is leaf, only connected to 0.
        # So no path of length 3 exists in a star graph (diameter 2).
        # So Star is valid.
        best_edges = [[0, i] for i in range(1, n)]
        
    return best_edges
