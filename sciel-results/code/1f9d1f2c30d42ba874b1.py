import time
import random

def _check_valid(adj, n):
    """Check if no 3-path has endpoints of equal degree."""
    deg = [len(adj[i]) for i in range(n)]
    # Pre-compute degree sets for faster lookup?
    # For each edge (u, v), check all s in N(u)\{v}, t in N(v)\{u, s}
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _add_edge(adj, n, u, v):
    adj[u].add(v)
    adj[v].add(u)

def _remove_edge(adj, n, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def construct():
    n = 15
    deadline = time.time() + 80
    best_edges = []
    best_count = 0
    
    # Pre-generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Randomize edge order
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        
        # Greedy add
        for u, v in all_edges:
            if time.time() > deadline:
                break
            _add_edge(adj, n, u, v)
            if _check_valid(adj, n):
                pass # Keep
            else:
                _remove_edge(adj, n, u, v)
        
        # Local Search: Try to add more edges
        # Identify missing edges
        current_edges = set()
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    current_edges.add((i, j))
        
        missing_edges = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                _add_edge(adj, n, u, v)
                if _check_valid(adj, n):
                    improved = True
                    # Remove from missing list? 
                    # Since we are iterating a copy, we just mark it as added in adj
                    # But we need to update missing_edges for next iteration?
                    # Simpler: just try all missing edges again in next pass if improved
                    break 
                else:
                    _remove_edge(adj, n, u, v)
        
        # Update best
        count = sum(len(adj[i]) for i in range(n)) // 2
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        best_edges.append([i, j])
                        
    # Final verification and cleanup
    if not best_edges:
        return []
        
    # Convert to list of lists of ints
    final_edges = []
    for e in best_edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges
