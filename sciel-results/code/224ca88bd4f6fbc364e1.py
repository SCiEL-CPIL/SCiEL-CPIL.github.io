import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    A = list(range(7))       # vertices 0-6
    B = list(range(7, 15))   # vertices 7-14
    
    # Full edge space
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
    
    # Base: K_{7,8} (all inter edges)
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
    
    # Intra edges (candidates to add)
    intra_edges = list(all_edges - base_edges)
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        """Check no path s-u-v-t has deg[s] == deg[t] in the full graph."""
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Start with base
    best = set(base_edges)
    best_count = len(best)
    
    def try_add(edges, e):
        test = edges | {e}
        if is_valid(test):
            return test
        return None
    
    # 1. Try adding single intra edges to base
    for e in intra_edges:
        if time.time() > deadline:
            break
        result = try_add(best, e)
        if result is not None:
            best = result
            best_count = len(best)
            
    # 2. Try removing one base edge and adding one or two intra edges
    # This allows bypassing the "path through base edges" constraint
    # We will try all single removals + single additions
    # And single removals + double additions (if fast enough)
    
    # Pre-compute base adj for speed? No, just use sets.
    
    # Strategy: For each base edge to remove, try adding intra edges.
    # To save time, we only try a subset of removals/additions if time is tight.
    
    random.seed(42)
    shuffled_base = list(base_edges)
    random.shuffle(shuffled_base)
    
    # Try removing 1 edge and adding 1
    for rem in shuffled_base:
        if time.time() > deadline:
            break
        
        # Only try removals that might help: 
        # Removing an edge (a,b) breaks paths a-b-...
        # We want to add intra edges.
        
        candidate_base = base_edges - {rem}
        
        # Try adding each intra edge
        for add in intra_edges:
            if time.time() > deadline:
                break
            result = try_add(candidate_base, add)
            if result is not None:
                if len(result) > best_count:
                    best = result
                    best_count = len(result)
                
    # Try removing 1 edge and adding 2
    # This is O(R * A^2). R=56, A=63. 56*4000 = 224k checks.
    # Each check is ~1000 ops. 200M ops. Might be slow in Python.
    # Let's limit this.
    
    # We only try this if we have time.
    # Check time
    if time.time() < deadline - 30:
        for rem in shuffled_base:
            if time.time() > deadline - 10:
                break
            
            candidate_base = base_edges - {rem}
            
            # Try pairs of intra edges
            for i in range(len(intra_edges)):
                if time.time() > deadline - 5:
                    break
                e1 = intra_edges[i]
                # Add e1
                res1 = try_add(candidate_base, e1)
                if res1 is None:
                    continue
                # Try adding e2 to res1
                for j in range(i+1, len(intra_edges)):
                    if time.time() > deadline - 5:
                        break
                    e2 = intra_edges[j]
                    res2 = try_add(res1, e2)
                    if res2 is not None:
                        if len(res2) > best_count:
                            best = res2
                            best_count = len(res2)

    # Build result
    result = [list(e) for e in best]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
