import time
import random

def construct():
    n = 15
    deadline = time.time() + 85
    
    def check_valid(edges_list):
        """Check if graph satisfies constraint. Returns True if valid."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges_list:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edges_list:
            degs_u = set(deg[s] for s in adj[u] if s != v)
            degs_v = set(deg[t] for t in adj[v] if t != u)
            if degs_u & degs_v:
                return False
        return True
    
    def get_violations(edges_list):
        """Return set of violating edges."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges_list:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        violations = set()
        for u, v in edges_list:
            degs_u = set(deg[s] for s in adj[u] if s != v)
            degs_v = set(deg[t] for t in adj[v] if t != u)
            if degs_u & degs_v:
                violations.add((min(u,v), max(u,v)))
        return violations
    
    def local_search(edges_list):
        """Aggressive local search: add edges, remove violations, swap."""
        edges = set((min(u,v), max(u,v)) for u, v in edges_list)
        
        # First, remove all violations
        while True:
            if time.time() > deadline:
                break
            viols = get_violations(list(edges))
            if not viols:
                break
            # Remove one violating edge
            v = next(iter(viols))
            edges.discard(v)
        
        # Now try to add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get current degrees for quick check
            deg = [0] * n
            adj = [set() for _ in range(n)]
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
            
            # Try adding each missing edge
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in edges:
                        continue
                    # Quick check: would adding (i,j) create a violation?
                    # After adding, deg[i] and deg[j] increase by 1
                    # For edge (i,j) itself: neighbors of i (excl j) and neighbors of j (excl i)
                    # For other edges incident to i or j, degrees change
                    # This is complex, so just try and check
                    edges.add((i, j))
                    if not get_violations(list(edges)):
                        improved = True
                    else:
                        edges.discard((i, j))
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
        
        # Try edge swaps: remove one edge, add another
        if time.time() < deadline:
            current_edges = list(edges)
            for remove_e in current_edges:
                if time.time() > deadline:
                    break
                edges.discard(remove_e)
                # Try adding each missing edge
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) in edges or (i,j) == (min(remove_e), max(remove_e)):
                            continue
                        edges.add((i, j))
                        if not get_violations(list(edges)):
                            improved = True
                        else:
                            edges.discard((i, j))
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                # If no improvement from this removal, put it back
                if not improved:
                    edges.add(remove_e)
        
        return list(edges)
    
    best = []
    best_len = 0
    
    # Generate diverse starting candidates
    candidates = []
    
    # Candidate 1: K_{7,8} bipartite
    c1 = [(i, j) for i in range(7) for j in range(7, 15)]
    candidates.append(c1)
    
    # Candidate 2: K_{8,7} bipartite (same as above but different partition)
    c2 = [(i, j) for i in range(8) for j in range(8, 15)]
    candidates.append(c2)
    
    # Candidate 3: Complete graph on 15 minus edges to create degree variation
    # Try: include edge (i,j) if i+j < threshold
    for thresh in [12, 13, 14, 15, 16]:
        c = [(i, j) for i in range(n) for j in range(i+1, n) if i+j < thresh]
        candidates.append(c)
    
    # Candidate 4: Turan graph T(15,3)
    parts = [[0,3,6,9,12], [1,4,7,10,13], [2,5,8,11,14]]
    c4 = []
    for a in range(3):
        for b in range(a+1, 3):
            for i in parts[a]:
                for j in parts[b]:
                    if i < j:
                        c4.append((i, j))
                    else:
                        c4.append((j, i))
    candidates.append(c4)
    
    # Candidate 5: Random sparse graphs
    random.seed(42)
    for _ in range(5):
        c5 = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    c5.append((i, j))
        candidates.append(c5)
    
    # Evaluate candidates and run local search on promising ones
    for cand in candidates:
        if time.time() > deadline:
            break
        # First check validity, if not valid, try to fix
        valid = check_valid(cand)
        if valid and len(cand) > best_len:
            best_len = len(cand)
            best = list(cand)
        
        # Run local search to improve
        improved = local_search(cand)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved
    
    # Try multiple random restarts with local search
    random.seed(123)
    for restart in range(10):
        if time.time() > deadline:
            break
        # Start from a random graph and add edges greedily
        edges = set()
        # Add edges in random order, keeping valid ones
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for e in all_edges:
            if time.time() > deadline:
                break
            edges.add(e)
            if get_violations(list(edges)):
                edges.discard(e)
        
        # Local search
        improved = local_search(list(edges))
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved
    
    # Convert to proper format
    result = []
    for u, v in best:
        if u < v:
            result.append([int(u), int(v)])
        else:
            result.append([int(v), int(u)])
    
    return result
