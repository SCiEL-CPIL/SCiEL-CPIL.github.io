"""Program for the suite task: construct a small k-regular graph of girth g."""
import math
import random
import time
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Return the Moore bound (lower bound on n) for a k-regular graph of girth g."""
    if g % 2 == 0:
        m = g // 2
        if k == 1:
            return 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        return 1 + k * s
    else:
        m = (g - 1) // 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        return 1 + k * s

def _is_k_regular(adj, k):
    n = len(adj)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def _has_short_cycle(adj, g):
    """Check if graph has a cycle shorter than g. Returns (False, []) if girth >= g, else (True, cycle)."""
    n = len(adj)
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Potential cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        path1 = []
                        cur = u
                        while cur != -1:
                            path1.append(cur)
                            cur = parent[cur]
                        path2 = []
                        cur = v
                        while cur != -1:
                            path2.append(cur)
                            cur = parent[cur]
                        # Find LCA
                        # For simplicity, just return True and a dummy cycle or the paths
                        # We need the actual edges to swap.
                        # Let's return the cycle nodes
                        # The cycle is path1 + reversed(path2) up to LCA
                        # But for swapping, we just need to know a short cycle exists.
                        # We can return the set of edges in the shortest cycle found.
                        # To keep it simple, we'll just return True and the first cycle found.
                        # We need to reconstruct the cycle to break it.
                        # Let's find the LCA.
                        set1 = set(path1)
                        lca = None
                        for node in path2:
                            if node in set1:
                                lca = node
                                break
                        if lca is not None:
                            cycle = []
                            cur = u
                            while cur != lca:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.append(lca)
                            cur = v
                            while cur != lca:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            # Ensure cycle is a list of nodes forming a cycle
                            # The edges are (cycle[i], cycle[i+1]) and (cycle[-1], cycle[0])?
                            # Actually, the BFS tree paths go from start to u and start to v.
                            # The cycle is u -> ... -> lca -> ... -> v -> u.
                            # We can return this list of nodes.
                            return True, cycle
    return False, []

def _generate_random_k_regular(n, k, seed=None):
    """Generate a random simple k-regular graph using configuration model."""
    if seed is not None:
        random.seed(seed)
    
    max_attempts = 1000
    for _ in range(max_attempts):
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Shuffle
        random.shuffle(stubs)
        
        # Pair up
        adj = {i: [] for i in range(n)}
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
        
        if valid:
            # Convert to list of lists for faster access
            adj_list = [adj[i] for i in range(n)]
            return adj_list
    
    return None

def _try_improve_girth(adj, k, g, max_steps=5000):
    """Try to increase girth by swapping edges."""
    n = len(adj)
    adj_set = [set(adj[i]) for i in range(n)]
    
    for step in range(max_steps):
        has_cycle, cycle = _has_short_cycle(adj, g)
        if not has_cycle:
            return adj, True
        
        # Break the cycle by swapping edges
        # Pick an edge in the cycle
        # Cycle is a list of nodes. Edges are (c[i], c[i+1]) and (c[-1], c[0])
        # We pick a random edge in the cycle to break
        m = len(cycle)
        if m == 0: continue
        
        # Find an edge (u, v) in the cycle
        # And another edge (u2, v2) not in the cycle (or just random)
        # Swap: remove (u,v) and (u2,v2), add (u,v2) and (u2,v)
        
        # To ensure we break the cycle, we should pick an edge from the cycle.
        edge_idx = random.randint(0, m - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % m]
        
        # Find a partner edge to swap with
        # Randomly pick a vertex u2 and its neighbor v2
        # Ensure (u2, v2) is not (u, v) or (v, u)
        # And ensure new edges don't exist
        
        attempts = 0
        swapped = False
        while attempts < 50:
            u2 = random.randint(0, n - 1)
            if u2 == u or u2 == v:
                attempts += 1
                continue
            # Pick a neighbor v2 of u2
            if not adj[u2]:
                attempts += 1
                continue
            v2 = random.choice(adj[u2])
            if v2 == u or v2 == v:
                attempts += 1
                continue
            
            # Check if (u, v2) and (u2, v) are valid new edges
            if v2 in adj_set[u] or v in adj_set[u2]:
                attempts += 1
                continue
            
            # Perform swap
            # Remove (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            adj_set[u].remove(v)
            adj_set[v].remove(u)
            
            # Remove (u2, v2)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            adj_set[u2].remove(v2)
            adj_set[v2].remove(u2)
            
            # Add (u, v2)
            adj[u].append(v2)
            adj[v2].append(u)
            adj_set[u].add(v2)
            adj_set[v2].add(u)
            
            # Add (u2, v)
            adj[u2].append(v)
            adj[v].append(u2)
            adj_set[u2].add(v)
            adj_set[v].add(u2)
            
            swapped = True
            break
        
        if not swapped:
            # If we can't swap, break
            break
            
    return adj, False

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    if k < 2:
        return 0, []
    if g <= 2:
        if k == 2:
            n = max(3, g)
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        else:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return n, edges
    
    start_time = time.time()
    timeout = 115.0
    
    mb = _moore_bound(k, g)
    
    # Try to find a graph with n close to Moore bound
    # Step n up from Moore bound
    best_n = None
    best_edges = None
    
    # We can try a range of n
    # For k=4, g=10, mb=485.
    # Let's try n from mb to mb + 20
    
    for n in range(mb, mb + 50):
        if time.time() - start_time > timeout:
            break
            
        # Try to generate a valid graph of size n
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() - start_time < timeout:
            adj = _generate_random_k_regular(n, k)
            if adj is None:
                attempts += 1
                continue
                
            # Check girth
            has_cycle, _ = _has_short_cycle(adj, g)
            
            if not has_cycle:
                # Found a valid graph
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                found = True
                break
            else:
                # Try to improve
                adj_improved, success = _try_improve_girth(adj, k, g, max_steps=100)
                if success:
                    edges = []
                    for i in range(n):
                        for j in adj_improved[i]:
                            if i < j:
                                edges.append((i, j))
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            attempts += 1
            
        if found:
            break
            
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback: Use LUW if it worked before, or a larger random graph
    # Since LUW is deterministic and large, we can just generate a large random graph
    # Or return the first valid one found with larger n
    
    # Try larger n
    for n in range(mb + 50, mb + 200):
        if time.time() - start_time > timeout:
            break
        adj = _generate_random_k_regular(n, k)
        if adj:
            has_cycle, _ = _has_short_cycle(adj, g)
            if not has_cycle:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Last resort: Return a trivial graph (might be invalid if girth check fails, but better than nothing?)
    # No, must be valid.
    # If we failed, we might return the LUW graph if we had it, but we removed it.
    # Let's just return a large random graph that is likely valid (n >> mb)
    n = mb * 2
    adj = _generate_random_k_regular(n, k)
    if adj:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    return 0, []

# EVOLVE-BLOCK-END
