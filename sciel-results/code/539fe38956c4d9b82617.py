"""Program that searches for a 3-regular bipartite graph on 12 vertices with minimum distinct cycle lengths."""
import random
from itertools import combinations

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_triangle(adj, n):
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return True
    return False

def _cycle_lengths(adj, n, max_len=12):
    """Find all simple cycle lengths in the graph."""
    lengths = set()
    # For each pair of non-adjacent vertices, find shortest path
    # If shortest path length is k, then there's a cycle of length k+1
    # Actually, we need all simple cycle lengths.
    # For small graphs, we can use BFS from each vertex to find all simple cycles.
    # Better approach: for each edge, find all simple cycles containing that edge.
    # Or: enumerate all simple cycles using DFS with pruning.
    
    # Since n=12, we can try a DFS-based cycle enumeration
    # But that might be slow. Let's use a different approach.
    
    # For each pair of vertices (u,v) not connected by an edge, find all simple paths
    # of length up to max_len-1. Each path of length L gives a cycle of length L+1.
    # But this counts each cycle multiple times.
    
    # Simpler: for each edge (u,v), find all simple paths from u to v not using that edge.
    # Each such path of length L gives a cycle of length L+1.
    
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    
    for (u, v) in edge_set:
        # Find all simple paths from u to v that don't use edge (u,v)
        # Use DFS
        found = set()
        def dfs(node, target, path, visited):
            if len(path) > max_len:
                return
            if node == target and len(path) >= 2:
                found.add(len(path))
                return
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        visited = {u}
        dfs(u, v, [u], visited)
        for L in found:
            lengths.add(L + 1)
    
    return lengths

def construct():
    n = 12
    # Bipartition: A = 0..5, B = 6..11
    A = list(range(6))
    B = list(range(6, 12))
    
    # Try to find a graph with only 4-cycles (value 1)
    # If not found, fall back to K_{3,3} + K_{3,3} (value 2)
    
    best_edges = None
    best_value = 3  # K_{3,3}+K_{3,3} has value 2, so we want < 2 ideally, but 2 is fallback
    
    # First, try randomized search for value 1
    random.seed(42)
    for trial in range(500):
        # Start with empty graph
        edges = []
        adj = [set() for _ in range(n)]
        
        # Greedily add edges to reach degree 3 for all vertices
        # Only allow edges between A and B
        # Check that adding edge doesn't create cycle of length > 4
        
        success = True
        for _ in range(n):  # Need at least 18 edges total (12*3/2)
            # Find vertices with degree < 3
            candidates = []
            for u in A:
                if len(adj[u]) < 3:
                    for v in B:
                        if v not in adj[u]:
                            candidates.append((u, v))
            
            if not candidates:
                break
            
            # Try each candidate, pick one that doesn't create a bad cycle
            valid_candidates = []
            for (u, v) in candidates:
                # Check if adding (u,v) creates a cycle of length > 4
                # Temporarily add edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Check for cycles of length 5 or more involving this edge
                # Find paths from u to v of length >= 4 (which would make cycle length >= 5)
                # Actually, a cycle of length L means a path of length L-1 between u and v
                # So we need to check if there's a path of length >= 4 between u and v not using (u,v)
                
                bad = False
                # Check path of length 4 (cycle of length 5)
                # u -> x -> y -> z -> v where x in B, y in A, z in B
                for x in list(adj[u]):
                    if x == v:
                        continue
                    for y in adj[x]:
                        if y == u:
                            continue
                        for z in adj[y]:
                            if z == v:
                                # Path: u-x-y-z-v, length 4, cycle length 5
                                bad = True
                                break
                            if z in adj[v]:
                                # Wait, we need z connected to v
                                pass
                        if bad:
                            break
                    if bad:
                        break
                if bad:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    continue
                
                # Check path of length 5 (cycle of length 6)
                # u -> x -> y -> z -> w -> v
                bad = False
                for x in list(adj[u]):
                    if x == v:
                        continue
                    for y in adj[x]:
                        if y == u:
                            continue
                        for z in adj[y]:
                            if z == u or z == x:
                                continue
                            for w in adj[z]:
                                if w == v:
                                    bad = True
                                    break
                                if w == y or w == z:
                                    continue
                                # Check if w is connected to v
                                if v in adj[w]:
                                    bad = True
                                    break
                            if bad:
                                break
                        if bad:
                            break
                    if bad:
                        break
                if bad:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    continue
                
                # Also check longer paths up to length 10
                # For simplicity, check up to length 6
                bad = False
                for x in list(adj[u]):
                    if x == v:
                        continue
                    for y in adj[x]:
                        if y == u:
                            continue
                        for z in adj[y]:
                            if z == u or z == x:
                                continue
                            for w in adj[z]:
                                if w == v or w == y or w == z:
                                    continue
                                for t in adj[w]:
                                    if t == v:
                                        bad = True
                                        break
                                    if t == x or t == z or t == w:
                                        continue
                                    if v in adj[t]:
                                        bad = True
                                        break
                                if bad:
                                    break
                            if bad:
                                break
                        if bad:
                            break
                    if bad:
                        break
                if bad:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    continue
                
                valid_candidates.append((u, v))
            
            if not valid_candidates:
                success = False
                break
            
            # Pick a random valid candidate
            (u, v) = random.choice(valid_candidates)
            edges.append([u, v])
            # adj already has the edge from the check
        
        # Check if all vertices have degree >= 3
        if success:
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= 3:
                # Check triangle-free
                if not _has_triangle(adj, n):
                    # Count distinct cycle lengths
                    lengths = _cycle_lengths(adj, n)
                    value = len(lengths)
                    if value < best_value:
                        best_value = value
                        best_edges = [list(e) for e in edges]
                        if value == 1:
                            return best_edges
    
    # Fallback: K_{3,3} + K_{3,3}
    edges = []
    for offset in (0, 6):
        left = [offset, offset + 1, offset + 2]
        right = [offset + 3, offset + 4, offset + 5]
        for u in left:
            for v in right:
                edges.append([int(u), int(v)])
    return edges
