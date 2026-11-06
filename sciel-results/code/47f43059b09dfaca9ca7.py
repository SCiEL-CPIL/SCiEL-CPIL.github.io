"""Hexagon-free graphs on 40 vertices: search for C6-free graphs with many edges."""
import random
import time
import sys

def _has_c6(adj, n):
    """Check if a simple undirected graph (adj as list of sets) contains a 6-cycle."""
    # A C6 is a cycle of length 6: v0-v1-v2-v3-v4-v5-v0
    # We can enumerate paths of length 5 and check if the end is adjacent to the start
    # and the path is simple (all vertices distinct) and v5 != v0 (already ensured by distinctness if we check)
    # More efficiently: for each pair (u, v) with u < v, check if there is a path of length 4 from u to v
    # such that all intermediate vertices are distinct and none equal u or v, and u-v is an edge.
    # Actually, let's enumerate all paths of length 5 (6 vertices) and check for cycles.
    
    # Better approach: enumerate all 5-length paths (v0,v1,v2,v3,v4,v5) where all consecutive
    # are edges, all 6 vertices distinct, and v0-v5 is an edge.
    # This is expensive. Let's use a smarter check.
    
    # For each edge (u, v), remove it, and check if there's a path of length 5 from u to v
    # in the remaining graph. But that's also expensive.
    
    # Alternative: count number of 6-cycles using adjacency matrix multiplication?
    # Too complex for this. Let's do direct enumeration but with pruning.
    
    # Actually, for n=40 and ~105 edges, the graph is relatively sparse.
    # Let's try: for each vertex v, for each pair of neighbors (a, b) of v, 
    # for each pair of neighbors (c, d) of some vertex... this is getting complicated.
    
    # Let's just do: for each ordered pair (u, w) with u < w, check if there exists
    # a path u-x1-x2-x3-w with all vertices distinct and {u,w,x1,x2,x3} all distinct,
    # and (u,w) is an edge. If so, we have a 6-cycle u-x1-x2-x3-w-u.
    
    # For each edge (u, w):
    #   Find all paths of length 4 from u to w avoiding the edge (u,w) directly.
    #   A path of length 4: u-a-b-c-w where a,b,c are distinct, not u or w, 
    #   and edges (u,a),(a,b),(b,c),(c,w) exist.
    
    for u in range(n):
        for w in adj[u]:
            if u >= w:
                continue
            # Check if there's a path u-a-b-c-w of length 4 with distinct internal vertices
            found = False
            for a in adj[u]:
                if a == w:
                    continue
                for b in adj[a]:
                    if b == u or b == w:
                        continue
                    if b == a:
                        continue
                    for c in adj[b]:
                        if c == u or c == w or c == a:
                            continue
                        if c == b:
                            continue
                        if w in adj[c]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                return True
    return False

def construct():
    """
    Construct a C6-free graph on 40 vertices maximizing the number of edges.
    Uses a greedy + local search approach with random restarts.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Start with a good initial guess: 8 copies of K5
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                best_edges.append([5 * k + a, 5 * k + b])
    best_count = len(best_edges)
    
    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) to the graph; return True if C6-free."""
        if u in adj[v]:
            return False
        # Temporarily add edge
        adj[u].add(v)
        adj[v].add(u)
        # Check for C6
        has_c6 = _has_c6(adj, n)
        if has_c6:
            adj[u].remove(v)
            adj[v].remove(u)
            return False
        return True
    
    # Local search: try to add edges greedily
    def local_search(edge_list):
        adj = build_adj(edge_list)
        improved = True
        while improved:
            improved = False
            # Try adding all possible missing edges in random order
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        missing.append((u, v))
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(adj, u, v):
                    edge_list.append([u, v])
                    improved = True
                    if len(edge_list) > best_count:
                        # Update best (but we're in a local search, so just continue)
                        pass
        return edge_list
    
    # Random restarts with different initial configurations
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random initial graph
        edge_list = []
        # Start with some random edges
        for _ in range(60 + random.randint(0, 30)):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edge_list.append([min(u, v), max(u, v)])
        
        # Remove duplicates
        seen = set()
        unique_edges = []
        for e in edge_list:
            key = tuple(sorted(e))
            if key not in seen:
                seen.add(key)
                unique_edges.append(list(key))
        edge_list = unique_edges
        
        # Remove any edges that create C6
        adj = build_adj(edge_list)
        valid_edges = []
        for u, v in edge_list:
            if try_add_edge(adj, u, v):
                valid_edges.append([u, v])
            else:
                # Need to remove from adj if it was added
                pass
        # Rebuild adj properly
        adj = build_adj(valid_edges)
        edge_list = valid_edges
        
        # Local search
        edge_list = local_search(edge_list)
        
        if len(edge_list) > best_count:
            best_count = len(edge_list)
            best_edges = [e[:] for e in edge_list]
        
        if best_count >= 110:
            break
    
    # One more pass: try to add edges to the best graph
    if time.time() < deadline - 5:
        adj = build_adj(best_edges)
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            improved = False
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        missing.append((u, v))
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 2:
                    break
                if try_add_edge(adj, u, v):
                    best_edges.append([u, v])
                    improved = True
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
