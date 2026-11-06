"""Construct a Q3-free graph on 16 vertices maximizing edges."""

def construct():
    import time
    import random
    from itertools import combinations
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    total_pairs = len(all_pairs)  # 120
    
    def make_adj(edge_set):
        adj = [[0] * n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def has_q3(adj):
        """Check if graph contains Q3 as a subgraph."""
        # Q3 has 8 vertices, each of degree 3
        # Find all 8-vertex subsets and check if induced subgraph is Q3
        # For efficiency, first check degree constraints
        from itertools import combinations as comb
        
        # For each 8-subset, check if it induces Q3
        # Q3 properties: 8 vertices, 12 edges, each vertex degree 3 in the subgraph
        # Also Q3 is bipartite with parts of size 4 each
        
        for subset in comb(range(n), 8):
            idx = list(subset)
            idx_set = set(idx)
            
            # Count edges within the subset
            edge_count = 0
            degs = [0] * 8
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[idx[i]][idx[j]]:
                        edge_count += 1
                        degs[i] += 1
                        degs[j] += 1
            
            # Q3 has exactly 12 edges and all degrees = 3
            if edge_count != 12:
                continue
            if not all(d == 3 for d in degs):
                continue
            
            # Now check if the induced subgraph is isomorphic to Q3
            # Q3 is the unique 3-regular bipartite graph with 8 vertices
            # Check bipartiteness and that it's connected (Q3 is connected)
            
            # Build adjacency within the subset
            sub_adj = [[0] * 8 for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    if i != j and adj[idx[i]][idx[j]]:
                        sub_adj[i][j] = 1
            
            # Check if bipartite using BFS
            color = [-1] * 8
            from collections import deque
            for start in range(8):
                if color[start] != -1:
                    continue
                color[start] = 0
                q = deque([start])
                bipartite = True
                while q:
                    u = q.popleft()
                    for v in range(8):
                        if sub_adj[u][v]:
                            if color[v] == -1:
                                color[v] = 1 - color[u]
                                q.append(v)
                            elif color[v] == color[u]:
                                bipartite = False
                                break
                    if not bipartite:
                        break
                if not bipartite:
                    break
            
            if not bipartite:
                continue
            
            # Check connectedness
            visited = [False] * 8
            visited[0] = True
            stack = [0]
            while stack:
                u = stack.pop()
                for v in range(8):
                    if sub_adj[u][v] and not visited[v]:
                        visited[v] = True
                        stack.append(v)
            
            if not all(visited):
                continue
            
            # A 3-regular connected bipartite graph on 8 vertices with 4+4 parts is Q3
            return True
        
        return False
    
    def is_q3_free(edge_set):
        adj = make_adj(edge_set)
        return not has_q3(adj)
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Strategy: start from a dense candidate and greedily add edges
    # Initial dense candidate: complement of a sparse graph
    # Try multiple initializations
    
    best_edges = None
    best_count = 0
    
    # Try different initial seeds
    # Seed 1: K7 x K7 type structure (disconnected)
    # Actually, let's try to build a connected dense graph
    
    # Approach: start with a random dense graph and remove edges to eliminate Q3
    # Or start sparse and add edges greedily
    
    # Greedy addition approach
    for trial in range(50):
        if time.time() > deadline:
            break
        
        random.seed(trial)
        
        # Start with a base: complete bipartite K4,4 plus isolated vertices? No, that has Q3.
        # Let's start with K8 (one clique) + 8 isolated vertices, then add cross edges
        # K8 has 28 edges
        
        # Better: start with two K4s (12 edges) and add all cross edges (16) = 28 edges
        # Then add edges within each K4? No, K4 is already complete.
        # Add edges among the 8 isolated vertices and cross to make denser
        
        # Let's use a different strategy:
        # Start with all possible edges, then greedily remove to eliminate Q3
        # This might be too slow for 120 edges
        
        # Instead: greedy addition from a moderate base
        # Base: two K5s (20 edges) on {0-4} and {5-9}, plus 6 isolated
        # Then add edges among {10-15} and cross edges
        
        base_edges = []
        # Two K5s
        for i in range(5):
            for j in range(i+1, 5):
                base_edges.append((i, j))
        for i in range(5, 10):
            for j in range(i+1, 10):
                base_edges.append((i, j))
        
        current_edges = set(base_edges)
        
        if not is_q3_free(current_edges):
            continue
        
        current_count = len(current_edges)
        adj = make_adj(current_edges)
        
        # Candidate edges to add, in random order
        candidates = [e for e in all_pairs if e not in current_edges]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in candidates:
                if (u, v) in current_edges:
                    continue
                if time.time() > deadline:
                    break
                
                # Try adding edge
                current_edges.add((u, v))
                if not has_q3(make_adj(current_edges)):
                    improved = True
                else:
                    current_edges.discard((u, v))
            
            # Rebuild candidate list if improved
            if improved:
                candidates = [e for e in all_pairs if e not in current_edges]
                random.shuffle(candidates)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
    
    # Also try the known good K7 + tails construction
    k7_edges = set()
    for i in range(7):
        for j in range(i+1, 7):
            k7_edges.add((i, j))
    # Add vertex 14 connected to 0,1,2
    k7_edges.add((14, 0))
    k7_edges.add((14, 1))
    k7_edges.add((14, 2))
    # Add vertex 15 connected to 0,1
    k7_edges.add((15, 0))
    k7_edges.add((15, 1))
    # Connect 14, 15
    k7_edges.add((14, 15))
    
    if len(k7_edges) > best_count:
        best_count = len(k7_edges)
        best_edges = list(k7_edges)
    
    # Try K7 + K7 + K2 construction mentioned in findings
    # K7 on 0-6, K7 on 7-13, and edge (14,15)
    # This gives 21 + 21 + 1 = 43 edges
    k7k7k2_edges = set()
    for i in range(7):
        for j in range(i+1, 7):
            k7k7k2_edges.add((i, j))
    for i in range(7, 14):
        for j in range(i+1, 14):
            k7k7k2_edges.add((i, j))
    k7k7k2_edges.add((14, 15))
    
    if len(k7k7k2_edges) > best_count:
        best_count = len(k7k7k2_edges)
        best_edges = list(k7k7k2_edges)
    
    # Try to improve best_edges with local search
    if best_edges and time.time() < deadline:
        current_edges = set(best_edges)
        adj = make_adj(current_edges)
        
        # Try adding any missing edge
        for u, v in all_pairs:
            if (u, v) in current_edges:
                continue
            if time.time() > deadline:
                break
            current_edges.add((u, v))
            if not has_q3(make_adj(current_edges)):
                pass
            else:
                current_edges.discard((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
    
    if best_edges is None:
        # Fallback to K3,13
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    # Final verification
    adj = make_adj(best_edges)
    if has_q3(adj):
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return [[u, v] for u, v in best_edges]
