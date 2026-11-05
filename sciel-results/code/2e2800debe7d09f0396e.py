import time
import random

def construct():
    N = 20
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # masks[i] is a bitmask of neighbors of i
    masks = [0] * N
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(u, v):
        # Check if adding edge (u, v) creates a K3,3
        # Triples (u, x, y) with x, y != u
        for x in range(N):
            if x == u: continue
            for y in range(N):
                if y == u or y == x: continue
                common = (masks[u] | (1 << v)) & masks[x] & masks[y]
                if popcount(common) >= 3:
                    return False
        # Triples (x, y, v) with x, y != v, and x, y != u (to avoid double checking u triples? 
        # Actually, if x=u, it's covered above. If y=u, covered.
        # So x, y must be distinct from u and v?
        # If x=u, triple is (u, u, v) which is invalid (distinct vertices).
        # So we just need x, y != v.
        # But we already checked triples with u.
        # Triples with v but not u: (x, y, v) where x, y != u, v.
        for x in range(N):
            if x == v or x == u: continue
            for y in range(N):
                if y == v or y == u or y == x: continue
                common = masks[x] & masks[y] & (masks[v] | (1 << u))
                if popcount(common) >= 3:
                    return False
        return True

    def try_add_edge(u, v):
        if u == v: return False
        if (masks[u] >> v) & 1: return False
        if is_valid_add(u, v):
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            return True
        return False

    def count_edges():
        return sum(popcount(m) for m in masks) // 2

    # Initialize with GQ(2,2) structure on vertices 0..14
    # Points of GQ(2,2) can be mapped to edges of K3,3.
    # Vertices of K3,3 are A={0,1,2}, B={3,4,5}.
    # Edges are (a,b) for a in A, b in B. Total 9? No, 3*3=9.
    # Wait, GQ(2,2) has 15 points.
    # The collinearity graph of GQ(2,2) is the line graph of the Petersen graph? No.
    # Let's just use a random initialization if GQ construction is complex.
    # But the prompt suggests GQ.
    # Let's try to generate GQ(2,2) collinearity graph.
    # GQ(2,2) points: (i, j) where i,j in {0,1,2}? No, that's 9.
    # It has 15 points.
    # Actually, the complement of the collinearity graph of GQ(2,2) is the line graph of K3,3?
    # Let's stick to a strong heuristic: Random restart with local search.
    # The GQ structure is a good seed, but random might find a better configuration for n=20.
    
    best_edges = 0
    best_masks = masks[:]
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Reset masks
        masks = [0] * N
        
        # Initialization: Random graph with p=0.5, then prune
        # Or start with GQ if we can build it.
        # Let's try a dense random start and prune.
        # This might be slow to prune.
        # Better: Start with empty, add edges greedily.
        
        # Greedy Initialization
        random.seed(42 + restart)
        edges_to_try = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            try_add_edge(u, v)
            
        current_edges = count_edges()
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
            
        # Local Search
        # We will use a "kick" strategy: if stuck, remove random edges.
        while time.time() < deadline:
            improved = False
            # Try to add edges
            non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if not ((masks[u] >> v) & 1)]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if try_add_edge(u, v):
                    improved = True
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
            
            if not improved:
                # Stuck
                # Remove a random edge to escape
                all_edges = [(u, v) for u in range(N) for v in range(u+1, N) if (masks[u] >> v) & 1]
                if not all_edges:
                    break
                # Remove 1-3 random edges
                for _ in range(random.randint(1, 3)):
                    if not all_edges: break
                    u, v = random.choice(all_edges)
                    all_edges.remove((u,v))
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    current_edges -= 1
                    
                # Try to add edges again
                non_edges = [(a, b) for a in range(N) for b in range(a+1, N) if not ((masks[a] >> b) & 1)]
                random.shuffle(non_edges)
                for a, b in non_edges:
                    if try_add_edge(a, b):
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]

    # Reconstruct edge list from best_masks
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_masks[u] >> v) & 1:
                edges.append([u, v])
    return edges
