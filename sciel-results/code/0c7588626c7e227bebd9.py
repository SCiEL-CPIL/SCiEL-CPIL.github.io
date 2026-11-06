"""Optimized graph construction for minimizing max triangle degree sum."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_from_adj(adj, n, edge_set):
    """Compute the maximum degree sum over all triangles using bitmask intersection."""
    # Build adjacency bitmasks
    masks = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
    
    deg = adj.sum(axis=1).astype(int)
    
    best = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                # Common neighbors of u and v
                common = masks[u] & masks[v]
                while common:
                    # Extract lowest set bit
                    lsb = common & (-common)
                    w = lsb.bit_length() - 1
                    if w > v:  # Only count each triangle once
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                    common &= common - 1
    return best


def _local_search(adj, n, max_iters=50000, deadline=None):
    """Local search: try flipping edges to reduce max triangle degree sum."""
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
    
    current_score = _score_from_adj(adj, n, None)
    best_score = current_score
    best_adj = adj.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to try flipping
        idx = random.randint(0, len(edge_list) - 1)
        u, v = edge_list[idx]
        
        # Flip the edge
        adj[u, v] = 1 - adj[u, v]
        adj[v, u] = adj[u, v]
        
        new_score = _score_from_adj(adj, n, None)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
        else:
            # Revert
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
        
        # Occasionally try a random restart of local moves
        if it % 5000 == 0 and it > 0:
            # Try some random multi-edge flips
            for _ in range(3):
                idx2 = random.randint(0, len(edge_list) - 1)
                u2, v2 = edge_list[idx2]
                adj[u2, v2] = 1 - adj[u2, v2]
                adj[v2, u2] = adj[u2, v2]
                s2 = _score_from_adj(adj, n, None)
                if s2 < current_score:
                    current_score = s2
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                else:
                    adj[u2, v2] = 1 - adj[u2, v2]
                    adj[v2, u2] = adj[u2, v2]
    
    return best_score, best_adj


def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start from K_{6,10} + 5 internal edges (known better baseline)
    # Partition: 6 vertices in A, 10 in B
    # Complete bipartite K_{6,10} has 60 edges
    # Need 5 more edges, all within B (or within A)
    
    best_score = float('inf')
    best_edges = None
    
    # Initial construction: K_{6,10} with 5 edges within the 10-vertex part
    # Vertices 0-5 in A, 6-15 in B
    # K_{6,10}: all edges between A and B
    # Plus 5 edges within B: choose 5 edges from the 10-vertex set
    
    # Try several different sets of 5 internal edges
    b_vertices = list(range(6, 16))
    
    # Generate candidate internal edge sets
    random.seed(42)
    
    candidates = []
    
    # Strategy 1: spread internal edges to keep degrees balanced
    # In K_{6,10}, deg of A vertices = 10, deg of B vertices = 6
    # Adding internal edges in B increases some B degrees
    # We want to minimize the max triangle degree sum
    # Triangles come from: an A vertex connected to two B vertices that are also connected
    # deg(A) + deg(B1) + deg(B2) where B1-B2 is an internal edge
    
    # To minimize max, we should add internal edges between B vertices with lower current deg
    # But all B vertices have deg 6 initially
    # Adding edge (b1,b2) makes deg(b1)=7, deg(b2)=7
    # Triangle with A vertex: 10 + 7 + 7 = 24
    
    # If we add edges more spread out, some B vertices get higher degrees
    # Best to keep internal edge degrees as equal as possible
    
    # Try: 5 edges forming a matching plus one, or a path, or star, etc.
    # With 10 vertices and 5 edges, we can have a matching of 5 (perfect matching)
    # That gives each B vertex degree +1, so all B deg = 7
    # Triangle sum = 10 + 7 + 7 = 24
    
    # Can we do better? What if we use K_{7,9} instead?
    # K_{7,9} = 63 edges, need 2 more
    # A has 7 verts (deg 9), B has 9 verts (deg 7)
    # Add 2 internal edges in B: two B vertices get deg 8
    # Triangle: 9 + 8 + 8 = 25. Worse.
    
    # What about K_{5,11} = 55 edges, need 10 more
    # A deg 11, B deg 5
    # 10 internal edges in B (11 vertices)
    # If spread evenly, ~2 per vertex, B deg ~7
    # Triangle: 11 + 7 + 7 = 25. Worse.
    
    # So K_{6,10} + perfect matching on B seems optimal at 24.
    
    # Let's also try asymmetric internal edges
    # What if internal edges are within A instead?
    # K_{6,10}: A deg 10, B deg 6
    # 5 edges in A (6 vertices): perfect matching + 1
    # A vertices get higher deg
    # Triangle: 10+1+10+1... no, triangle needs one A and two B with B-B edge
    # If no B-B edges, no triangles! We need B-B edges for triangles.
    # Wait, we need exactly 65 edges. K_{6,10} = 60. Need 5 more.
    # If all 5 are in A, then no triangles exist (triangles need at least one edge in each part or 3 in one part)
    # Actually a triangle in a bipartite graph + internal edges: 
    # Triangle = (a1, b1, b2) where a1-b1, a1-b2, b1-b2 all present
    # So we NEED internal edges in B (or A) to create triangles
    # If internal edges are in A: triangle = (a1, a2, b1) where a1-a2, a1-b1, a2-b1
    # deg(a1)+deg(a2)+deg(b1) = (10+k1)+(10+k2)+6 where k1,k2 are extra degs from internal A edges
    
    # So both options work. Let's try both.
    
    # Option 1: 5 edges in B (perfect matching on 10 vertices)
    adj1 = np.zeros((n, n), dtype=int)
    for u in range(6):
        for v in range(6, 16):
            adj1[u, v] = 1
            adj1[v, u] = 1
    # Perfect matching on B: (6,7),(8,9),(10,11),(12,13),(14,15)
    for u, v in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
        adj1[u, v] = 1
        adj1[v, u] = 1
    
    score1 = _score_from_adj(adj1, n, None)
    if score1 < best_score:
        best_score = score1
        best_adj = adj1.copy()
    
    # Option 2: 5 edges in A
    adj2 = np.zeros((n, n), dtype=int)
    for u in range(6):
        for v in range(6, 16):
            adj2[u, v] = 1
            adj2[v, u] = 1
    # 5 edges in A (6 vertices): e.g., (0,1),(0,2),(0,3),(0,4),(0,5) - star
    for u, v in [(0,1),(0,2),(0,3),(0,4),(0,5)]:
        adj2[u, v] = 1
        adj2[v, u] = 1
    
    score2 = _score_from_adj(adj2, n, None)
    if score2 < best_score:
        best_score = score2
        best_adj = adj2.copy()
    
    # Option 3: 5 edges in A as matching + one
    adj3 = np.zeros((n, n), dtype=int)
    for u in range(6):
        for v in range(6, 16):
            adj3[u, v] = 1
            adj3[v, u] = 1
    # (0,1),(2,3),(4,5),(0,2),(0,4)
    for u, v in [(0,1),(2,3),(4,5),(0,2),(0,4)]:
        adj3[u, v] = 1
        adj3[v, u] = 1
    
    score3 = _score_from_adj(adj3, n, None)
    if score3 < best_score:
        best_score = score3
        best_adj = adj3.copy()
    
    # Now do local search from the best starting point
    if best_score > 23:  # Try to improve below 24
        best_score_ls, best_adj_ls = _local_search(
            best_adj.copy(), n, 
            max_iters=100000,
            deadline=deadline
        )
        if best_score_ls < best_score:
            best_score = best_score_ls
            best_adj = best_adj_ls
    
    # If still at 24, try more aggressive search
    if best_score >= 24 and time.time() < deadline - 10:
        # Try random restarts with different initial structures
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            adj_r = best_adj.copy()
            # Flip 5-10 random edges
            for _ in range(random.randint(5, 10)):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    adj_r[u, v] = 1 - adj_r[u, v]
                    adj_r[v, u] = adj_r[u, v]
            
            # Check edge count
            ec = adj_r.sum() // 2
            # Adjust to get exactly 65 edges
            while ec > 65:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and adj_r[u, v]:
                    adj_r[u, v] = 0
                    adj_r[v, u] = 0
                    ec -= 1
            while ec < 65:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and not adj_r[u, v]:
                    adj_r[u, v] = 1
                    adj_r[v, u] = 1
                    ec += 1
            
            sr = _score_from_adj(adj_r, n, None)
            if sr < best_score:
                best_score, best_adj = sr, adj_r.copy()
            
            # Local search from here
            sr2, adj_r2 = _local_search(adj_r, n, max_iters=30000, deadline=deadline)
            if sr2 < best_score:
                best_score = sr2
                best_adj = adj_r2
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    # Verify
    assert len(edges) == 65, f"Expected 65 edges, got {len(edges)}"
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
