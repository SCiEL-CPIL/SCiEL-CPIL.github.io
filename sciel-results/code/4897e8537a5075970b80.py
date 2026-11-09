import time
import random
import numpy as np
from collections import defaultdict

N = 40
DEADLINE = 110  # Seconds

def _creates_c6(adj, u, v):
    """
    Checks if adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate a in N(u)
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        if a == v: continue
        
        # Iterate d in N(v)
        d_mask = nv
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            if d == u or d == a: continue
            
            # We need b in N(a) and c in N(d) such that (b, c) is an edge.
            # b, c must be distinct from u, v, a, d.
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            # Candidates for b: neighbors of a, excluding forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask: continue
            
            # Candidates for c: neighbors of d, excluding forbidden
            c_mask = adj[d] & ~forbidden
            if not c_mask: continue
            
            # Check if any b in b_mask is connected to any c in c_mask
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if (adj[b] & c_mask):
                    return True
                        
    return False

def _build_conflict_graph(adj):
    """
    Build conflict graph where nodes are non-edges of the original graph.
    Two non-edges conflict if adding both would create a C6.
    """
    # Get all non-edges
    non_edges = []
    non_edge_idx = {}
    for i in range(N):
        for j in range(i+1, N):
            if not (adj[i] >> j) & 1:
                idx = len(non_edges)
                non_edges.append((i, j))
                non_edge_idx[(i, j)] = idx
    
    m = len(non_edges)
    if m == 0:
        return non_edges, [], 0
    
    # Build conflict adjacency as bitmasks
    conflict = [0] * m
    
    # For each pair of non-edges, check if they conflict
    # Two edges (u,v) and (x,y) conflict if adding both creates a C6
    # This is expensive, so we use a heuristic: check if they share endpoints or form C6
    
    for i in range(m):
        u, v = non_edges[i]
        for j in range(i+1, m):
            x, y = non_edges[j]
            # Quick check: do they share a vertex?
            if u in (x, y) or v in (x, y):
                # They share a vertex, need to check more carefully
                # Actually, sharing a vertex doesn't necessarily mean C6
                # But it's a necessary condition for some C6 configurations
                # Let's just mark them as potential conflicts and verify later
                # For now, we'll use a simpler conflict model
                pass
            
            # Check if adding both edges creates a C6
            # Temporarily add edge (u,v) and check if (x,y) creates C6
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if _creates_c6(adj, x, y):
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
            
            # Remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return non_edges, conflict, m

def _greedy_color(conflict, m):
    """
    Greedy coloring of conflict graph to get upper bound.
    Returns list of color classes (each a list of node indices).
    """
    colors = [-1] * m
    color_classes = []
    
    for i in range(m):
        # Find a color not used by neighbors
        neighbor_colors = set()
        conf = conflict[i]
        while conf:
            lsb = conf & -conf
            j = lsb.bit_length() - 1
            conf ^= lsb
            if colors[j] != -1:
                neighbor_colors.add(colors[j])
        
        # Find smallest available color
        c = 0
        while c in neighbor_colors:
            c += 1
        
        colors[i] = c
        if c >= len(color_classes):
            color_classes.append([])
        color_classes[c].append(i)
    
    return color_classes

def _branch_and_bound(non_edges, conflict, m, adj, deadline, best_count):
    """
    Branch and bound for maximum independent set in conflict graph.
    """
    start_time = time.time()
    best_set = []
    best_weight = 0
    
    # Precompute: for each node, its weight (always 1 for this problem)
    weights = [1] * m
    
    # Initial greedy solution
    current_set = set()
    remaining = set(range(m))
    
    # Greedy: pick nodes with fewest conflicts first
    order = sorted(range(m), key=lambda i: bin(conflict[i]).count('1'))
    
    for i in order:
        if i in remaining:
            # Check if i conflicts with any in current_set
            conflicts_with_set = False
            conf = conflict[i]
            for j in current_set:
                if (conf >> j) & 1:
                    conflicts_with_set = True
                    break
            if not conflicts_with_set:
                current_set.add(i)
                remaining.discard(i)
                # Remove neighbors of i from remaining
                conf = conflict[i]
                to_remove = set()
                while conf:
                    lsb = conf & -conf
                    j = lsb.bit_length() - 1
                    conf ^= lsb
                    if j in remaining:
                        to_remove.add(j)
                remaining -= to_remove
    
    best_weight = len(current_set)
    best_set = list(current_set)
    
    # Branch and bound
    def bbs(current, remaining, depth):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > deadline:
            return
        
        if not remaining:
            if len(current) > best_weight:
                best_weight = len(current)
                best_set = list(current)
            return
        
        # Upper bound: current + greedy coloring of remaining
        # For simplicity, use |remaining| as upper bound
        if len(current) + len(remaining) <= best_weight:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with highest degree in remaining subgraph
        best_v = -1
        best_deg = -1
        rem = list(remaining)
        for v in rem:
            deg = 0
            conf = conflict[v]
            for u in rem:
                if (conf >> u) & 1:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_current = current + [best_v]
        new_remaining = set(remaining)
        new_remaining.discard(best_v)
        # Remove neighbors of best_v
        conf = conflict[best_v]
        to_remove = set()
        while conf:
            lsb = conf & -conf
            u = lsb.bit_length() - 1
            conf ^= lsb
            if u in new_remaining:
                to_remove.add(u)
        new_remaining -= to_remove
        
        bbs(new_current, new_remaining, depth + 1)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: exclude best_v
        new_remaining2 = set(remaining)
        new_remaining2.discard(best_v)
        
        bbs(current, new_remaining2, depth + 1)
    
    bbs([], set(range(m)), 0)
    
    return best_set

def _add_edges_from_set(adj, non_edges, edge_set):
    """
    Add edges from the independent set to the adjacency.
    """
    for idx in edge_set:
        u, v = non_edges[idx]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def construct():
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Greedy construction with random restarts
    while time.time() - start_time < DEADLINE * 0.4:
        random.shuffle(all_edges)
        adj = [0] * N
        count = 0
        
        for u, v in all_edges:
            if not _creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
    
    # Phase 2: Use MWIS approach on the best graph
    if best_count > 0 and time.time() - start_time < DEADLINE * 0.7:
        # Reconstruct adjacency from best_edges
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Build conflict graph
        non_edges, conflict, m = _build_conflict_graph(adj)
        
        if m > 0 and time.time() - start_time < DEADLINE * 0.8:
            # Use branch and bound with remaining time
            remaining_time = DEADLINE - (time.time() - start_time) - 5  # 5s safety margin
            if remaining_time > 10:
                edge_set = _branch_and_bound(non_edges, conflict, m, adj, remaining_time, best_count)
                
                if len(edge_set) > 0:
                    # Add these edges
                    new_adj = adj[:]
                    new_count = best_count
                    for idx in edge_set:
                        u, v = non_edges[idx]
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                        new_count += 1
                    
                    if new_count > best_count:
                        best_count = new_count
                        best_edges = []
                        for i in range(N):
                            for j in range(i+1, N):
                                if (new_adj[i] >> j) & 1:
                                    best_edges.append([i, j])
    
    # Phase 3: Local search polish
    if time.time() - start_time < DEADLINE - 5:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try to add more edges
        changed = True
        while changed and time.time() - start_time < DEADLINE - 5:
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    if not (adj[i] >> j) & 1:
                        if not _creates_c6(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            best_count += 1
                            changed = True
        
        # Update best_edges
        best_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if (adj[i] >> j) & 1:
                    best_edges.append([i, j])
    
    return best_edges
