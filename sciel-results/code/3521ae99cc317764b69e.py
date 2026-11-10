import time
import random
import numpy as np
from collections import defaultdict

N = 31
M = 91  # number of edges

def count_c4(adj_list):
    """Count 4-cycles in a graph given adjacency list."""
    n = len(adj_list)
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        adj_set[i] = set(adj_list[i])
    
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj_set[u] & adj_set[v])
            if common >= 2:
                c4_count += common * (common - 1) // 2
    return c4_count

def polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5) which has 31 vertices and 90 edges.
    """
    points = []
    reps = {}
    idx = 0
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                if a != 0:
                    inv_a = pow(a, 4, 5)
                    rep = (1, b * inv_a % 5, c * inv_a % 5)
                elif b != 0:
                    inv_b = pow(b, 4, 5)
                    rep = (0, 1, c * inv_b % 5)
                else:
                    rep = (0, 0, 1)
                
                if rep not in reps:
                    reps[rep] = idx
                    points.append(rep)
                    idx += 1
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: P·Q = a*e + b*d + c*f = 0
    edges = []
    for i in range(31):
        for j in range(i + 1, 31):
            a, b, c = points[i]
            d, e, f = points[j]
            val = (a * e + b * d + c * f) % 5
            if val == 0:
                edges.append((i, j))
    
    return edges

def greedy_add_best_edge(edges, edge_set, adj, target_edges=91):
    """
    Given a starting set of edges, greedily add edges to reach target_edges,
    minimizing incremental C4 count at each step.
    """
    current_edges = list(edges)
    current_edge_set = set(edge_set)
    current_adj = [set() for _ in range(N)]
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
    
    while len(current_edges) < target_edges:
        best_edge = None
        best_new_c4s = float('inf')
        
        # Generate all non-edges
        all_non_edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) not in current_edge_set:
                    all_non_edges.append((u, v))
        
        if not all_non_edges:
            break
        
        # For each candidate edge, count how many C4s it would create
        for u, v in all_non_edges:
            new_c4s = 0
            for a in current_adj[v]:
                for b in current_adj[u]:
                    if a == b:
                        continue
                    pair = (min(a, b), max(a, b))
                    if pair in current_edge_set:
                        new_c4s += 1
            
            if new_c4s < best_new_c4s:
                best_new_c4s = new_c4s
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_edges.append((u, v))
        current_edge_set.add((u, v))
        current_adj[u].add(v)
        current_adj[v].add(u)
    
    return current_edges

def local_search_improve(edges, time_budget):
    """
    Improve edge set via local search: try swapping edges to reduce C4 count.
    Uses efficient C4 counting for individual edges.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_edges = list(edges)
    current_edge_set = set(current_edges)
    current_adj = [set() for _ in range(N)]
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
    
    current_c4s = count_c4([list(s) for s in current_adj])
    best_c4s = current_c4s
    best_edges = list(current_edges)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_edges_set = set(all_edges)
    
    # Helper to count C4s containing a specific edge
    def count_c4s_containing_edge(u, v, adj_set, edge_set):
        """Count C4s that contain edge (u,v)."""
        count = 0
        for a in adj_set[u]:
            if a == v:
                continue
            for b in adj_set[v]:
                if b == u or b == a:
                    continue
                pair = (min(a, b), max(a, b))
                if pair in edge_set:
                    count += 1
        return count
    
    while time.time() < deadline:
        made_progress = False
        random.shuffle(current_edges)  # Add randomness to avoid local minima
        
        for idx, (u1, v1) in enumerate(current_edges):
            if time.time() > deadline:
                break
            
            u1n = min(u1, v1)
            v1n = max(u1, v1)
            
            # Count C4s lost by removing (u1,v1)
            lost = count_c4s_containing_edge(u1, v1, current_adj, current_edge_set)
            
            # Try all non-edges
            for u2, v2 in all_edges:
                if (u2, v2) in current_edge_set:
                    continue
                
                # Count C4s gained by adding (u2,v2)
                gained = count_c4s_containing_edge(u2, v2, current_adj, current_edge_set)
                
                delta = gained - lost
                if delta < 0:
                    # Accept the swap
                    current_edges[idx] = (u2, v2)
                    current_edge_set.discard((u1n, v1n))
                    current_edge_set.add((u2, v2))
                    current_adj[u1].discard(v1)
                    current_adj[v1].discard(u1)
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                    current_c4s += delta
                    made_progress = True
                    
                    if current_c4s < best_c4s:
                        best_c4s = current_c4s
                        best_edges = list(current_edges)
                    
                    break  # Restart scan after accepting a move
        
        if not made_progress:
            break
    
    return best_edges

def construct():
    """
    Main entry point: construct a graph with 91 edges on 31 vertices minimizing C4 count.
    Strategy: Start with polarity graph of PG(2,5) (90 edges, very few C4s),
    then greedily add the best edge, then do local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Step 1: Construct polarity graph
    pol_edges = polarity_graph_q5()
    
    # Verify count
    pol_count = count_c4([list(set(s)) for s in [set(v for v in range(N) if (u,v) in pol_edges or (v,u) in pol_edges) for u in range(N)]])
    
    # Step 2: If we have exactly 90 edges, greedily add 1 more
    if len(pol_edges) == 90:
        edge_set = set(pol_edges)
        adj = [set() for _ in range(N)]
        for u, v in pol_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Greedily add the edge that minimizes new C4s
        best_edge = None
        best_new = float('inf')
        
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in edge_set:
                    continue
                new_c4s = 0
                for a in adj[v]:
                    for b in adj[u]:
                        if a == b:
                            continue
                        pair = (min(a, b), max(a, b))
                        if pair in edge_set:
                            new_c4s += 1
                if new_c4s < best_new:
                    best_new = new_c4s
                    best_edge = (u, v)
        
        if best_edge:
            final_edges = list(pol_edges) + [best_edge]
        else:
            final_edges = list(pol_edges)
            # Add any edge
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) not in edge_set:
                        final_edges.append((u, v))
                        break
                if len(final_edges) > len(pol_edges):
                    break
    elif len(pol_edges) > 90:
        # Too many edges - remove some
        final_edges = list(pol_edges[:90])
    else:
        # Too few edges - add more
        final_edges = list(pol_edges)
        edge_set = set(final_edges)
        for u in range(N):
            for v in range(u + 1, N):
                if len(final_edges) >= 91:
                    break
                if (u, v) not in edge_set:
                    final_edges.append((u, v))
                    edge_set.add((u, v))
            if len(final_edges) >= 91:
                break
    
    # Ensure exactly 91 edges
    if len(final_edges) != 91:
        edge_set = set(final_edges)
        if len(final_edges) < 91:
            for u in range(N):
                for v in range(u + 1, N):
                    if len(final_edges) >= 91:
                        break
                    if (u, v) not in edge_set:
                        final_edges.append((u, v))
                        edge_set.add((u, v))
                if len(final_edges) >= 91:
                    break
        else:
            final_edges = final_edges[:91]
    
    # Step 3: Local search to improve
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        improved_edges = local_search_improve(final_edges, remaining_time)
        if improved_edges and len(improved_edges) == 91:
            final_edges = improved_edges
    
    # Convert to required format: list of [u, v]
    result = [[u, v] for u, v in final_edges]
    
    # Final validation
    if len(result) != 91:
        # Fallback: simple construction
        result = []
        edge_set = set()
        for u in range(N):
            for v in range(u + 1, N):
                if len(result) >= 91:
                    break
                result.append([u, v])
                edge_set.add((u, v))
            if len(result) >= 91:
                break
    
    return result
