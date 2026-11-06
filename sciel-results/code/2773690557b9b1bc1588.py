import itertools
import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Generate all C4 faces
    # A face is defined by a vertex x and two distinct dimensions i, j.
    # The edges are (x, x^1<<i), (x^1<<i, x^1<<i^1<<j), (x^1<<i^1<<j, x^1<<j), (x^1<<j, x)
    # We need to identify which hypercube edges conflict (share a face).
    
    # Instead of generating faces and then conflicts, we can generate conflicts directly.
    # Two edges e1=(u1,v1) and e2=(u2,v2) conflict if they form a path of length 2 in a C4?
    # No, they conflict if they are in the same C4.
    # In Q7, two edges are in a C4 if:
    # 1. They share a vertex and are in different dimensions (they form a corner of a C4).
    # 2. They are disjoint and "parallel" in a specific way?
    #    If e1 is in dim i and e2 is in dim j (i!=j).
    #    e1 = (u, u^1<<i). e2 = (v, v^1<<j).
    #    They form a C4 if {u, u^1<<i, v, v^1<<j} are the 4 vertices of a C4.
    #    This happens if v = u^1<<j and v^1<<j = u^1<<i (i.e., v = u^1<<j and v^1<<j = u^1<<i => u^1<<j^1<<j = u^1<<i => u = u^1<<i, impossible) OR
    #    v = u^1<<i^1<<j and v^1<<j = u^1<<i?
    #    Let's check the vertices of the C4 spanned by u and dims i,j:
    #    u, u^1<<i, u^1<<j, u^1<<i^1<<j.
    #    Edges:
    #    e_i1 = (u, u^1<<i)
    #    e_j1 = (u, u^1<<j)
    #    e_i2 = (u^1<<j, u^1<<i^1<<j)
    #    e_j2 = (u^1<<i, u^1<<i^1<<j)
    #    
    #    So, for any vertex u and dims i, j:
    #    The 4 edges are:
    #    e1 = (min(u, u^1<<i), max(u, u^1<<i))
    #    e2 = (min(u, u^1<<j), max(u, u^1<<j))
    #    e3 = (min(u^1<<j, u^1<<i^1<<j), max(u^1<<j, u^1<<i^1<<j))
    #    e4 = (min(u^1<<i, u^1<<i^1<<j), max(u^1<<i, u^1<<i^1<<j))
    #    
    #    All pairs of these 4 edges conflict.
    
    conflict_graph = [set() for _ in range(n_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v_i = u ^ (1 << i)
                v_j = u ^ (1 << j)
                v_ij = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v_i), max(u, v_i))
                e2 = (min(u, v_j), max(u, v_j))
                e3 = (min(v_j, v_ij), max(v_j, v_ij))
                e4 = (min(v_i, v_ij), max(v_i, v_ij))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                edges_in_face = [idx1, idx2, idx3, idx4]
                for k in range(4):
                    for l in range(k + 1, 4):
                        conflict_graph[edges_in_face[k]].add(edges_in_face[l])
                        conflict_graph[edges_in_face[l]].add(edges_in_face[k])

    # 3. Greedy MIS
    # State: available nodes, selected nodes
    available = [True] * n_edges
    selected = [False] * n_edges
    
    # Calculate degrees
    degrees = [len(conflict_graph[i]) for i in range(n_edges)]
    
    # To speed up greedy, we can use a priority queue or just scan.
    # Scanning 448 nodes 448 times is ~200k ops, very fast.
    
    while True:
        min_deg = float('inf')
        best_node = -1
        for i in range(n_edges):
            if available[i]:
                # Recalculate degree in available subgraph? 
                # For simplicity and speed, using original degree is a common heuristic (Minimum Degree Heuristic).
                # It is often better to use current degree.
                # Let's use current degree for better quality.
                curr_deg = 0
                for nb in conflict_graph[i]:
                    if available[nb]:
                        curr_deg += 1
                
                if curr_deg < min_deg:
                    min_deg = curr_deg
                    best_node = i
                    
        if best_node == -1:
            break
            
        selected[best_node] = True
        available[best_node] = False
        for nb in conflict_graph[best_node]:
            available[nb] = False
            
    # 4. Local Search
    # We want to improve the independent set 'selected'.
    # selected is a boolean array.
    # U = nodes not in selected and not "blocked"? 
    # Actually, any node not in selected is a candidate.
    # But we must ensure independence.
    
    # Helper: check if node i can be added to selected
    def can_add(i):
        if selected[i]:
            return False
        for nb in conflict_graph[i]:
            if selected[nb]:
                return False
        return True
        
    # Helper: check if node i is in selected
    # We maintain a set of selected nodes for fast lookup? 
    # selected is a list of bools. 
    # To check neighbors in selected, we iterate conflict_graph[i].
    
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Add any node that can be added
        for i in range(n_edges):
            if not selected[i]:
                if can_add(i):
                    selected[i] = True
                    improved = True
                    
        # Phase 2: Try swaps
        # If no node could be added, try swapping.
        # A swap consists of removing s from selected and adding u to selected.
        # This is valid if u is not in selected, and N(u) \cap selected \subseteq {s}.
        # i.e., the only neighbor of u in selected is s.
        
        # To make this efficient, we can iterate over s in selected, and u in neighbors of s (not in selected).
        
        swaps_made = 0
        for s in range(n_edges):
            if selected[s]:
                # Try to find u such that swapping s->u is valid and potentially beneficial.
                # Valid: N(u) \cap selected == {s}
                # Beneficial: We hope that after swapping, we can add more nodes.
                # Since size stays same, we need a second pass to add.
                
                # Optimization: Only try if degree of u is low?
                for u in conflict_graph[s]:
                    if selected[u]:
                        continue
                    
                    # Check if N(u) \cap selected is subset of {s}
                    valid_swap = True
                    for nb in conflict_graph[u]:
                        if selected[nb] and nb != s:
                            valid_swap = False
                            break
                            
                    if valid_swap:
                        # Perform swap
                        selected[s] = False
                        selected[u] = True
                        swaps_made += 1
                        improved = True
                        # Break out to re-evaluate "add" phase immediately?
                        # Yes, because the swap might open up space for other additions.
                        break
                if improved:
                    break # Restart the while loop to do Phase 1 again
                    
    # Construct result
    result = []
    for e_idx in range(n_edges):
        if selected[e_idx]:
            u, v = idx_to_edge[e_idx]
            result.append([u, v])
            
    return result
