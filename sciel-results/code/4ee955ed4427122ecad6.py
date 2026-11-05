import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin

    n = 20
    num_colours = 20

    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and one of 3 orderings.
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    # Total: C(20,4) * 3 = 14535 cycles.
    
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Precompute for each edge (u,v) the list of cycles that contain it.
    edge_to_cycles = {}
    for i, cycle in enumerate(cycles):
        for e in cycle:
            u, v = e
            if u > v:
                u, v = v, u
            key = (u, v)
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(i)

    # Strategy:
    # 1. Start with K_{10,10} (vertices 0-9 and 10-19).
    #    This graph has 100 edges and no 4-cycles.
    #    We color these edges to maximize flexibility.
    #    A good coloring for bipartite edges might be random or structured.
    #    Let's use a proper edge coloring of K_{10,10} if possible, or just random distinct colors per vertex.
    #    Since K_{10,10} is 10-regular, we can color it with 10 colors properly.
    #    But we have 20 colors.
    
    # 2. Try to add the 45 edges within {0..9} and 45 edges within {10..19}.
    #    Total potential edges: 100 + 45 + 45 = 190 (K_20).
    #    We will iterate through these 90 candidate edges in random order.
    #    For each, try to find a valid color.
    #    If no valid color, try to recolor existing edges to free up a color.
    
    def is_valid_full(adj):
        """
        Checks if the current graph is valid.
        """
        for cycle in cycles:
            e1, e2, e3, e4 = cycle
            c1 = adj[e1[0], e1[1]]
            c2 = adj[e2[0], e2[1]]
            c3 = adj[e3[0], e3[1]]
            c4 = adj[e4[0], e4[1]]
            
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                continue
            
            if len({c1, c2, c3, c4}) < 4:
                return False
        return True

    def get_forbidden_colors(adj, u, v):
        """
        Returns the set of colors forbidden for edge (u,v) due to 4-cycle constraints
        and proper coloring (if we enforce it, though the problem doesn't strictly require 
        adjacent edges to be distinct UNLESS they are part of a 4-cycle.
        However, if (u,v) and (u,w) have the same color, and there is a path v-x-w-u, 
        then the cycle v-x-w-u-v has edges (v,x), (x,w), (w,u), (u,v).
        If (w,u) and (u,v) have the same color, the cycle is NOT rainbow.
        So, effectively, if the graph is dense, adjacent edges MUST have distinct colors 
        to avoid creating non-rainbow 4-cycles via short paths.
        So we should enforce proper edge coloring as a heuristic constraint.
        """
        forbidden = set()
        
        # 1. Proper coloring constraint (heuristic but likely necessary for density)
        for w in range(n):
            if w != u and adj[u, w] != -1:
                forbidden.add(adj[u, w])
            if w != v and adj[w, v] != -1:
                forbidden.add(adj[w, v])
                
        # 2. 4-cycle rainbow constraint
        key = (min(u,v), max(u,v))
        if key in edge_to_cycles:
            for idx in edge_to_cycles[key]:
                cycle = cycles[idx]
                other_edges = []
                for e in cycle:
                    eu, ev = e
                    if (eu == u and ev == v) or (eu == v and ev == u):
                        continue
                    other_edges.append(e)
                
                valid_cycle = True
                colours = []
                for e in other_edges:
                    c = adj[e[0], e[1]]
                    if c == -1:
                        valid_cycle = False
                        break
                    colours.append(c)
                
                if not valid_cycle:
                    continue
                
                # If the other 3 edges don't have distinct colors, 
                # adding (u,v) won't fix the cycle's rainbow property 
                # (it's already broken or will be broken regardless of (u,v)'s color 
                # if the other 3 have a duplicate).
                # Wait, if the other 3 have a duplicate, the cycle is NOT rainbow 
                # no matter what color (u,v) gets. 
                # So if such a cycle exists in the graph, the graph is ALREADY INVALID.
                # But we maintain validity invariant. 
                # So if we reach this state, the other 3 MUST be distinct.
                
                # The new color must be distinct from these 3.
                forbidden.update(colours)
                
        return forbidden

    def try_add_edge(adj, u, v):
        """
        Attempts to add edge (u,v) with a valid color.
        Returns True if added, False otherwise.
        """
        forbidden = get_forbidden_colors(adj, u, v)
        
        # Find a color not in forbidden
        for c in range(num_colours):
            if c not in forbidden:
                adj[u, v] = c
                adj[v, u] = c
                return True
        return False

    def recolor_to_add(adj, u, v):
        """
        If try_add_edge fails, try to recolor some existing edges to free up a color.
        This is a simplified local search step.
        """
        # This is complex to implement robustly in short time.
        # For now, we rely on the order of addition.
        return False

    best_edges = 0
    best_adj = None
    
    # Restart loop
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Initialize adj with -1
        adj = np.full((n, n), -1, dtype=int)
        
        # Step 1: Add K_{10,10} edges
        # Vertices 0-9 and 10-19
        # We can color these properly.
        # A simple way: color (u, v) with (u % 10) + (v % 10) % 10? 
        # Let's just use random distinct colors for now, or a structured one.
        # Since there are no 4-cycles in K_{10,10}, any coloring is valid.
        # But we want to leave room for intra-partition edges.
        
        # Let's use a random proper coloring for the bipartite part.
        # Or just random.
        for u in range(10):
            for v in range(10, 20):
                # Assign random color
                c = random.randint(0, num_colours - 1)
                adj[u, v] = c
                adj[v, u] = c
        
        # Step 2: Add intra-partition edges
        # Candidate edges: within 0-9 and within 10-19
        candidates = []
        for u in range(10):
            for v in range(u + 1, 10):
                candidates.append((u, v))
        for u in range(10, 20):
            for v in range(u + 1, 20):
                candidates.append((u, v))
                
        random.shuffle(candidates)
        
        # Greedy addition
        for u, v in candidates:
            if try_add_edge(adj, u, v):
                pass
            else:
                # Try a few times to recolor? 
                # For simplicity in this iteration, we skip.
                # Future improvement: implement backtracking or recoloring.
                pass
        
        # Count edges
        edge_count = int(np.sum(adj != -1)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
            
        # If we hit 190, we are done
        if best_edges == 190:
            break

    # Fallback if best_adj is None
    if best_adj is None:
        adj = np.full((n, n), -1, dtype=int)
        for u in range(10):
            for v in range(10, 20):
                adj[u, v] = 0
                adj[v, u] = 0
        best_adj = adj
        best_edges = 100

    # Final verification
    valid = True
    for cycle in cycles:
        e1, e2, e3, e4 = cycle
        c1 = best_adj[e1[0], e1[1]]
        c2 = best_adj[e2[0], e2[1]]
        c3 = best_adj[e3[0], e3[1]]
        c4 = best_adj[e4[0], e4[1]]
        
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            continue
            
        if len({c1, c2, c3, c4}) < 4:
            valid = False
            break
            
    if not valid:
        # Return a known valid graph: K_{10,10} with all edges color 0
        # Wait, if all edges are color 0, and there are no 4-cycles, it is valid.
        adj = np.full((n, n), -1, dtype=int)
        for u in range(10):
            for v in range(10, 20):
                adj[u, v] = 0
                adj[v, u] = 0
        best_adj = adj
        best_edges = 100

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
