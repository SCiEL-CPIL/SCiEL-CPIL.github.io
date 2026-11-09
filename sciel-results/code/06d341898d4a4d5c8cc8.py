import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_pos = {}
    for idx, (u, v) in enumerate(P['edge_order'], 1):
        if u > v:
            u, v = v, u
        edge_pos[(u, v)] = idx

    def get_pos(u, v):
        if u > v:
            u, v = v, u
        return edge_pos.get((u, v), 10000)

    def calc_cost(perm):
        max_pos = 0
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            p = get_pos(u, v)
            if p > max_pos:
                max_pos = p
        return max_pos

    def greedy_construct(rng):
        # State: degree of each node, list of edges in current path (as a cycle eventually)
        # We build a path, then close it.
        # To ensure it's a single cycle, we track connectivity or just rely on local search to fix.
        # A simple greedy: pick the smallest edge that doesn't create a degree > 2.
        # This might create multiple cycles. We will join them in local search.
        
        degrees = [0] * n
        edges = []
        used_edges = set()
        
        # Sort all edges by position
        all_edges = sorted(P['edge_order'], key=lambda e: edge_pos[(min(e[0], e[1]), max(e[0], e[1]))])
        
        for u, v in all_edges:
            if degrees[u] < 2 and degrees[v] < 2:
                # Check if adding this edge creates a small cycle prematurely?
                # For greedy, we just add it.
                # Actually, we want to avoid closing the cycle until the end.
                # Let's just build a 2-regular graph (collection of cycles).
                edges.append((u, v))
                degrees[u] += 1
                degrees[v] += 1
                used_edges.add((min(u, v), max(u, v)))
                if len(edges) == n:
                    break
        
        if len(edges) < n:
            # Fallback: random permutation
            perm = list(range(n))
            rng.shuffle(perm)
            return perm

        # We have a set of edges forming disjoint cycles.
        # We need to order them into a single Hamilton cycle.
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find cycles
        visited = [False] * n
        cycles = []
        for i in range(n):
            if not visited[i]:
                cycle = []
                curr = i
                prev = -1
                while not visited[curr]:
                    visited[curr] = True
                    cycle.append(curr)
                    nexts = adj[curr]
                    # Pick the next unvisited neighbor
                    next_node = -1
                    for nb in nexts:
                        if not visited[nb] or nb == prev:
                            # This logic is tricky for simple cycles.
                            # Since it's a 2-regular graph, we can just traverse.
                            pass
                    
                    # Simpler traversal for 2-regular graph
                    if len(cycle) == 1:
                         # Start
                         next_node = adj[curr][0]
                    else:
                         # Move to the other neighbor
                         next_node = adj[curr][0] if adj[curr][1] == prev else adj[curr][1]
                    
                    if next_node == prev and len(cycle) > 1:
                         break # Closed cycle
                    
                    prev = curr
                    curr = next_node
                cycles.append(cycle)
        
        # Now we have a list of cycles. We need to merge them.
        # If there's only one cycle, we are done.
        if len(cycles) == 1:
            return cycles[0]
        
        # Merge cycles by swapping edges.
        # This is complex to implement robustly in a few lines.
        # Instead, let's just return a permutation that visits all nodes, 
        # and let the local search (Or-opt) fix the structure.
        # A simple permutation: concatenate the cycles.
        # But this creates a path, not a cycle, and the "implicit closing edge" might be bad.
        # Let's just return the first cycle if it's length n, otherwise a random perm.
        for c in cycles:
            if len(c) == n:
                return c
        
        # If no single cycle, return random permutation to let local search work.
        perm = list(range(n))
        rng.shuffle(perm)
        return perm

    rng = random.Random(614)
    
    # Initial solution: Greedy
    current_perm = greedy_construct(rng)
    current_cost = calc_cost(current_perm)
    best_perm = current_perm[:]
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 55.0
    
    # Local Search with Or-opt moves
    # Or-opt: remove a segment of length 1, 2, or 3 and insert it elsewhere.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try random Or-opt moves
        for _ in range(100):
            if time.time() - start_time >= time_limit:
                break
                
            # Choose segment length
            seg_len = rng.randint(1, 3)
            if seg_len >= n:
                continue
                
            # Choose start index of segment
            i = rng.randint(0, n - 1)
            
            # Extract segment
            # Note: perm is a cycle. Removing a segment from the cycle breaks it into a path.
            # Inserting it back closes it.
            
            # To handle cycle permutation, we can "cut" the cycle at a random point to make it a linear list,
            # perform the move, and then it's still a cycle.
            # Or just work with the list representation.
            # List: [v0, v1, ..., vn-1]. Edges: (v0,v1), ..., (vn-2,vn-1), (vn-1,v0).
            
            # Remove segment [i, i+seg_len-1]
            # Be careful with indices wrapping around.
            # Let's linearize by rotating so i is at 0.
            # Actually, simpler: just manipulate the list.
            
            # Get the segment
            # If the segment wraps around the end, it's complicated.
            # Let's just pick i such that i + seg_len <= n for simplicity, or handle wrap.
            # For n=24, wrap is rare if we pick i in [0, n-seg_len].
            
            if i + seg_len > n:
                continue # Skip wrap-around for simplicity in this iteration, or handle it.
                
            segment = current_perm[i:i+seg_len]
            
            # Remove segment
            new_perm = current_perm[:i] + current_perm[i+seg_len:]
            
            # Find insertion point
            # new_perm has length n - seg_len.
            # Insert into new_perm at position k.
            # The resulting list will have length n.
            
            k = rng.randint(0, len(new_perm))
            
            # Insert
            candidate = new_perm[:k] + segment + new_perm[k:]
            
            # The cycle is implicitly closed by the list ends.
            # Wait, the original cycle had an edge between current_perm[i-1] and current_perm[i]
            # and between current_perm[i+seg_len-1] and current_perm[i+seg_len].
            # After removal, new_perm connects new_perm[i-1] and new_perm[i+seg_len] (which is old i+seg_len).
            # So the edge (old i-1, old i+seg_len) is added.
            # The segment is inserted between new_perm[k-1] and new_perm[k].
            # So edges (new_perm[k-1], segment[0]) and (segment[-1], new_perm[k]) are added.
            # And the internal edges of the segment are preserved.
            # This is a valid 2-opt/or-opt move on the cycle.
            
            cand_cost = calc_cost(candidate)
            
            if cand_cost < current_cost:
                current_perm = candidate
                current_cost = cand_cost
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_perm = current_perm[:]
                    
        # If no improvement in last batch, maybe random restart or just continue
        # To avoid getting stuck, we can occasionally perturb.
        if not improved and rng.random() < 0.05:
             # Random restart
             current_perm = list(range(n))
             rng.shuffle(current_perm)
             current_cost = calc_cost(current_perm)
             if current_cost < best_cost:
                 best_cost = current_cost
                 best_perm = current_perm[:]

    return best_perm
# EVOLVE-BLOCK-END
