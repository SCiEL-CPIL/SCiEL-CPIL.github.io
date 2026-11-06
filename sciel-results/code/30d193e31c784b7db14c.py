import itertools
import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 16
    
    # Helper: Check if graph (list of edges) is C4-free
    # Using bitsets for speed
    def is_c4_free(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Count C4s
        # A C4 is formed by two vertices u, v sharing two neighbors.
        # Sum over all pairs u, v of C(common_neighbors, 2)
        # If sum > 0, there is a C4.
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                # Count bits in common
                # If count >= 2, it contributes to C4 count
                # Actually, each pair of common neighbors forms a C4 with u and v.
                # So we just need to know if common has >= 2 bits.
                # For efficiency, we can just return False if found.
                if common & (common - 1): # Has at least 2 bits
                    return False
        return True

    def edge_set_to_list(edge_set):
        return [[u, v] for u, v in edge_set]

    best_edges = 0
    best_graph = set()
    
    # Initial construction: Random dense graph
    # Try multiple random restarts
    max_restarts = 50
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Start with a random graph with p=0.5
        current_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    current_edges.add((u, v))
        
        # Phase 1: Remove edges to make it C4-free
        # Greedy removal: Find a C4, remove a random edge from it.
        # To find a C4 quickly, we can iterate pairs.
        
        def find_c4(edges_set):
            adj = [0] * n
            for u, v in edges_set:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if common & (common - 1):
                        # Find two common neighbors w, x
                        # w and x are in common
                        # The C4 is u-w-v-x-u
                        # Edges: (u,w), (w,v), (v,x), (x,u)
                        # We need to return one of these edges to remove
                        # Let's get the bits
                        bits = []
                        mask = common
                        while mask:
                            bit = mask & -mask
                            bits.append(bit.bit_length() - 1)
                            mask ^= bit
                        if len(bits) >= 2:
                            w = bits[0]
                            x = bits[1]
                            # Return a random edge from the cycle
                            cycle_edges = [(u, w), (w, v), (v, x), (x, u)]
                            # Normalize tuples
                            cycle_edges_norm = [tuple(sorted(e)) for e in cycle_edges]
                            return random.choice(cycle_edges_norm)
            return None

        while True:
            if time.time() > deadline:
                break
            c4_edge = find_c4(current_edges)
            if c4_edge is None:
                break
            current_edges.remove(c4_edge)
        
        # Phase 2: Try to add edges back
        # Iterate through all non-edges
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            # Shuffle non-edges for randomness
            all_pairs = list(itertools.combinations(range(n), 2))
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if (u, v) in current_edges:
                    continue
                # Try adding
                new_edges = current_edges | {(u, v)}
                if is_c4_free(new_edges):
                    current_edges = new_edges
                    added_any = True
            
            # If we added any, we might be able to add more, so loop again
            # But this is O(E^2) in worst case. 
            # Optimization: Only check if the new edge creates a C4.
            # A new edge (u,v) creates a C4 if there is a path u-w-v-x-u? No.
            # It creates a C4 if there exists w, x such that u-w-v-x-u is a cycle.
            # i.e., w is neighbor of u and v, and x is neighbor of u and v?
            # No, adding (u,v) creates a C4 if there is a path of length 3 between u and v.
            # Path u-w-x-v. Then u-w-x-v-u is a C4.
            # So we need to check if dist(u,v) == 3 in G without (u,v).
            
            # Let's stick to the full check for simplicity, n=16 is small.
            # But we can optimize:
            # Only check if u and v have a common neighbor? No, that's length 2.
            # We need length 3 path.
            # Let's just use the full is_c4_free check. It's fast enough.
        
        # Phase 3: Local Search (Swaps)
        # Try to swap an existing edge with a non-edge to increase density?
        # No, we want to maximize edges.
        # We can try to remove an edge e1 and add two edges e2, e3?
        # That's a "2-opt" move.
        
        # For now, let's just record the best
        if len(current_edges) > best_edges:
            best_edges = len(current_edges)
            best_graph = set(current_edges)
            
        # If we found a very good one, we can stop early?
        # Theoretical max is around 40-50.
        # If we hit 50, we are likely done.
        if best_edges >= 50:
            break
            
    # Convert best_graph to list of lists
    return [[u, v] for u, v in best_graph]
