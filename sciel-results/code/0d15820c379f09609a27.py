import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    # Initialize with 8 disjoint K5s
    adj = [0] * n
    edges = []
    
    for k in range(8):
        base = 5 * k
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    
    def creates_c6(u, v):
        # Check if there is a simple path of length 5 from u to v
        # Using BFS with bitmasks
        # We want to see if v is reachable from u in exactly 5 steps
        # Note: Since we are adding (u,v), the path must not use (u,v).
        # The path is in the current graph G.
        
        # Optimization: If u and v are not connected at all, no path.
        # But we just check reachability in 5 steps.
        
        current = 1 << u
        visited = 0
        # We traverse 5 steps
        # Step 1
        for step in range(5):
            next_frontier = 0
            # Iterate through nodes in current frontier
            # Since n=40, we can just loop through bits or use a helper
            # For speed, we can precompute neighbor masks? No, adj is already masks.
            # We need OR of adj[x] for all x in current.
            
            # Unrolling the bit iteration for n=40 is possible but Python loop is okay
            # if we do it carefully.
            
            # Efficient way to get union of neighbors of a set of nodes:
            # This is essentially matrix-vector multiplication with boolean algebra.
            # For n=40, we can just loop through the set bits.
            
            c = current
            while c:
                # Get lowest set bit
                lsb = c & -c
                node_idx = lsb.bit_length() - 1
                next_frontier |= adj[node_idx]
                c -= lsb
            
            # Remove visited nodes to ensure simple path
            next_frontier &= ~visited
            # Also remove u? No, u is in visited after step 0? 
            # Let's initialize visited with u.
            
            if step == 0:
                visited = current # This is u
            else:
                visited |= next_frontier
            
            # Check if v is in next_frontier
            if next_frontier & (1 << v):
                return True
            
            current = next_frontier
            if current == 0:
                return False
                
        return False

    # Greedy Phase
    # Try to add all possible edges
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                if not creates_c6(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    
    # Local Search Phase
    # We will try to swap edges to increase density.
    # Strategy: Randomly remove an edge, then try to add any missing edge that fits.
    # If we can add more than we removed, keep it.
    
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.discard((min(u, v), max(u, v)))
        
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
        
    best_edges = len(edges)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges) - 1)
        u_rem, v_rem = edges[idx]
        
        # Save state to revert if necessary
        # We remove one edge
        remove_edge(u_rem, v_rem)
        
        # Try to add missing edges
        added = 0
        added_edges = []
        
        # We can iterate through all missing edges. 
        # To make it faster, we can prioritize edges that were previously "close" to being added.
        # But for now, just iterate all missing pairs.
        # n=40, ~780 pairs. 780 * creates_c6 cost.
        # creates_c6 is fast (~50-100 ops).
        # 780 * 100 = 78k ops per restart.
        # 110s / (time per restart) -> many restarts.
        
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if not creates_c6(u, v):
                        add_edge(u, v)
                        added_edges.append([u, v])
                        added += 1
                        if time.time() > deadline:
                            break
            if time.time() > deadline:
                break
        
        # If we added more edges than we removed (which is 1), keep it.
        # Actually, we removed 1. If added > 1, net gain.
        # If added == 1, net neutral, but graph structure changed. Might help future steps.
        # If added < 1, net loss.
        
        if added <= 1:
            # Revert: Re-add the removed edge, remove the added edges
            # Note: The added edges might have created new C6s with each other?
            # No, we checked each against the current graph.
            # But if we added (a,b) and then (c,d), (c,d) was checked against graph with (a,b).
            # So the set of added edges is valid together.
            # To revert, we just remove them and add back the removed one.
            
            for e in added_edges:
                remove_edge(e[0], e[1])
            add_edge(u_rem, v_rem)
        else:
            # Keep the changes
            edges = list(edge_set)
            # Update edges list to match edge_set
            # We need to be careful to maintain the list structure for random selection
            # Rebuilding list from set is O(E)
            edges = [[min(e), max(e)] for e in edge_set]
            if len(edges) > best_edges:
                best_edges = len(edges)

    return edges

# EVOLVE-BLOCK-START
# The function construct is defined above.
# We need to wrap it in the required block format?
# The prompt says: "Keep the program's entry-point function name(s) ... Output the COMPLETE program in a single python code block."
# And "Keep the function names and signatures of the seed program."
# The seed program had:
# def construct():
#     ...
#     return edges
# 
# I will place the code inside the EVOLVE-BLOCK as requested by the "CURRENT PROGRAM" structure, 
# although the instruction "Output the COMPLETE program" usually implies the whole file.
# I will provide the whole file content.
# EVOLVE-BLOCK-END
