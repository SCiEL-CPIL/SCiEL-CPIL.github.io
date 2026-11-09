import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145  # 145s limit, 5s safety margin
    n = 20
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def is_valid_add(u, v):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Assumes (u, v) is NOT currently in the graph.
        """
        # 1. Check triples (u, v, w)
        # Common neighbors of u and v: C
        C = adj[u] & adj[v]
        for w in range(n):
            if w == u or w == v:
                continue
            # Common neighbors of u, v, w is C & adj[w]
            if (C & adj[w]).bit_count() >= 3:
                return False
        
        # 2. Check triples (u, x, y) where v is a new common neighbor
        # v becomes a common neighbor of u, x, y if v in N(x) and v in N(y)
        # In that case, the number of common neighbors increases by 1.
        # So we need |N(u) ∩ N(x) ∩ N(y)| <= 1 if v is in N(x) ∩ N(y)
        # Note: N(u) here is the OLD N(u).
        for x in range(n):
            if x == u or x == v:
                continue
            if not (adj[x] & (1 << v)):
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if not (adj[y] & (1 << v)):
                    continue
                # v is in N(x) and N(y)
                # Check if adding v to N(u) makes |N(u) ∩ N(x) ∩ N(y)| >= 3
                # Current common neighbors: adj[u] & adj[x] & adj[y]
                common = (adj[u] & adj[x] & adj[y]).bit_count()
                if common >= 2:
                    return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    best_adj = [0] * n
    best_edges = 0

    # Multiple restarts to escape local optima
    # We have ~145 seconds.
    # Each restart:
    # 1. Randomly initialize? Or start empty?
    #    Starting empty and greedily adding is fast.
    #    Let's try several random orderings.
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline - 10:
        # Restart phase
        current_adj = [0] * n
        # Use a different adj list for this trial? 
        # Or just reset the global one.
        adj = current_adj[:]
        
        # 1. Greedy addition
        missing = all_edges[:]
        random.shuffle(missing)
        
        # We need to track existing edges to facilitate swaps later
        # But for the initial greedy phase, we just try to add.
        
        # To make greedy smarter, we can prioritize edges?
        # Random is often fine with multiple restarts.
        
        for u, v in missing:
            if time.time() > deadline - 15:
                break
            if is_valid_add(u, v):
                add_edge(u, v)
        
        current_edges = count_edges()
        
        # 2. Local Search (Hill Climbing with Swaps)
        # Build lists of existing and missing edges
        existing = []
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    existing.append((i, j))
                else:
                    missing.append((i, j))
        
        # Iterate: try to add missing edges. If stuck, swap.
        progress = True
        # Limit iterations to avoid infinite loops if no progress
        max_iter = 100
        iter_count = 0
        
        while progress and time.time() < deadline - 15 and iter_count < max_iter:
            iter_count += 1
            progress = False
            
            # Try adding edges
            random.shuffle(missing)
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 15:
                    new_missing.append((u, v))
                    continue
                if is_valid_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    progress = True
                    # Move to existing
                    existing.append((u, v))
                else:
                    new_missing.append((u, v))
            
            if progress:
                missing = new_missing
                continue
            
            # If no additions, try swaps
            if existing and missing:
                # Try a few random swaps
                swaps_tried = 0
                # Limit swaps per iteration to keep time bounded
                while swaps_tried < 50 and time.time() < deadline - 15:
                    swaps_tried += 1
                    e1 = random.choice(existing)
                    e2 = random.choice(missing)
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    # Remove e1
                    remove_edge(u1, v1)
                    # Check if we can add e2
                    if is_valid_add(u2, v2):
                        add_edge(u2, v2)
                        # Update lists
                        existing.remove(e1)
                        existing.append(e2)
                        missing.remove(e2)
                        missing.append(e1)
                        progress = True
                    else:
                        # Revert
                        add_edge(u1, v1)
            
            # If no progress after swapping, break to restart
            if not progress:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
        
        # Optional: Shake?
        # If we are stuck, we can remove a random edge and try to fill.
        # But restart is simpler.
        
    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result
