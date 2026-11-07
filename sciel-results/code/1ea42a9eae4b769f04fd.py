import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 15
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initialize with K_{7,8}
    # Vertices 0-6 in A, 7-14 in B
    adj = [set() for _ in range(n)]
    for u in range(7):
        for v in range(7, 15):
            adj[u].add(v)
            adj[v].add(u)
            
    def get_degrees():
        return [len(adj[i]) for i in range(n)]
    
    def is_valid():
        deg = get_degrees()
        # Check all paths s-u-v-t
        # Iterate over all edges (u, v)
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # Each edge once
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
        
    def edge_count():
        return sum(len(adj[i]) for i in range(n)) // 2

    best_adj = [set(s) for s in adj]
    best_count = edge_count()
    
    # Local Search
    # 1. Greedy Add
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in all_pairs:
            if time.time() > deadline - 5:
                break
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                if is_valid():
                    improved = True
                    ec = edge_count()
                    if ec > best_count:
                        best_count = ec
                        best_adj = [set(s) for s in adj]
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
    # 2. Swaps (Remove 1, Add 1)
    # We need to be careful with time.
    # Number of edges ~ 56. Missing ~ 49.
    # 56 * 49 ~ 2700 swaps.
    # Each swap check is fast.
    
    # To escape local optima, we might need to accept worse moves?
    # Or just try all swaps.
    
    for _ in range(10): # Limit iterations of swap phase
        if time.time() > deadline - 10:
            break
        improved = False
        # Create list of current edges and missing edges
        current_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_edges.append((u, v))
        
        missing_edges = [p for p in all_pairs if p not in adj[p[0]]]
        
        for e1 in current_edges:
            if time.time() > deadline - 10:
                break
            u1, v1 = e1
            # Remove e1
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            
            for e2 in missing_edges:
                if time.time() > deadline - 10:
                    break
                u2, v2 = e2
                # Add e2
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                if is_valid():
                    ec = edge_count()
                    if ec > best_count:
                        best_count = ec
                        best_adj = [set(s) for s in adj]
                        improved = True
                        # If we improved, we might want to restart the swap loop or continue?
                        # Continue to find more improvements in this state.
                    else:
                        # Revert e2, keep e1 removed for now?
                        # Usually swap means 1-for-1. If not improved, revert.
                        adj[u2].discard(v2)
                        adj[v2].discard(u2)
                else:
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
            
            # Restore e1
            adj[u1].add(v1)
            adj[v1].add(u1)
            
            if improved:
                # If we found an improvement, the graph state has changed.
                # We should break out to recalculate edges/missing?
                # Or just break this inner loop and restart outer?
                break
            
            if time.time() > deadline - 10:
                break
            
        if not improved:
            # No improvement in this full pass of swaps
            # Try 2-swaps? Or break?
            # Let's try a few 2-swaps (remove 2, add 1)
            if time.time() < deadline - 20:
                improved2 = False
                ce = list(current_edges)
                for i in range(len(ce)):
                    if time.time() > deadline - 20:
                        break
                    e1 = ce[i]
                    for j in range(i+1, len(ce)):
                        if time.time() > deadline - 20:
                            break
                        e2 = ce[j]
                        
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        adj[u1].discard(v1); adj[v1].discard(u1)
                        adj[u2].discard(v2); adj[v2].discard(u2)
                        
                        for e3 in missing_edges:
                            if time.time() > deadline - 20:
                                break
                            u3, v3 = e3
                            adj[u3].add(v3); adj[v3].add(u3)
                            
                            if is_valid():
                                ec = edge_count()
                                if ec > best_count:
                                    best_count = ec
                                    best_adj = [set(s) for s in adj]
                                    improved2 = True
                                    break
                            else:
                                adj[u3].discard(v3); adj[v3].discard(u3)
                        
                        if improved2:
                            break
                        else:
                            # Restore e2
                            adj[u2].add(v2); adj[v2].add(u2)
                        
                        if improved2:
                            break
                        
                        # Restore e1
                        adj[u1].add(v1); adj[v1].add(u1)
                        
                        if improved2:
                            break
                            
                if improved2:
                    continue # Restart swap phase with new graph
            else:
                break

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
                
    return edges
