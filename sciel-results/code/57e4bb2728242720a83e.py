import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def try_construct(n, max_restarts=100):
        min_deg_target = n - 39
        # Start with a degree slightly higher to allow removal
        # If n=47, target=8. Start with 10 or 12.
        start_deg = min_deg_target + 4
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Generate random graph
            # We want a random graph with degrees close to start_deg
            # Simple method: random permutation of edges? No, just random edges.
            # Better: Start with empty, add random edges until degrees are met?
            # Or: Use a random regular graph generation algorithm.
            # For small n, we can just pick random edges and check degrees.
            
            adj = [set() for _ in range(n)]
            edges = []
            
            # Randomly add edges
            # We want roughly start_deg per vertex. Total edges ~ n * start_deg / 2
            num_edges_target = int(n * start_deg / 2)
            
            # Shuffle all possible edges
            all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_possible)
            
            # Add edges greedily until we have enough or no more
            # This might not give regular degrees, but that's okay.
            # We just need min_deg >= target at the end.
            # Starting high ensures we have room to remove.
            
            count = 0
            for u, v in all_possible:
                if count >= num_edges_target:
                    break
                # Add edge if it doesn't already exist (it won't, we are iterating unique pairs)
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                count += 1
            
            # Check if we can even meet the target degree with this many edges?
            # If max possible degree < target, fail.
            # But we picked enough edges.
            
            # Compute common counts
            # common[u][v] = len(adj[u] & adj[v])
            common = [[0]*n for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if v in adj[u]:
                        c = len(adj[u] & adj[v])
                        common[u][v] = c
                        common[v][u] = c
            
            # Compute edge weights: binom(common[u][v], 2)
            edge_weight = {}
            for u, v in edges:
                c = common[u][v]
                edge_weight[(u, v)] = c * (c - 1) // 2
            
            # Greedy removal
            # We maintain a set of current edges
            current_edges = set(edges)
            # We need a way to find the max weight edge efficiently
            # Since n is small, we can just scan all current edges each time.
            # O(E * n^2) roughly. E ~ 100. n^2 ~ 2000. 200k ops per removal.
            # 100 removals -> 20M ops. Might be slow in Python.
            # Optimization: Keep a list of edges and their weights, update only affected ones.
            
            # Let's use a priority queue or just a list of (weight, u, v)
            # But weights change.
            # Given n=47, E~100, scanning all edges is probably fine if we do it smartly.
            # Let's try the scan approach first. If too slow, optimize.
            
            success = True
            # We might need to add edges back if degrees drop?
            # The strategy is: only remove if it doesn't violate min_deg.
            # If we get stuck (all C4-breaking edges violate min_deg), we fail this restart.
            
            while True:
                # Find max weight edge
                max_w = -1
                best_edge = None
                for e in list(current_edges):
                    u, v = e
                    w = edge_weight.get(e, 0)
                    if w > max_w:
                        max_w = w
                        best_edge = e
                
                if max_w <= 0:
                    # No C4s left
                    break
                
                u, v = best_edge
                # Check if we can remove
                if len(adj[u]) <= min_deg_target or len(adj[v]) <= min_deg_target:
                    # Cannot remove this edge without dropping below target
                    # Try to find another edge with same max weight that we CAN remove?
                    # Or just fail?
                    # Let's try to find any edge with weight > 0 that we can remove.
                    removed_any = False
                    for e in list(current_edges):
                        uu, vv = e
                        if e == best_edge: continue
                        if edge_weight.get(e, 0) > 0:
                            if len(adj[uu]) > min_deg_target and len(adj[vv]) > min_deg_target:
                                # Remove this one instead
                                best_edge = e
                                u, v = e
                                removed_any = True
                                break
                    
                    if not removed_any:
                        # Stuck
                        success = False
                        break
                
                # Remove best_edge (u, v)
                current_edges.remove((u, v))
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Update common counts and weights
                # Affected common counts:
                # 1. common[u][v] becomes 0? No, common[u][v] is count of common neighbors.
                #    Removing edge (u,v) does NOT change the set of common neighbors of u and v.
                #    It only changes the adjacency.
                #    However, the weight of edge (u,v) is now irrelevant as it's removed.
                
                # 2. For any w in N(u) (old N(u), which included v, now excludes v):
                #    If w was connected to v, then u was a common neighbor of w and v.
                #    Now u is not a neighbor of v. So common[w][v] decreases by 1.
                #    Similarly, if w was connected to u, common[w][u] decreases by 1.
                
                # Let's iterate over neighbors of u and v
                # Note: adj[u] and adj[v] have already been updated (v removed from u, u from v)
                # So we need to use the OLD neighbor sets?
                # We can just iterate over current adj[u] and adj[v].
                # But we need to know which w were connected to the other one.
                
                # For w in adj[u]:
                #   If w in adj[v]:
                #      u was a common neighbor of w and v?
                #      Yes, u~w and u~v (before removal).
                #      So common[w][v] -= 1.
                #      And common[v][w] -= 1.
                
                # For w in adj[v]:
                #   If w in adj[u]:
                #      v was a common neighbor of w and u?
                #      Yes, v~w and v~u (before removal).
                #      So common[w][u] -= 1.
                #      And common[u][w] -= 1.
                
                for w in list(adj[u]):
                    if w in adj[v]:
                        common[w][v] -= 1
                        common[v][w] -= 1
                        # Update weight of edge (w, v) if it exists
                        if (w, v) in current_edges or (v, w) in current_edges:
                            e_key = (w, v) if w < v else (v, w)
                            c = common[e_key[0]][e_key[1]]
                            edge_weight[e_key] = c * (c - 1) // 2
                
                for w in list(adj[v]):
                    if w in adj[u]:
                        common[w][u] -= 1
                        common[u][w] -= 1
                        # Update weight of edge (w, u) if it exists
                        if (w, u) in current_edges or (u, w) in current_edges:
                            e_key = (w, u) if w < u else (u, w)
                            c = common[e_key[0]][e_key[1]]
                            edge_weight[e_key] = c * (c - 1) // 2
                
                # Also, the weight of edges incident to u or v might change
                # because common[u][w] or common[v][w] changed?
                # Yes, if w was connected to the other one.
                # We updated common[w][v] and common[w][u].
                # Do we need to update weights of edges (u, w) and (v, w)?
                # Weight of (u, w) depends on common[u][w].
                # Did common[u][w] change?
                # common[u][w] = |N(u) \cap N(w)|.
                # N(u) lost v. If v in N(w), then v was in the intersection.
                # So if v in N(w), common[u][w] decreased by 1.
                # We handled this in the loop "for w in adj[v]: if w in adj[u]: common[w][u] -= 1".
                # So common[u][w] is updated.
                # So we must update edge_weight for (u, w) if it exists.
                
                # Let's just recalculate weights for all edges incident to u or v to be safe.
                for w in adj[u]:
                    e_key = (u, w) if u < w else (w, u)
                    if e_key in current_edges:
                        c = common[e_key[0]][e_key[1]]
                        edge_weight[e_key] = c * (c - 1) // 2
                
                for w in adj[v]:
                    e_key = (v, w) if v < w else (w, v)
                    if e_key in current_edges:
                        c = common[e_key[0]][e_key[1]]
                        edge_weight[e_key] = c * (c - 1) // 2

            if success:
                # Verify min degree
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_target:
                    # Verify C4 free
                    is_c4_free = True
                    for u in range(n):
                        for v in range(u+1, n):
                            if common[u][v] >= 2:
                                is_c4_free = False
                                break
                        if not is_c4_free: break
                    if is_c4_free:
                        return [[u, v] for u, v in current_edges]
        
        return None

    # Try n=47
    for n in [47, 46, 45, 44]:
        if time.time() > deadline:
            break
        
        result = try_construct(n)
        if result is not None:
            return {'n': n, 'edges': result}
    
    # Fallback
    return {'n': 39, 'edges': []}
