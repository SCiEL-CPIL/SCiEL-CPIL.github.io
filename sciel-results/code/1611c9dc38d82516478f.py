import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v
        mask_v = adj[v]
        # Iterate through set bits in mask_v
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            mask_v ^= lsb
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            mask_u ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update cc
        old_mask_v = adj[v] ^ (1 << u)
        m = old_mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            m ^= lsb
            
        old_mask_u = adj[u] ^ (1 << v)
        m = old_mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update cc
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            m ^= lsb
            
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            m ^= lsb

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    random.seed(42)
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Multiple restarts with greedy construction + local search
    # Use the remaining time budget for restarts
    max_restarts = 20
    time_per_restart = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Reset
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Greedy construction with random order
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
                
        current_count = count_edges()
        
        # Local Search: Simulated Annealing style
        T = 1.0
        cooling = 0.9995
        no_improve = 0
        
        while time.time() < deadline - 2.0 and T > 0.01:
            # Pick a random move: swap (remove one, add one) or just add
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap: remove a random existing edge, add a random non-existing edge
                # Find existing edges
                existing = []
                non_existing = []
                for u, v in all_edges:
                    if (adj[u] >> v) & 1:
                        existing.append((u, v))
                    else:
                        non_existing.append((u, v))
                        
                if not existing or not non_existing:
                    break
                
                u1, v1 = random.choice(existing)
                remove_edge(u1, v1)
                
                # Try to add a random non-existing edge
                # Prefer edges that are "close" to being addable? Or just random.
                # To be efficient, pick a few random candidates from non_existing
                # and try to add the first one that works.
                added = False
                # Shuffle a subset to avoid bias? Or just pick random indices.
                # For speed, pick up to 5 random candidates.
                candidates = random.sample(non_existing, min(5, len(non_existing)))
                for u2, v2 in candidates:
                    if can_add(u2, v2):
                        add_edge(u2, v2)
                        added = True
                        break
                        
                if not added:
                    # Revert
                    add_edge(u1, v1)
                    T *= cooling
                    continue
                
                new_count = count_edges()
                delta = new_count - current_count
                
                if delta > 0:
                    current_count = new_count
                    no_improve = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = get_edges()
                elif delta == 0:
                    # Accept with some probability to explore?
                    # For swaps, delta is usually 0.
                    # If we accept all delta=0, we do a random walk.
                    # Let's accept delta=0 always to explore the plateau.
                    current_count = new_count
                    no_improve += 1
                else:
                    # delta < 0 (should not happen for swap if we only add if valid, 
                    # but if we removed 1 and added 1, delta is 0. 
                    # Wait, if we remove 1, count drops by 1. Then add 1, count goes up by 1.
                    # So delta is always 0 for a successful swap.
                    # If we failed to add, we reverted, so we skip.
                    # So delta is always 0 here.
                    pass
                
            else:
                # Try to add a single edge
                # Pick random non-existing edge
                # We don't have a list of non-existing easily without scanning.
                # Scan is O(n^2) = 190, which is fast.
                non_existing = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
                if non_existing:
                    u, v = random.choice(non_existing)
                    if can_add(u, v):
                        add_edge(u, v)
                        current_count += 1
                        no_improve = 0
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = get_edges()
                    else:
                        no_improve += 1
                        
            T *= cooling
            if no_improve > 100:
                # Restart local search with higher T? Or break?
                # Let's just continue, maybe lower T helps.
                # Or break to try a new global restart.
                if T < 0.1:
                    break

    # Final polish: try to add any remaining edges
    # This is already done in the loop, but let's do a final sweep on the best state?
    # The best_edges is a snapshot. We need to reconstruct or just return it.
    # Since we return edges, we can just return best_edges.
    
    return best_edges
