import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin
    
    def is_regular_subset(adj, subset):
        # subset is a tuple of indices
        # Extract submatrix
        # For speed, we can just compute degrees for these vertices within the subset
        # deg[i] = sum(adj[i, j] for j in subset if i != j)
        # Since n is small, this is fast enough
        sub_adj = adj[np.ix_(subset, subset)]
        degrees = np.sum(sub_adj, axis=1)
        return np.all(degrees == degrees[0])

    def count_regular_6(adj):
        count = 0
        n = adj.shape[0]
        # Pre-compute combinations? No, generate on fly
        for combo in itertools.combinations(range(n), 6):
            if is_regular_subset(adj, combo):
                count += 1
        return count

    def full_check(adj):
        n = adj.shape[0]
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                if is_regular_subset(adj, combo):
                    return False
        return True

    def get_affected_subsets(adj, u, v, n):
        # Returns list of subsets of size 6 containing u and v
        # And their current regularity status
        # We need to check subsets S of size 6 such that u, v in S.
        # S = {u, v} + 4 others from [0..n-1] \ {u, v}
        others = [i for i in range(n) if i != u and i != v]
        affected = []
        for combo4 in itertools.combinations(others, 4):
            subset = tuple(sorted((u, v) + tuple(combo4)))
            # Check current state
            current_reg = is_regular_subset(adj, subset)
            affected.append((subset, current_reg))
        return affected

    def try_flip(adj, u, v, n, current_cost):
        # Flip edge (u, v)
        # Check affected subsets
        affected = get_affected_subsets(adj, u, v, n)
        
        # Temporarily flip
        adj[u, v] = not adj[u, v]
        adj[v, u] = adj[u, v]
        
        new_cost = current_cost
        for subset, was_reg in affected:
            is_reg_now = is_regular_subset(adj, subset)
            if was_reg and not is_reg_now:
                new_cost -= 1
            elif not was_reg and is_reg_now:
                new_cost += 1
        
        # Revert if not better? Or return new cost and let caller decide
        # For now, assume we want to minimize cost.
        # If new_cost < current_cost, we accept.
        # If new_cost == current_cost, we might accept to escape local minima (SA).
        
        return new_cost

    def local_search(adj, n, time_limit):
        current_cost = count_regular_6(adj)
        if current_cost == 0:
            return adj, 0
        
        best_adj = adj.copy()
        best_cost = current_cost
        
        while time.time() < time_limit:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Calculate new cost if flipped
            # We need to be careful with the delta logic.
            # The function try_flip modifies adj and returns new cost.
            # We need to revert if we don't accept.
            
            # Let's do a simple greedy step
            # Try flipping (u,v)
            # To avoid side effects, we can just simulate.
            
            # Optimization: 
            # 1. Get affected subsets and their current state.
            # 2. Flip in memory.
            # 3. Check new state.
            # 4. Calculate delta.
            # 5. Revert.
            # 6. If delta < 0, apply flip permanently.
            
            affected = get_affected_subsets(adj, u, v, n)
            
            # Flip
            adj[u, v] = not adj[u, v]
            adj[v, u] = adj[u, v]
            
            new_cost = current_cost
            for subset, was_reg in affected:
                is_reg_now = is_regular_subset(adj, subset)
                if was_reg and not is_reg_now:
                    new_cost -= 1
                elif not was_reg and is_reg_now:
                    new_cost += 1
            
            # Revert
            adj[u, v] = not adj[u, v]
            adj[v, u] = adj[u, v]
            
            if new_cost < current_cost:
                # Accept
                adj[u, v] = not adj[u, v]
                adj[v, u] = adj[u, v]
                current_cost = new_cost
                if current_cost == 0:
                    break
            elif new_cost == current_cost and random.random() < 0.1:
                # Accept with small prob to escape local minima
                adj[u, v] = not adj[u, v]
                adj[v, u] = adj[u, v]
                current_cost = new_cost
                
        return adj, current_cost

    def run_search(n, time_limit):
        # Try to find a graph on n vertices with no regular 6-subsets
        while time.time() < time_limit:
            adj = np.random.randint(0, 2, (n, n), dtype=bool)
            np.fill_diagonal(adj, False)
            adj = (adj + adj.T) % 2
            
            adj, cost = local_search(adj, n, time_limit)
            
            if cost == 0:
                # Verify full validity
                if full_check(adj):
                    return adj
            # If time is running out, break
            if time.time() > time_limit - 10:
                break
                
        return None

    # Try n=21
    adj21 = run_search(21, time_limit)
    if adj21 is not None:
        edges = []
        n = adj21.shape[0]
        for u in range(n):
            for v in range(u + 1, n):
                if adj21[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}

    # Try n=20
    adj20 = run_search(20, time_limit)
    if adj20 is not None:
        edges = []
        n = adj20.shape[0]
        for u in range(n):
            for v in range(u + 1, n):
                if adj20[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}

    # Try n=19
    adj19 = run_search(19, time_limit)
    if adj19 is not None:
        edges = []
        n = adj19.shape[0]
        for u in range(n):
            for v in range(u + 1, n):
                if adj19[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}

    # Fallback: Empty graph on 5 vertices (always valid)
    return {'n': 5, 'edges': []}
