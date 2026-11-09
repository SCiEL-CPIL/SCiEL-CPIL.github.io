import time
import random
import sys

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_sa(edges, n):
        # Initial state: all edges as separate pieces
        # Representation: list of pieces. Piece is either [u, v] or [v1, v2, ..., vk]
        # For efficiency, we track:
        # 1. The set of edges currently in cycles (to know which are "free")
        # 2. The list of pieces
        # 3. A map from edge to index in pieces? Or just check membership.
        
        # Let's represent the state as a set of "active cycles" and the rest are edges.
        # Actually, simpler: State is a set of edge tuples that are part of a cycle.
        # But we need to know the cycle structure to validate moves.
        # Let's stick to a list of pieces, but maintain an adjacency set for current remaining edges?
        # No, SA needs to evaluate cost. Cost = len(pieces).
        # Moves:
        # 1. Pick a cycle piece, break it into edges. (Cost increases by k-1)
        # 2. Pick a set of edges that form a cycle, merge them. (Cost decreases by k-1)
        # 3. Pick a cycle, remove one edge, add it to edges. (Cost increases by 1)
        # 4. Pick an edge, see if it can be added to a cycle to form a larger cycle? 
        #    This is complex.
        
        # Alternative: 
        # State: A list of cycles (lists of vertices) and a set of remaining edges.
        # Cost: len(cycles) + len(remaining_edges)
        
        # Initial State:
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u, v), max(u, v)))
        
        # Greedy initialization to get a good starting point
        rem_edges = set(all_edges)
        cycles = []
        
        def find_cycle_in(rem):
            # Simple DFS to find any cycle
            if not rem:
                return None
            adj = {i: [] for i in range(n)}
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = set()
            parent = {}
            def dfs(u, start):
                visited.add(u)
                for v in adj[u]:
                    if v == start and len(visited) > 2:
                        # Found cycle
                        cycle = [v]
                        cur = u
                        while cur != v:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.reverse()
                        return cycle
                    if v not in visited:
                        parent[v] = u
                        res = dfs(v, start)
                        if res:
                            return res
                return None
            
            for u in range(n):
                if adj[u]:
                    visited = set()
                    parent = {}
                    res = dfs(u, u)
                    if res:
                        return res
            return None

        while True:
            if not rem_edges:
                break
            cyc = find_cycle_in(rem_edges)
            if not cyc:
                break
            # Remove edges of cycle from rem_edges
            k = len(cyc)
            for i in range(k):
                u = cyc[i]
                v = cyc[(i+1)%k]
                e = (min(u,v), max(u,v))
                rem_edges.discard(e)
            cycles.append(cyc)
            
        # Now we have an initial decomposition: 'cycles' and 'rem_edges'
        # Cost = len(cycles) + len(rem_edges)
        
        # SA Setup
        start_time = time.time()
        deadline = start_time + 115
        current_cost = len(cycles) + len(rem_edges)
        
        # Data structures for fast access
        # edge_to_cycle: dict mapping edge tuple to index in cycles list
        # This is hard to maintain dynamically if we merge/split.
        # Let's just use the set of edges in cycles for validity checks.
        
        # For SA, we need to be able to:
        # 1. Remove a cycle (add its edges to rem_edges)
        # 2. Add a cycle (remove its edges from rem_edges)
        
        # To find candidate cycles to add, we can look at rem_edges.
        # To find cycles to remove, we look at cycles.
        
        # Optimization: Pre-compute a list of potential cycles? No, graph changes.
        
        # Let's define a helper to get neighbors in rem_edges
        def get_rem_adj():
            adj = {i: set() for i in range(n)}
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        # SA Loop
        temp = 1.0
        cooling = 0.9995
        rng_sa = Random(42)
        
        # To make it faster, we can limit the number of iterations or time.
        # Since n is small (20-30), we can afford many iterations.
        
        # Helper to check if a list of vertices forms a valid cycle in rem_edges
        def is_valid_cycle(cyc):
            k = len(cyc)
            if k < 3:
                return False
            if len(set(cyc)) != k:
                return False
            for i in range(k):
                u = cyc[i]
                v = cyc[(i+1)%k]
                e = (min(u,v), max(u,v))
                if e not in rem_edges:
                    return False
            return True

        # Helper to extract a cycle from a path in rem_adj
        def extract_cycle(start, end, path):
            # path is list of vertices from start to end
            # We need to close the loop.
            # This is tricky. Better to use DFS to find cycles directly.
            pass

        # Let's use a different move set for SA:
        # Move 1: Remove a random cycle from 'cycles' and add its edges to 'rem_edges'.
        # Move 2: Find a cycle in 'rem_edges' (using randomized DFS) and move it to 'cycles'.
        
        # To find a cycle in rem_edges quickly:
        def find_random_cycle_in_rem():
            if not rem_edges:
                return None
            adj = get_rem_adj()
            # Pick a random edge to start
            if not rem_edges:
                return None
            u, v = rng_sa.choice(list(rem_edges))
            # Try to find a cycle containing u-v? Or just any cycle.
            # Randomized DFS from u
            visited = {u: True}
            parent = {u: None}
            stack = [u]
            # We want to find a back edge.
            # This might be slow if we do it from scratch every time.
            # Given n<=30, it's fine.
            
            # Standard DFS cycle detection
            def dfs_cycle(u, start):
                for v in adj[u]:
                    if v == start:
                        if len(visited) > 2:
                            # Reconstruct
                            cycle = [start]
                            cur = u
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            return cycle
                    elif v not in visited:
                        visited[v] = True
                        parent[v] = u
                        res = dfs_cycle(v, start)
                        if res:
                            return res
                return None
            
            # Try starting from u
            visited = {u: True}
            parent = {u: None}
            res = dfs_cycle(u, u)
            if res:
                return res
            
            # If not found, try another random start
            # Actually, the above DFS explores the whole component. 
            # If no cycle found in component of u, return None.
            return None

        # We need to track the best solution found
        best_cost = current_cost
        best_cycles = [c[:] for c in cycles]
        best_rem = set(rem_edges)
        
        while time.time() < deadline:
            # Generate a move
            if rng_sa.random() < 0.5:
                # Remove a cycle
                if not cycles:
                    continue
                idx = rng_sa.randint(0, len(cycles)-1)
                cyc = cycles[idx]
                # Remove from cycles
                cycles.pop(idx)
                # Add edges to rem
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1)%len(cyc)]
                    rem_edges.add((min(u,v), max(u,v)))
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                # delta should be len(cyc)-1 > 0
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    # Reject: Revert
                    cycles.insert(idx, cyc)
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i+1)%len(cyc)]
                        rem_edges.discard((min(u,v), max(u,v)))
            else:
                # Add a cycle
                if not rem_edges:
                    continue
                # Find a cycle in rem_edges
                # To make it faster, we can restrict search or use the previous adj
                # But rem_edges changes.
                cyc = find_random_cycle_in_rem()
                if not cyc:
                    continue
                
                # Validate
                if not is_valid_cycle(cyc):
                    continue
                
                # Apply move
                k = len(cyc)
                for i in range(k):
                    u = cyc[i]
                    v = cyc[(i+1)%k]
                    rem_edges.discard((min(u,v), max(u,v)))
                cycles.append(cyc)
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                # delta should be -(k-1) < 0
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    # Reject: Revert
                    cycles.pop()
                    for i in range(k):
                        u = cyc[i]
                        v = cyc[(i+1)%k]
                        rem_edges.add((min(u,v), max(u,v)))
            
            # Cool down
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
        
        # Reconstruct pieces from best state
        pieces = []
        for cyc in best_cycles:
            pieces.append(list(cyc))
        for e in best_rem:
            pieces.append(list(e))
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n = spec[0]
        edges = instance(*spec)
        # Run SA for each graph
        # To ensure we finish in time, we distribute the time budget.
        # 115s total. 3 graphs. ~38s each.
        # The decompose_sa function uses a deadline relative to its start.
        # We should pass the deadline or let it manage.
        # Current decompose_sa uses time.time() + 115. 
        # We need to adjust this to use the remaining global time.
        
        # Let's modify decompose_sa to take a deadline.
        # Or just run it and let it stop.
        # Since they run sequentially, we can just let each take its share.
        # But if the first takes 100s, the others fail.
        # Better: Pass a deadline.
        
        # I will rewrite decompose_sa to accept a deadline.
        # Wait, I can't rewrite the function definition inside the loop easily without changing the code structure.
        # I will just define a local deadline in the main loop and pass it.
        
        # Let's refactor decompose_sa to accept deadline.
        # I'll do a quick edit in the code below.
        
        pass

    # Re-implementing properly to handle deadlines
    def decompose_sa_v2(edges, n, deadline):
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u, v), max(u, v)))
        
        rem_edges = set(all_edges)
        cycles = []
        
        def find_cycle_in(rem):
            if not rem:
                return None
            adj = {i: [] for i in range(n)}
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = set()
            parent = {}
            def dfs(u, start):
                visited.add(u)
                for v in adj[u]:
                    if v == start and len(visited) > 2:
                        cycle = [v]
                        cur = u
                        while cur != v:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.reverse()
                        return cycle
                    if v not in visited:
                        parent[v] = u
                        res = dfs(v, start)
                        if res:
                            return res
                return None
            
            for u in range(n):
                if adj[u]:
                    visited = set()
                    parent = {}
                    res = dfs(u, u)
                    if res:
                        return res
            return None

        while True:
            if not rem_edges:
                break
            cyc = find_cycle_in(rem_edges)
            if not cyc:
                break
            k = len(cyc)
            for i in range(k):
                u = cyc[i]
                v = cyc[(i+1)%k]
                e = (min(u,v), max(u,v))
                rem_edges.discard(e)
            cycles.append(cyc)
            
        current_cost = len(cycles) + len(rem_edges)
        rng_sa = Random(42)
        temp = 1.0
        cooling = 0.9995
        
        def get_rem_adj():
            adj = {i: set() for i in range(n)}
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def is_valid_cycle(cyc):
            k = len(cyc)
            if k < 3:
                return False
            if len(set(cyc)) != k:
                return False
            for i in range(k):
                u = cyc[i]
                v = cyc[(i+1)%k]
                e = (min(u,v), max(u,v))
                if e not in rem_edges:
                    return False
            return True

        def find_random_cycle_in_rem():
            if not rem_edges:
                return None
            adj = get_rem_adj()
            u, v = rng_sa.choice(list(rem_edges))
            
            visited = {u: True}
            parent = {u: None}
            
            def dfs_cycle(u, start):
                for v in adj[u]:
                    if v == start:
                        if len(visited) > 2:
                            cycle = [start]
                            cur = u
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            return cycle
                    elif v not in visited:
                        visited[v] = True
                        parent[v] = u
                        res = dfs_cycle(v, start)
                        if res:
                            return res
                return None
            
            res = dfs_cycle(u, u)
            return res

        best_cost = current_cost
        best_cycles = [c[:] for c in cycles]
        best_rem = set(rem_edges)
        
        while time.time() < deadline:
            if rng_sa.random() < 0.5:
                if not cycles:
                    continue
                idx = rng_sa.randint(0, len(cycles)-1)
                cyc = cycles[idx]
                cycles.pop(idx)
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1)%len(cyc)]
                    rem_edges.add((min(u,v), max(u,v)))
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    cycles.insert(idx, cyc)
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i+1)%len(cyc)]
                        rem_edges.discard((min(u,v), max(u,v)))
            else:
                if not rem_edges:
                    continue
                cyc = find_random_cycle_in_rem()
                if not cyc:
                    continue
                if not is_valid_cycle(cyc):
                    continue
                
                k = len(cyc)
                for i in range(k):
                    u = cyc[i]
                    v = cyc[(i+1)%k]
                    rem_edges.discard((min(u,v), max(u,v)))
                cycles.append(cyc)
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    cycles.pop()
                    for i in range(k):
                        u = cyc[i]
                        v = cyc[(i+1)%k]
                        rem_edges.add((min(u,v), max(u,v)))
            
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
        
        pieces = []
        for cyc in best_cycles:
            pieces.append(list(cyc))
        for e in best_rem:
            pieces.append(list(e))
        return pieces

    global_deadline = time.time() + 115
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n = spec[0]
        edges = instance(*spec)
        # Allocate time proportional to complexity? Or equal?
        # Let's just use the remaining time for the last one, and split earlier ones.
        # Simple: give each 1/3 of the time, but check global deadline.
        # Actually, just pass the global deadline to the last one, and estimate for others.
        # Better: Calculate remaining time and pass it.
        # But we need to finish all 3.
        # Let's allocate 35s to each.
        # If one finishes early, the next gets more?
        # The function takes a deadline.
        
        # To be safe, we'll just pass a deadline that is 1/3 of the way through for the first two,
        # and the global deadline for the last.
        
        # Actually, simpler:
        # t_start = time.time()
        # For i, spec:
        #   deadline = t_start + (i+1) * (115/3)
        
        pass

    t_start = time.time()
    total_budget = 115
    results = []
    for i, spec in enumerate(((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))):
        n = spec[0]
        edges = instance(*spec)
        # Deadline for this instance
        if i < 2:
            dl = t_start + (i + 1) * (total_budget / 3.0)
        else:
            dl = t_start + total_budget
        
        pieces = decompose_sa_v2(edges, n, dl)
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    construct()
