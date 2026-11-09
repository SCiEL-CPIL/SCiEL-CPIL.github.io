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

    def optimize_decomposition(edges, n, deadline, seed_offset=0):
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u, v), max(u, v)))
        
        # Helper to build adjacency from edge set
        def build_adj(edge_set):
            adj = {i: set() for i in range(n)}
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        # Helper to find a cycle using Randomized Walk
        def find_cycle_walk(rem_edges, adj_rem, rng):
            if not rem_edges:
                return None
            # Pick a random vertex with degree > 0
            candidates = [u for u in range(n) if adj_rem[u]]
            if not candidates:
                return None
            start = rng.choice(candidates)
            visited = {start: 0}
            path = [start]
            current = start
            # Limit steps
            for _ in range(150):
                if not adj_rem[current]:
                    break
                neighbors = list(adj_rem[current])
                if not neighbors:
                    break
                next_v = rng.choice(neighbors)
                if next_v in visited:
                    start_idx = visited[next_v]
                    cycle = path[start_idx:]
                    if len(cycle) >= 3:
                        return cycle
                else:
                    visited[next_v] = len(path)
                    path.append(next_v)
                    current = next_v
            return None

        # Greedy Initialization with Random Restarts
        best_initial_cost = float('inf')
        best_initial_cycles = []
        best_initial_rem = set()
        
        for restart in range(20):
            rng_init = Random(1000 + seed_offset + restart)
            rem_edges = set(all_edges)
            cycles = []
            adj_rem = build_adj(rem_edges)
            
            while True:
                if not rem_edges:
                    break
                # Find a cycle using walk
                cyc = find_cycle_walk(rem_edges, adj_rem, rng_init)
                if not cyc:
                    # Try standard DFS if walk fails? Or just break.
                    # Walk is usually good enough.
                    break
                
                # Remove cycle edges
                k = len(cyc)
                for i in range(k):
                    u = cyc[i]
                    v = cyc[(i+1)%k]
                    e = (min(u,v), max(u,v))
                    rem_edges.discard(e)
                    adj_rem[u].discard(v)
                    adj_rem[v].discard(u)
                cycles.append(cyc)
            
            cost = len(cycles) + len(rem_edges)
            if cost < best_initial_cost:
                best_initial_cost = cost
                best_initial_cycles = [c[:] for c in cycles]
                best_initial_rem = set(rem_edges)
        
        cycles = best_initial_cycles
        rem_edges = best_initial_rem
        current_cost = best_initial_cost
        
        # SA Phase
        rng_sa = Random(42 + seed_offset)
        temp = 1.0
        cooling = 0.999
        best_cost = current_cost
        best_cycles = [c[:] for c in cycles]
        best_rem = set(rem_edges)
        
        adj_rem = build_adj(rem_edges)
        
        while time.time() < deadline:
            # Rebuild adj_rem periodically if it gets stale? 
            # Since we update incrementally, it should be fine.
            # But we need to be careful with updates.
            
            if rng_sa.random() < 0.5:
                # Move: Remove a cycle
                if not cycles:
                    continue
                idx = rng_sa.randint(0, len(cycles)-1)
                cyc = cycles[idx]
                cycles.pop(idx)
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1)%len(cyc)]
                    e = (min(u,v), max(u,v))
                    rem_edges.add(e)
                    adj_rem[u].add(v)
                    adj_rem[v].add(u)
                
                new_cost = len(cycles) + len(rem_edges)
                delta = new_cost - current_cost
                
                if delta < 0 or rng_sa.expovariate(1.0/temp) > delta:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cycles = [c[:] for c in cycles]
                        best_rem = set(rem_edges)
                else:
                    # Revert
                    cycles.insert(idx, cyc)
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i+1)%len(cyc)]
                        e = (min(u,v), max(u,v))
                        rem_edges.discard(e)
                        adj_rem[u].discard(v)
                        adj_rem[v].discard(u)
            else:
                # Move: Add a cycle
                if not rem_edges:
                    continue
                cyc = find_cycle_walk(rem_edges, adj_rem, rng_sa)
                if not cyc:
                    continue
                
                # Validate (walk guarantees it, but double check just in case)
                valid = True
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i+1)%len(cyc)]
                    e = (min(u,v), max(u,v))
                    if e not in rem_edges:
                        valid = False
                        break
                if not valid:
                    continue
                
                k = len(cyc)
                for i in range(k):
                    u = cyc[i]
                    v = cyc[(i+1)%k]
                    e = (min(u,v), max(u,v))
                    rem_edges.discard(e)
                    adj_rem[u].discard(v)
                    adj_rem[v].discard(u)
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
                    # Revert
                    cycles.pop()
                    for i in range(k):
                        u = cyc[i]
                        v = cyc[(i+1)%k]
                        e = (min(u,v), max(u,v))
                        rem_edges.add(e)
                        adj_rem[u].add(v)
                        adj_rem[v].add(u)
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        pieces = []
        for cyc in best_cycles:
            pieces.append(list(cyc))
        for e in best_rem:
            pieces.append(list(e))
        return pieces

    t_start = time.time()
    total_budget = 115
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time: 30s, 35s, rest
    deadlines = [
        t_start + 30,
        t_start + 65,
        t_start + total_budget
    ]
    
    for i, spec in enumerate(specs):
        n = spec[0]
        edges = instance(*spec)
        pieces = optimize_decomposition(edges, n, deadlines[i], seed_offset=i*100)
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    construct()
