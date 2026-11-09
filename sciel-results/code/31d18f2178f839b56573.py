import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_greedy_decomposition(edges, rng, max_cycles=100):
        """
        Greedily find cycles to minimize pieces.
        Returns (list_of_cycles, set_of_remaining_edges)
        """
        if not edges:
            return [], set()
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        remaining_edges = set(edges)
        
        # We want to find cycles that are "good". 
        # Heuristic: Find a cycle, remove it, repeat.
        # To get a good initial solution, we try to find long cycles.
        
        for _ in range(max_cycles):
            if not any(len(neigh) > 0 for neigh in adj.values()):
                break
            
            # Find a cycle using a randomized DFS/Warnsdorff
            # Start from a random vertex with high degree
            vertices = [u for u in adj if adj[u]]
            if not vertices:
                break
                
            # Try a few starts
            best_cycle = None
            for _ in range(10):
                start = rng.choice(vertices)
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    neighbors = [n for n in adj[current] if n not in visited]
                    if not neighbors:
                        break
                    
                    # Choose neighbor with fewest options (Warnsdorff)
                    # to close the loop
                    min_deg = float('inf')
                    next_node = None
                    for n in neighbors:
                        deg = len(adj[n] - visited)
                        if deg < min_deg:
                            min_deg = deg
                            next_node = n
                        elif deg == min_deg and rng.random() < 0.5:
                            next_node = n
                            
                    if next_node is None:
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    
                    if len(path) > 20: # Cap for speed
                        break
                
                if len(path) >= 3 and path[-1] in adj[path[0]]:
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = path
            
            if best_cycle:
                cycles.append(best_cycle)
                # Remove edges
                for i in range(len(best_cycle)):
                    u, v = best_cycle[i], best_cycle[(i+1) % len(best_cycle)]
                    e = (min(u,v), max(u,v))
                    remaining_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                break
                
        return cycles, remaining_edges

    def decompose(edges):
        adj_initial = defaultdict(set)
        for u, v in edges:
            adj_initial[u].add(v)
            adj_initial[v].add(u)
        
        # Initial Greedy Solution
        rng = random.Random(42)
        cycles, remaining = find_greedy_decomposition(edges, rng)
        
        # State representation for Local Search
        # We store the list of cycles and the set of remaining edges.
        current_cycles = cycles
        current_remaining = remaining
        
        # Objective: Minimize len(cycles) + len(remaining)
        current_score = len(current_cycles) + len(current_remaining)
        best_score = current_score
        best_cycles = [list(c) for c in current_cycles]
        best_remaining = set(current_remaining)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        steps = 0
        max_steps = 50000 # Limit steps to ensure time
        
        while time.time() - start_time < DEADLINE and steps < max_steps:
            steps += 1
            
            # Proposal: Pick k cycles to re-decompose
            # k=1 is weak. k=2 or 3 is better.
            if len(current_cycles) < 2:
                # If few cycles, try adding a new one from remaining
                # Find a cycle in remaining
                if current_remaining:
                    sub_cycles, sub_rem = find_greedy_decomposition(list(current_remaining), rng, max_cycles=1)
                    if sub_cycles:
                        # Propose adding this cycle
                        new_cycles = current_cycles + sub_cycles
                        # The edges of sub_cycles are removed from remaining
                        new_remaining = set(current_remaining)
                        for c in sub_cycles:
                            for i in range(len(c)):
                                u, v = c[i], c[(i+1) % len(c)]
                                e = (min(u,v), max(u,v))
                                new_remaining.discard(e)
                        
                        new_score = len(new_cycles) + len(new_remaining)
                        delta = new_score - current_score
                        if delta < 0 or rng.random() < pow(2.71828, -delta/T):
                            current_cycles = new_cycles
                            current_remaining = new_remaining
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_cycles = [list(c) for c in current_cycles]
                                best_remaining = set(current_remaining)
                T *= cooling
                continue
            
            k = rng.choice([2, 3])
            if len(current_cycles) < k:
                k = len(current_cycles)
            
            # Pick k random cycles
            indices = rng.sample(range(len(current_cycles)), k)
            selected_cycles = [current_cycles[i] for i in indices]
            
            # Get edges from these cycles
            pool_edges = set()
            for c in selected_cycles:
                for i in range(len(c)):
                    u, v = c[i], c[(i+1) % len(c)]
                    e = (min(u,v), max(u,v))
                    pool_edges.add(e)
            
            # Remove these cycles from current state
            new_cycles = [c for i, c in enumerate(current_cycles) if i not in indices]
            
            # Re-decompose the pool_edges
            # We want to find a decomposition of pool_edges that uses fewer pieces than k
            # Or at least doesn't use more, to allow exploration.
            new_sub_cycles, new_sub_remaining = find_greedy_decomposition(list(pool_edges), rng, max_cycles=k+5)
            
            # The new state is new_cycles + new_sub_cycles
            # And new_remaining = current_remaining + new_sub_remaining
            candidate_cycles = new_cycles + new_sub_cycles
            candidate_remaining = current_remaining | new_sub_remaining
            
            candidate_score = len(candidate_cycles) + len(candidate_remaining)
            delta = candidate_score - current_score
            
            if delta < 0 or (T > T_min and rng.random() < pow(2.71828, -delta/T)):
                current_cycles = candidate_cycles
                current_remaining = candidate_remaining
                current_score = candidate_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_cycles = [list(c) for c in current_cycles]
                    best_remaining = set(current_remaining)
            
            T *= cooling
            
        # Construct final result
        pieces = []
        for c in best_cycles:
            pieces.append(list(c))
        for e in best_remaining:
            pieces.append([e[0], e[1]])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
