"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
import math
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        remaining = set(edges)
        cycles = []
        
        def build_adj(rem):
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        def find_cycle_in(rem_edges):
            """Find a cycle in the graph defined by rem_edges using randomized DFS."""
            if len(rem_edges) < 3:
                return None
            
            adj = build_adj(rem_edges)
            
            # Find vertices with degree >= 2
            degs = {}
            for u, v in rem_edges:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1
            
            candidates = [v for v in degs if degs[v] >= 2]
            if not candidates:
                return None
            
            # Try a few random starts
            for _ in range(10):
                start = random.choice(candidates)
                path = [start]
                visited = {start}
                current = start
                
                for _ in range(30): # Limit path length for speed
                    neighbors = [nb for nb in adj.get(current, []) 
                                if (min(current, nb), max(current, nb)) in rem_edges]
                    if not neighbors:
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if start in neighbors and len(path) >= 3:
                        return path[:]
                    
                    if unvisited:
                        # Pick random unvisited neighbor
                        next_node = random.choice(unvisited)
                    else:
                        # If no unvisited, check if we can close
                        if start in neighbors:
                            return path[:]
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
            
            return None

        # 1. Initial Greedy Construction
        while remaining:
            cycle = find_cycle_in(remaining)
            if not cycle:
                break
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                remaining.discard((min(u, v), max(u, v)))
            cycles.append(cycle)

        def current_score():
            return len(cycles) + len(remaining)

        best_cycles = [c[:] for c in cycles]
        best_remaining = remaining.copy()
        best_score = current_score()
        
        # 2. Simulated Annealing / Local Search
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        start_time = time.time()
        
        while T > min_T and time.time() < deadline - 5:
            # Generate a move
            move_type = random.random()
            
            # Copy state
            curr_cycles = [c[:] for c in cycles]
            curr_remaining = remaining.copy()
            
            if move_type < 0.4:
                # Try to add a cycle
                new_cycle = find_cycle_in(curr_remaining)
                if new_cycle:
                    for i in range(len(new_cycle) - 1):
                        u, v = new_cycle[i], new_cycle[i + 1]
                        curr_remaining.discard((min(u, v), max(u, v)))
                    curr_cycles.append(new_cycle)
            elif move_type < 0.7:
                # Remove a random cycle
                if curr_cycles:
                    idx = random.randint(0, len(curr_cycles) - 1)
                    rem_cycle = curr_cycles.pop(idx)
                    for i in range(len(rem_cycle) - 1):
                        u, v = rem_cycle[i], rem_cycle[i + 1]
                        curr_remaining.add((min(u, v), max(u, v)))
            else:
                # Swap: Remove one, try to add another
                if curr_cycles:
                    idx = random.randint(0, len(curr_cycles) - 1)
                    rem_cycle = curr_cycles.pop(idx)
                    for i in range(len(rem_cycle) - 1):
                        u, v = rem_cycle[i], rem_cycle[i + 1]
                        curr_remaining.add((min(u, v), max(u, v)))
                    
                    new_cycle = find_cycle_in(curr_remaining)
                    if new_cycle:
                        for i in range(len(new_cycle) - 1):
                            u, v = new_cycle[i], new_cycle[i + 1]
                            curr_remaining.discard((min(u, v), max(u, v)))
                        curr_cycles.append(new_cycle)

            # Calculate delta
            new_score = len(curr_cycles) + len(curr_remaining)
            delta = new_score - current_score()
            
            # Accept
            if delta < 0 or random.random() < math.exp(-delta / T):
                cycles = curr_cycles
                remaining = curr_remaining
                
                if new_score < best_score:
                    best_score = new_score
                    best_cycles = [c[:] for c in cycles]
                    best_remaining = remaining.copy()
            
            T *= cooling

        # Reconstruct final result from best state
        pieces = []
        for c in best_cycles:
            pieces.append(c[:-1][:])
        for e in sorted(best_remaining):
            pieces.append([e[0], e[1]])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
