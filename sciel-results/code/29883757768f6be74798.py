"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(rem):
        adj = defaultdict(set)
        for u, v in rem:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_cycle(remaining, adj, rng, max_attempts=20, max_len=30):
        """Find a cycle in the remaining edges. Returns list of vertices or None."""
        if len(remaining) < 3:
            return None
        
        # Find vertices with degree >= 2
        candidates = [v for v, nbs in adj.items() if len(nbs) >= 2]
        if not candidates:
            return None
            
        best_cycle = None
        
        # Shuffle candidates to randomize start
        rng.shuffle(candidates)
        
        for start in candidates:
            # Try to find a cycle starting at start
            # Use a simple randomized DFS
            for _ in range(max_attempts):
                path = [start]
                visited = {start}
                current = start
                found = False
                
                for _ in range(max_len):
                    neighbors = [nb for nb in adj.get(current, []) 
                                if (min(current, nb), max(current, nb)) in remaining]
                    
                    if not neighbors:
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if start in neighbors and len(path) >= 3:
                        path.append(start)
                        found = True
                        break
                    
                    if unvisited:
                        next_node = rng.choice(unvisited)
                    else:
                        if start in neighbors and len(path) >= 3:
                            path.append(start)
                            found = True
                            break
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if found:
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = path[:]
                    # If we found a long cycle, we might stop early to save time
                    if len(best_cycle) > 10: 
                         break
            if best_cycle and len(best_cycle) > 10:
                break

        return best_cycle

    def greedy_decompose(edges, rng, max_attempts=20):
        """
        Decompose edges into cycles and remaining singletons.
        Returns (cycles, remaining_edges)
        """
        remaining = set(edges)
        cycles = []
        
        while remaining:
            adj = build_adj(remaining)
            cycle = find_cycle(remaining, adj, rng, max_attempts=max_attempts)
            
            if cycle:
                # Extract cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
                cycles.append(cycle[:-1])
            else:
                break
        
        return cycles, remaining

    def decompose(edges):
        """
        Main decomposition using Greedy + Local Search.
        """
        if not edges:
            return []
            
        rng = Random(42) # Fixed seed for reproducibility in local search? Or time-based?
        # Use time-based for variety if needed, but fixed is safer for consistency.
        # Let's use a seed derived from edges to be deterministic per graph.
        seed_val = hash(frozenset(edges)) % (2**32)
        rng = Random(seed_val)

        # 1. Initial Greedy Solution
        cycles, remaining = greedy_decompose(edges, rng, max_attempts=50)
        
        current_savings = sum(len(c) - 1 for c in cycles)
        
        # 2. Local Search
        # Objective: Maximize savings = sum(|C_i| - 1)
        # Move: Pick a cycle C, remove it, try to re-decompose (Remaining + C) to get better savings.
        
        time_limit = deadline - 10
        while time.time() < time_limit:
            if not cycles:
                break
            
            # Pick a random cycle to "uncover"
            idx = rng.randint(0, len(cycles) - 1)
            C = cycles[idx]
            edges_C = set((min(C[i], C[i+1]), max(C[i], C[i+1])) for i in range(len(C)-1))
            
            # Pool of edges to re-decompose
            pool = remaining | edges_C
            
            # Try to find a better decomposition of the pool
            # We compare against the current savings of C (which is len(C)-1)
            # and the savings of the rest of the pool (which is 0, as they are singletons)
            # So we just need to find cycles in `pool` that have total savings > len(C)-1
            
            # Use a slightly less aggressive greedy to find potential improvements
            # If we find a single cycle longer than C, or multiple cycles with higher sum, it's a win.
            
            new_cycles, new_remaining = greedy_decompose(pool, rng, max_attempts=10)
            
            new_savings_pool = sum(len(c) - 1 for c in new_cycles)
            old_savings_pool = len(C) - 1 # Since other edges in pool were singletons (savings 0)
            
            if new_savings_pool > old_savings_pool:
                # Accept the move
                # Update cycles: remove C, add new_cycles
                cycles = [c for i, c in enumerate(cycles) if i != idx]
                cycles.extend(new_cycles)
                
                # Update remaining:
                # Old remaining was `remaining`.
                # Pool was `remaining` + `edges_C`.
                # New remaining is `new_remaining`.
                # Note: `new_remaining` contains edges from `remaining` that were not covered by `new_cycles`
                # and edges from `edges_C` that were not covered.
                # This is correct.
                remaining = new_remaining
                
                # Update global savings
                current_savings += (new_savings_pool - old_savings_pool)
                
                # Optional: If we improved significantly, we could reset time or continue.
                # We just continue the loop.

        # 3. Final Result Construction
        pieces = []
        for c in cycles:
            pieces.append(c)
        for e in remaining:
            pieces.append([e[0], e[1]])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 20:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
