"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115

    def get_edges_list(adj):
        edges = set()
        for u in range(n):
            mask = adj[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    edges.add((u, v))
        return edges

    def calc_energy(adj, edges):
        # Sum of squares of book sizes
        energy = 0
        for (u, v) in edges:
            common = adj[u] & adj[v]
            bs = common.bit_count()
            energy += bs * bs
        return energy

    def max_book_fast(adj, edges):
        mb = 0
        for (u, v) in edges:
            common = adj[u] & adj[v]
            bs = common.bit_count()
            if bs > mb:
                mb = bs
        return mb

    best_edges = None
    best_max = 999
    best_energy = float('inf')

    # Simulated Annealing Parameters
    T0 = 100.0
    T_min = 0.1
    cooling_rate = 0.999

    for seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        adj = [0] * n
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                key = (min(u,v), max(u,v))
                if key not in edges:
                    edges.add(key)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        current_energy = calc_energy(adj, edges)
        current_max = max_book_fast(adj, edges)
        T = T0
        steps = 0
        max_steps = 5000
        
        while time.time() < deadline and steps < max_steps and current_max > 7:
            # Generate move: 1-swap
            e1 = random.choice(list(edges))
            u1, v1 = e1
            
            # Remove e1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e1)
            
            # Add e2
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 != v2 and (min(u2,v2), max(u2,v2)) not in edges:
                    break
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add((min(u2,v2), max(u2,v2)))
            
            new_energy = calc_energy(adj, edges)
            delta_e = new_energy - current_energy
            
            accepted = False
            if delta_e < 0:
                accepted = True
            else:
                # Metropolis criterion
                if T > T_min:
                    prob = math.exp(-delta_e / T)
                    if random.random() < prob:
                        accepted = True
            
            if not accepted:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edges.add(e1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                edges.discard((min(u2,v2), max(u2,v2)))
            else:
                current_energy = new_energy
                current_max = max_book_fast(adj, edges)
                
                if current_max < best_max:
                    best_max = current_max
                    best_edges = edges.copy()
                    best_energy = current_energy
                    if best_max <= 7:
                        break

            T *= cooling_rate
            steps += 1

    if best_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for (u, v) in sorted(best_edges)]
# EVOLVE-BLOCK-END
