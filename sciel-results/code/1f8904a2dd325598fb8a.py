"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    def soft_max_score(adj, deg, beta=10.0):
        """Compute a smooth surrogate for max triangle degree sum."""
        # Collect all triangle degree sums
        sums = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    sums.append(deg[i] + deg[j] + deg[k_idx])
                    common &= common - 1
        if not sums:
            return 0
        max_s = max(sums)
        # Log-sum-exp for soft max
        shifted = [s - max_s for s in sums]
        log_sum = math.log(sum(math.exp(beta * s) for s in shifted)) / beta
        return max_s + log_sum
    
    def simulated_annealing(initial_edges, max_iters=200000, t_start=2.0, t_end=0.01):
        """Simulated annealing with full edge swaps."""
        current = initial_edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        edge_set = set()
        for u, v in current:
            edge_set.add((min(u,v), max(u,v)))
        
        best_edges = current[:]
        best_score = current_score
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Random edge swap: remove one edge, add another
            u1, v1 = random.choice(current)
            u1, v1 = min(u1, v1), max(u1, v1)
            
            # Pick a random non-edge
            for _ in range(20):
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) == (u1, v1):
                    continue
                if (u2, v2) in edge_set:
                    continue
                break
            else:
                continue
            
            # Perform swap
            new_adj = adj[:]
            new_deg = deg[:]
            
            # Remove edge (u1, v1)
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            
            # Add edge (u2, v2)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            new_score = compute_score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
                # Accept
                adj = new_adj
                deg = new_deg
                current_score = new_score
                edge_set.discard((u1, v1))
                edge_set.add((u2, v2))
                
                # Update edge list
                current = [e for e in current if (min(e[0],e[1]), max(e[0],e[1])) != (u1, v1)]
                current.append([u2, v2])
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = current[:]
            
            t *= cooling
        
        return best_edges, best_score
    
    # Strategy: K_{6,10} + 5 edges in the 10-vertex part
    base_edges = [[u, v] for u in range(6) for v in range(6, 16)]
    
    # Start with a matching in the large part
    initial_internal = [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    initial_edges = base_edges + initial_internal
    initial_score = compute_score(initial_edges)
    
    best_edges = initial_edges[:]
    best_score = initial_score
    
    # Run simulated annealing from initial
    if time.time() < deadline - 15.0:
        sa_edges, sa_score = simulated_annealing(initial_edges, max_iters=150000, t_start=1.5, t_end=0.005)
        if sa_score < best_score:
            best_score = sa_score
            best_edges = sa_edges[:]
    
    # Try multiple random restarts with different initial structures
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a random graph with 65 edges
        random.shuffle(all_possible_edges)
        random_edges = [[u, v] for u, v in all_possible_edges[:target_edges]]
        
        if not is_valid(random_edges):
            continue
        
        random_score = compute_score(random_edges)
        if random_score < best_score:
            best_score = random_score
            best_edges = random_edges[:]
        
        # Simulated annealing from random start
        if time.time() < deadline - 10.0:
            sa_edges2, sa_score2 = simulated_annealing(random_edges, max_iters=50000, t_start=1.0, t_end=0.005)
            if sa_score2 < best_score:
                best_score = sa_score2
                best_edges = sa_edges2[:]
    
    # Final polish: local search on best result
    if time.time() < deadline - 8.0 and best_score > 23:
        # Try more aggressive local search
        final_edges, final_score = simulated_annealing(best_edges, max_iters=100000, t_start=0.5, t_end=0.001)
        if final_score < best_score:
            best_score = final_score
            best_edges = final_edges[:]
    
    # Final verification
    if not is_valid(best_edges):
        best_edges = base_edges + [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        best_score = compute_score(best_edges)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
