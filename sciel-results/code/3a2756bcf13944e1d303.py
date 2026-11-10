import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s for cleanup/return

    def elapsed():
        return time.time() - start_time

    def _count_regular_6(adj, n):
        """Count regular induced subgraphs of size 6."""
        if n < 6:
            return 0
        
        count = 0
        verts = range(n)
        # Precompute bit masks for vertices? No, adj is already bitmask.
        for comb in combinations(verts, 6):
            mask = 0
            # Unroll for speed?
            mask |= (1 << comb[0])
            mask |= (1 << comb[1])
            mask |= (1 << comb[2])
            mask |= (1 << comb[3])
            mask |= (1 << comb[4])
            mask |= (1 << comb[5])
            
            deg0 = (adj[comb[0]] & mask).bit_count()
            
            if (adj[comb[1]] & mask).bit_count() != deg0:
                continue
            if (adj[comb[2]] & mask).bit_count() != deg0:
                continue
            if (adj[comb[3]] & mask).bit_count() != deg0:
                continue
            if (adj[comb[4]] & mask).bit_count() != deg0:
                continue
            if (adj[comb[5]] & mask).bit_count() != deg0:
                continue
            
            count += 1
        return count

    def _is_valid(adj, n):
        return _count_regular_6(adj, n) == 0

    def _flip_edge(adj, u, v):
        bit_v = 1 << v
        bit_u = 1 << u
        if adj[u] & bit_v:
            adj[u] &= ~bit_v
            adj[v] &= ~bit_u
        else:
            adj[u] |= bit_v
            adj[v] |= bit_u

    def _to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def _solve_sa(n, time_budget, seed):
        rng = random.Random(seed)
        target_time = time.time() + time_budget
        
        # Initial random graph
        p = 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = _count_regular_6(adj, n)
        if current_cost == 0:
            return adj

        best_adj = adj[:]
        best_cost = current_cost

        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        # Precompute list of edges for faster random access?
        # Or just random u, v.
        
        while time.time() < target_time:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            
            _flip_edge(adj, u, v)
            
            new_cost = _count_regular_6(adj, n)
            delta = new_cost - current_cost
            
            if delta <= 0 or rng.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                    if best_cost == 0:
                        return best_adj
            else:
                _flip_edge(adj, u, v)
            
            T *= cooling
            if T < min_T:
                T = 1.0
                # Random restart
                if rng.random() < 0.2:
                    adj = [0] * n
                    for i in range(n):
                        for j in range(i + 1, n):
                            if rng.random() < 0.5:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                    current_cost = _count_regular_6(adj, n)
                    if current_cost == 0:
                        return adj
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj[:]

        return best_adj if best_cost == 0 else None

    best_graph = None
    best_n = 0

    # Try n=20
    t0 = time.time()
    budget_20 = min(130, deadline - t0 - 5)
    if budget_20 > 5:
        # Use all budget for a single long SA run, or split?
        # Splitting allows diversity.
        # Let's try 5 seeds.
        for seed in range(5):
            if time.time() > t0 + budget_20 or elapsed() > deadline - 5:
                break
            per_seed_budget = (t0 + budget_20 - time.time()) / (5 - seed)
            if per_seed_budget < 1:
                break
            g = _solve_sa(20, per_seed_budget, seed)
            if g is not None:
                if _is_valid(g, 20):
                    best_graph = g
                    best_n = 20
                    break

    if best_n < 19:
        t1 = time.time()
        budget_19 = min(10, deadline - t1 - 5)
        if budget_19 > 5:
            for seed in range(3):
                if time.time() > t1 + budget_19 or elapsed() > deadline - 5:
                    break
                per_seed_budget = (t1 + budget_19 - time.time()) / (3 - seed)
                if per_seed_budget < 1:
                    break
                g = _solve_sa(19, per_seed_budget, seed)
                if g is not None and _is_valid(g, 19):
                    best_graph = g
                    best_n = 19
                    break

    if best_graph is None:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': _to_edges(best_graph, best_n)}
