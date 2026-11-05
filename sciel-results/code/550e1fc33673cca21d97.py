import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    best_result = None
    best_n = float('inf')

    def _try_record(n, edges):
        nonlocal best_result, best_n
        if n < best_n:
            best_n = n
            best_result = (n, edges)

    # --- LUW fallback (from seed) ---
    def _luw_graph(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Fast girth verification (numpy BFS) ---
    def _check_girth(adj, n, g_limit):
        max_depth = (g_limit - 1) // 2
        adj_np = np.array(adj, dtype=np.int32)
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                neighbors = adj_np[u]
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- Petersen Base Graph Definition ---
    # Vertices 0..9
    # Edges: (0,1), (1,2), (2,3), (3,4), (4,0) [Outer 5-cycle]
    #        (0,5), (1,6), (2,7), (3,8), (4,9) [Spokes]
    #        (5,7), (7,9), (9,6), (6,8), (8,5) [Inner 5-cycle]
    PET_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)
    ]
    PET_VERTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Adjacency list for base graph (indices into PET_EDGES)
    base_adj = [[] for _ in range(10)]
    for i, (u, v) in enumerate(PET_EDGES):
        base_adj[u].append(i)
        base_adj[v].append(i)

    def _voltage_graph_petersen(voltages, N):
        """
        Construct graph from Petersen base with voltages over Z_N.
        voltages: list of 15 ints (0..N-1) corresponding to PET_EDGES.
        Returns (n, edges) or (None, None) if invalid (loops/parallel).
        """
        n_total = 10 * N
        adj = [[] for _ in range(n_total)]
        edge_set = set()

        for i, (u, v) in enumerate(PET_EDGES):
            vol = voltages[i]
            for a in range(N):
                i_idx = u * N + a
                b = (a + vol) % N
                j_idx = v * N + b
                
                if i_idx == j_idx:
                    return None, None
                if (i_idx, j_idx) in edge_set or (j_idx, i_idx) in edge_set:
                    return None, None
                
                edge_set.add((i_idx, j_idx))
                adj[i_idx].append(j_idx)
                adj[j_idx].append(i_idx)
        
        # Verify 3-regular
        for i in range(n_total):
            if len(adj[i]) != 3:
                return None, None
        
        edges = sorted(edge_set)
        return n_total, edges

    def _count_short_cycles(voltages, N, max_len=7):
        """
        Count non-backtracking walks of length L < g that return to start.
        For girth g, we need no cycles of length < g.
        Since the graph is 3-regular, we can check cycles up to length g-1.
        For g=16, we check up to 15.
        However, checking all cycles is expensive.
        We check for cycles of length L where L < g.
        A cycle of length L corresponds to a closed non-backtracking walk of length L.
        
        Optimization: We only need to check even lengths? No, odd cycles are possible.
        But in a bipartite graph, only even cycles. Petersen is not bipartite.
        
        Strategy: BFS from each vertex? No, that's O(N^2 * g).
        Better: Count closed walks of length L.
        Let A be the adjacency matrix.
        Number of closed walks of length L is trace(A^L).
        But we need non-backtracking.
        
        Alternative: For small N (e.g., 50), N_total = 500.
        We can do a BFS from each vertex up to depth (g-1)/2?
        Actually, for girth >= g, the BFS tree from any vertex must be a tree up to depth (g-1)/2.
        If we find a cycle of length < g, it will be detected.
        
        Let's use the BFS check but only count violations.
        """
        n_total = 10 * N
        adj = [[] for _ in range(n_total)]
        for i, (u, v) in enumerate(PET_EDGES):
            vol = voltages[i]
            for a in range(N):
                i_idx = u * N + a
                b = (a + vol) % N
                j_idx = v * N + b
                # Skip loops/parallel for speed in fitness, assume valid structure mostly
                if i_idx != j_idx:
                    adj[i_idx].append(j_idx)
                    adj[j_idx].append(i_idx)
        
        violations = 0
        max_depth = (g - 1) // 2 # e.g., 7 for g=16
        
        for start in range(n_total):
            dist = [-1] * n_total
            parent = [-1] * n_total
            dist[start] = 0
            queue = deque([start])
            found_cycle = False
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                neighbors = adj[u]
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            violations += 1
                            found_cycle = True
                            # Break early for this start vertex to save time
                            break
                if found_cycle:
                    break
        
        return violations

    def _ga_search(N, time_limit):
        """
        Run GA to find voltage assignment with 0 short cycles for base N.
        """
        n_edges = 15
        pop_size = 50
        max_generations = 100
        mutation_rate = 0.2
        crossover_rate = 0.8
        
        # Initialize population
        population = [random.randint(0, N - 1) for _ in range(n_edges)]
        pop = [population[:]]
        
        # Evaluate initial fitness
        fitnesses = [_count_short_cycles(ind, N) for ind in pop]
        
        best_fitness = min(fitnesses)
        if best_fitness == 0:
            return pop[0]
            
        for gen in range(max_generations):
            if time.time() > time_limit:
                break
                
            # Sort by fitness
            sorted_pop = sorted(zip(fitnesses, pop), key=lambda x: x[0])
            fitnesses = [f for f, _ in sorted_pop]
            pop = [p for _, p in sorted_pop]
            
            if fitnesses[0] == 0:
                return pop[0]
                
            new_pop = []
            # Elitism
            new_pop.append(pop[0][:])
            new_pop.append(pop[1][:])
            
            while len(new_pop) < pop_size:
                if random.random() < crossover_rate:
                    # Tournament selection
                    idx1 = random.choice(range(3))
                    idx2 = random.choice(range(3))
                    parent1 = pop[idx1]
                    parent2 = pop[idx2]
                    
                    # Uniform crossover
                    child = [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]
                else:
                    idx1 = random.choice(range(5))
                    child = pop[idx1][:]
                
                # Mutation
                for i in range(n_edges):
                    if random.random() < mutation_rate:
                        child[i] = random.randint(0, N - 1)
                
                new_pop.append(child)
            
            pop = new_pop
            fitnesses = [_count_short_cycles(ind, N) for ind in pop]
            
        return pop[0] if fitnesses[0] < best_fitness else None

    # --- Main Logic ---
    if k == 3 and g == 16:
        moore_bound = 512
        
        # We know N_total = 10 * N.
        # We want N_total >= 512 => N >= 51.2 => N >= 52.
        # Try N from 52 upwards.
        # Time budget: 115s.
        
        # Phase 1: Try small N values with GA
        # N=52 -> 520 vertices
        # N=53 -> 530 vertices
        # ...
        
        # We will try N in range 52 to 100.
        # For each N, run GA for a fixed time slice.
        
        # If we find a valid graph, record it and stop (or try smaller? No, we go up from min).
        # Since we start from min, the first valid one is the best for that N.
        # We should check N=52, 53, 54... and return the first one found?
        # Yes, because n=10*N is minimized by smallest N.
        
        # However, GA might fail for N=52 but succeed for N=53.
        # We should allocate time to each N.
        
        N_start = 52
        N_end = 120
        
        found = False
        for N in range(N_start, N_end + 1):
            if time.time() > deadline - 10:
                break
            
            # Time per N: give more time to smaller N?
            # Or equal time?
            # Let's give 2 seconds per N initially, increasing if not found.
            time_slice = 2.0
            limit = time.time() + time_slice
            
            result_volt = _ga_search(N, limit)
            
            if result_volt is not None:
                # Verify strictly
                n_c, edges_c = _voltage_graph_petersen(result_volt, N)
                if n_c is not None:
                    adj = [[] for _ in range(n_c)]
                    for u, v in edges_c:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _check_girth(adj, n_c, g):
                        _try_record(n_c, edges_c)
                        found = True
                        # Since we are iterating N upwards, this is the smallest n so far.
                        # We can break if we are confident, but maybe a smaller N was skipped?
                        # No, we started from 52.
                        # So this is the optimal for this family.
                        break

        # Phase 2: If GA failed, try random search for a few N values?
        # Or fallback to LUW.
        
        if not found and time.time() < deadline - 5:
            # Try a few random samples for N=52..60
            for N in range(52, 65):
                if time.time() > deadline - 5:
                    break
                for _ in range(50):
                    volt = [random.randint(0, N - 1) for _ in range(15)]
                    n_c, edges_c = _voltage_graph_petersen(volt, N)
                    if n_c is None:
                        continue
                    adj = [[] for _ in range(n_c)]
                    for u, v in edges_c:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _check_girth(adj, n_c, g):
                        _try_record(n_c, edges_c)
                        found = True
                        break
                if found:
                    break

        # Phase 3: LUW fallback
        if not found and time.time() < deadline - 2:
            n_luw, edges_luw = _luw_graph(k, g)
            _try_record(n_luw, edges_luw)

    else:
        # Generic fallback
        n_luw, edges_luw = _luw_graph(k, g)
        _try_record(n_luw, edges_luw)

    if best_result is not None:
        return best_result

    # Absolute fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw

