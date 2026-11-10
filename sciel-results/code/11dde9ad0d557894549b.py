"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def compute_2core(adj):
        """Compute the 2-core of the graph (largest subgraph where all degrees >= 2)."""
        from collections import deque
        deg = {v: len(neigh) for v, neigh in adj.items()}
        queue = deque([v for v in deg if deg[v] < 2])
        removed = set()
        while queue:
            v = queue.popleft()
            if v in removed:
                continue
            removed.add(v)
            for nb in list(adj.get(v, set())):
                if nb not in removed:
                    deg[nb] = deg.get(nb, 0) - 1
                    if deg[nb] < 2:
                        queue.append(nb)
        return removed

    def find_longest_cycle_in_2core(adj, start, max_len, time_limit):
        """Find a long cycle in the 2-core using randomized DFS with pruning."""
        best_cycle = None
        best_len = 0
        
        # Use iterative deepening with random restarts
        for attempt in range(50):
            if time.time() > time_limit:
                break
            
            path = [start]
            visited = {start}
            
            def dfs(current, depth):
                nonlocal best_cycle, best_len
                if time.time() > time_limit:
                    return False
                if depth >= max_len:
                    return False
                
                neighbors = list(adj.get(current, set()))
                random.shuffle(neighbors)
                
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        return True
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        if dfs(nb, depth + 1):
                            return True
                        path.pop()
                        visited.discard(nb)
                return False
            
            dfs(start, 0)
            if best_cycle is not None:
                break
        
        return best_cycle

    def decompose_with_local_search(edges, time_limit, seed):
        """Decompose using 2-core + greedy cycle extraction + local search optimization."""
        random.seed(seed)
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))
        
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
        
        # Greedy phase: repeatedly find long cycles in the 2-core
        while remaining:
            if time.time() > time_limit:
                break
            
            # Compute 2-core to focus on cyclic parts
            removed = compute_2core(adj)
            core_vertices = [v for v in adj if v not in removed]
            
            if not core_vertices:
                # No cycles left
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            # Find a long cycle in the 2-core
            best_cycle = None
            best_len = 0
            
            # Try multiple starting vertices
            random.shuffle(core_vertices)
            for start in core_vertices[:10]:
                if time.time() > time_limit:
                    break
                cycle = find_longest_cycle_in_2core(adj, start, 30, time_limit)
                if cycle is not None and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len >= 15:
                        break
            
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                # Fallback: remove a single edge
                u, v = list(remaining)[0]
                commit_single_edge(u, v)
        
        # Remaining edges as singles
        for u, v in list(remaining):
            commit_single_edge(u, v)
        
        # Local search: try to merge two cycles or replace a cycle + edge with a longer cycle
        # This is a simplified 2-opt style improvement
        if time.time() < time_limit - 5:
            improved = True
            while improved and time.time() < time_limit - 5:
                improved = False
                # Try to find a longer cycle by combining edges from existing pieces
                # Simple heuristic: for each pair of adjacent pieces (sharing a vertex),
                # try to find a cycle using edges from both
                for i in range(len(pieces)):
                    if time.time() > time_limit - 5:
                        break
                    if len(pieces[i]) < 3:
                        continue
                    for j in range(len(pieces)):
                        if i == j:
                            continue
                        # Check if pieces share a vertex
                        set_i = set(pieces[i])
                        set_j = set(pieces[j])
                        common = set_i & set_j
                        if not common:
                            continue
                        # Try to find a cycle using edges from both pieces
                        # This is complex; skip for now
                        pass
        
        return pieces

    def decompose_fast(edges, time_limit, seed):
        """Fast decomposition: 2-core + greedy longest cycle."""
        random.seed(seed)
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))
        
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
        
        while remaining:
            if time.time() > time_limit:
                break
            
            removed = compute_2core(adj)
            core_vertices = [v for v in adj if v not in removed]
            
            if not core_vertices:
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            best_cycle = None
            best_len = 0
            
            random.shuffle(core_vertices)
            for start in core_vertices[:8]:
                if time.time() > time_limit:
                    break
                cycle = find_longest_cycle_in_2core(adj, start, 25, time_limit)
                if cycle is not None and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len >= 12:
                        break
            
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                u, v = list(remaining)[0]
                commit_single_edge(u, v)
        
        for u, v in list(remaining):
            commit_single_edge(u, v)
        
        return pieces

    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        best_pieces = None
        best_count = float('inf')
        
        time_per_instance = min(40, deadline - time.time() - 15)
        if time_per_instance < 5:
            time_per_instance = 5
        
        num_trials = 30
        trial_time = time_per_instance / num_trials
        
        for t in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            seed_val = 3000 + t
            trial_deadline = time.time() + trial_time
            
            pieces = decompose_fast(edges, trial_deadline, seed_val)
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
