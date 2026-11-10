"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
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
    
    def get_2_core(adj, remaining):
        """
        Compute the 2-core of the graph defined by 'adj' and 'remaining'.
        Returns a set of vertices in the 2-core and a set of edges in the 2-core.
        Edges not in the 2-core are 'tree edges' and must be single pieces.
        """
        # We need to work on a copy of the degree information
        # Degrees are calculated based on 'remaining' edges
        degrees = defaultdict(int)
        for u, v in remaining:
            degrees[u] += 1
            degrees[v] += 1
        
        # Queue of vertices with degree < 2
        queue = [v for v in degrees if degrees[v] < 2]
        in_core = set(degrees.keys())
        
        while queue:
            v = queue.pop()
            if v not in in_core:
                continue
            # Remove v from the core
            in_core.discard(v)
            for nb in list(adj[v]):
                # Only consider edges that are still in 'remaining'
                ek = edge_key(v, nb)
                if ek in remaining:
                    remaining.discard(ek)
                    if nb in in_core:
                        degrees[nb] -= 1
                        if degrees[nb] < 2:
                            queue.append(nb)
        
        # Reconstruct the 2-core edges
        core_edges = set()
        core_adj = defaultdict(set)
        for u, v in remaining:
            if u in in_core and v in in_core:
                core_edges.add(edge_key(u, v))
                core_adj[u].add(v)
                core_adj[v].add(u)
        
        return in_core, core_edges, core_adj

    def find_cycle_dfs(adj, start, remaining, time_limit):
        """
        Find a cycle starting at 'start' using DFS.
        Returns the list of vertices in the cycle or None.
        """
        if time.time() > time_limit:
            return None
            
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            if time.time() > time_limit:
                return None
            if depth > 50: # Prevent infinite loops or extremely deep recursion in small graphs
                return None
                
            neighbors = [nb for nb in adj[current] if edge_key(current, nb) in remaining]
            random.shuffle(neighbors)
            
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    return path[:]
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(nb, depth + 1)
                    if result:
                        return result
                    path.pop()
                    visited.discard(nb)
            return None
        
        return dfs(start, 0)

    def decompose_graph(edges, time_limit, seed):
        """
        Decompose edges into cycles and single edges.
        1. Extract 2-core.
        2. Greedily find cycles in the core.
        3. Remaining edges are singles.
        """
        random.seed(seed)
        
        # Initial setup
        remaining = set(edge_key(u, v) for u, v in edges)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
            
        if not remaining:
            return []

        pieces = []
        
        # Step 1: Extract 2-core
        # The 2-core contains all cycles. Edges removed are trees/bridges.
        core_vertices, core_edges, core_adj = get_2_core(adj, remaining)
        
        # Edges not in core_edges are tree edges
        tree_edges = remaining - core_edges
        for u, v in tree_edges:
            pieces.append([u, v])
            
        # Now we only care about core_edges
        remaining = core_edges
        adj = core_adj
        
        # Step 2: Greedy cycle extraction from the core
        while remaining:
            if time.time() > time_limit:
                break
                
            # Find a vertex with degree >= 2 in the current remaining graph
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                # Should not happen if we maintain the 2-core property correctly,
                # but if it does, just take remaining edges as singles
                for u, v in list(remaining):
                    pieces.append([u, v])
                break
            
            # Pick a random candidate to start cycle search
            # We might try a few candidates if the first one fails
            random.shuffle(candidates)
            
            cycle_found = None
            for start in candidates[:3]: # Try up to 3 starts
                cycle = find_cycle_dfs(adj, start, remaining, time_limit)
                if cycle:
                    cycle_found = cycle
                    break
            
            if cycle_found:
                # Commit the cycle
                cycle_edges = set()
                for i in range(len(cycle_found)):
                    u = cycle_found[i]
                    v = cycle_found[(i + 1) % len(cycle_found)]
                    ek = edge_key(u, v)
                    cycle_edges.add(ek)
                    remaining.discard(ek)
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(cycle_found[:])
                
                # Optional: Re-extract 2-core after removing a cycle?
                # Removing a cycle might leave new tree parts.
                # For efficiency, we can just continue, but let's be safe and check degrees.
                # Actually, just checking candidates is enough. If a vertex drops to deg 1, it won't be a candidate.
                # But its neighbors might drop to deg 1.
                # The `candidates` list is rebuilt in the next iteration of the while loop.
                
            else:
                # No cycle found from these starts.
                # This is strange for a 2-core. It might be due to time limits or DFS limits.
                # Fallback: remove a random edge to break the structure and retry.
                # This is a heuristic to ensure progress.
                if remaining:
                    u, v = random.choice(list(remaining))
                    pieces.append([u, v])
                    remaining.discard(edge_key(u, v))
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        # Any remaining edges in the core (if we broke out early) become singles
        for u, v in list(remaining):
            pieces.append([u, v])
            
        return pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for this instance
        time_per_instance = min(45, deadline - time.time() - 10)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Run multiple trials with different random seeds
        num_trials = 30
        trial_time = time_per_instance / num_trials
        
        for t in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Set a specific seed for reproducibility within the trial
            random.seed(3000 + t)
            
            # Time limit for this specific trial
            trial_deadline = time.time() + trial_time
            
            pieces = decompose_graph(edges, trial_deadline, 3000 + t)
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
