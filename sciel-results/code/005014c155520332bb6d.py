import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Generate the three fixed instance graphs
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def decompose(edges, time_limit):
        """
        Decompose edges into cycles and single edges using Multi-Start Greedy.
        """
        if not edges:
            return []

        # Pre-compute sorted edge list for consistency
        edge_list = sorted(edges)
        best_pieces = None
        best_count = float('inf')
        
        # We will run multiple greedy passes with different random seeds
        # to diversify the search and escape local optima.
        pass_seed = 0
        
        while time.time() < time_limit:
            # Estimate time per pass. 
            # A greedy pass on N=30 should be fast (<1s) if DFS is pruned.
            # Let's allocate 2.0s per pass as a soft limit for the pass logic.
            pass_deadline = time.time() + 2.0
            if pass_deadline > time_limit:
                pass_deadline = time_limit
            
            pieces = greedy_pass(edges, pass_seed, pass_deadline)
            
            if pieces is not None:
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
                    # If we found a solution with no single edges (all cycles), 
                    # it might be optimal or near-optimal. 
                    # Minimum pieces is at least |V|/3 roughly? 
                    # Actually, if all edges are covered by cycles, 
                    # we can't do better than the number of cycles.
                    # We can't easily know the lower bound, so we continue.
            
            pass_seed += 1
            
            # If we are running out of time, break
            if time.time() > time_limit - 5.0:
                break

        if best_pieces is None:
            # Fallback: just return all edges as single pieces
            return [[u, v] for u, v in edges]
            
        return best_pieces

    def greedy_pass(edges, seed, deadline):
        """
        Run a single greedy decomposition pass.
        """
        rng = random.Random(seed)
        
        # Build adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        # Shuffle neighbors to diversify the DFS path
        # We store the shuffled lists to use in DFS
        adj_shuffled = {u: list(v) for u, v in adj.items()}
        for u in adj_shuffled:
            rng.shuffle(adj_shuffled[u])
            
        # Also shuffle the order in which we pick start nodes
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            for u, v in sorted(remaining_edges):
                pieces.append([u, v])
            return pieces
        
        rng.shuffle(candidates)
        
        def get_edge(u, v):
            return (min(u, v), max(u, v))

        def remove_cycle_edges(cycle):
            n = len(cycle)
            if n < 3:
                return
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                e = get_edge(u, v)
                if e in remaining_edges:
                    remaining_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    # Update shuffled adj list
                    if v in adj_shuffled[u]:
                        adj_shuffled[u].remove(v)
                    if u in adj_shuffled[v]:
                        adj_shuffled[v].remove(u)

        def find_longest_cycle(start, max_time):
            """
            Find the longest cycle starting and ending at 'start' using DFS.
            """
            best_cycle = None
            best_len = 0
            visited = {start}
            path = [start]
            
            def dfs(node, current_len):
                nonlocal best_cycle, best_len
                if time.time() > max_time:
                    return
                
                # Check if we can close the cycle
                if current_len >= 3:
                    if node in adj[start]:
                        if current_len > best_len:
                            best_len = current_len
                            best_cycle = path[:]
                
                # Extend path
                # Use shuffled neighbors
                for neighbor in adj_shuffled[node]:
                    if neighbor == start:
                        continue
                    if neighbor in visited:
                        continue
                    
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, current_len + 1)
                    path.pop()
                    visited.discard(neighbor)
                    
                    if time.time() > max_time:
                        return

            dfs(start, 1)
            return best_cycle

        def find_any_cycle():
            """
            Fallback: Find any cycle, preferably a triangle.
            """
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                neighbors_u = list(adj[u])
                for i in range(len(neighbors_u)):
                    v = neighbors_u[i]
                    for w in neighbors_u[i+1:]:
                        if w in adj[v]:
                            return [u, v, w]
            return None

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > deadline:
                break

            search_deadline = time.time() + 0.3
            if deadline - time.time() < 0.3:
                search_deadline = deadline

            best_cycle = None
            best_len = 0

            # Re-evaluate candidates? 
            # The set of candidates with degree >= 2 changes.
            # We can just iterate through the original shuffled list 
            # and skip those with degree < 2.
            
            # To speed up, we can limit the number of start nodes we try per iteration
            # The current program tries top 5 by degree. 
            # Here, we try the first few from the shuffled list that are still valid.
            
            starts_tried = 0
            for start in candidates:
                if time.time() > search_deadline:
                    break
                
                if len(adj[start]) < 2:
                    continue
                
                starts_tried += 1
                if starts_tried > 5:
                    break
                
                cycle = find_longest_cycle(start, search_deadline)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    if best_len >= 10:
                        break

            if best_cycle:
                pieces.append(best_cycle)
                remove_cycle_edges(best_cycle)
            else:
                cycle = find_any_cycle()
                if cycle:
                    pieces.append(cycle)
                    remove_cycle_edges(cycle)
                else:
                    for u, v in sorted(remaining_edges):
                        pieces.append([u, v])
                    remaining_edges.clear()
                    break

        for u, v in sorted(remaining_edges):
            pieces.append([u, v])

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
