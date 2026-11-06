import time
import random
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        remaining_edges = set(edges)
        pieces = []
        
        # Precompute initial degrees for sorting heuristics
        def get_adjacency(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        # Helper to find a long cycle in the current graph
        # We use a randomized DFS approach to find a long cycle, 
        # trying to avoid getting stuck in short cycles.
        def find_long_cycle(edge_set, max_tries=50, max_depth=30):
            if not edge_set:
                return None
            
            adj = get_adjacency(edge_set)
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree descending to start from "hub" vertices
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            
            # Try a few random starting points and DFS paths
            # To keep it fast, we limit the number of DFS explorations
            for _ in range(max_tries):
                start = random.choice(candidates[:min(10, len(candidates))])
                
                # Iterative DFS to find a cycle
                # We want to find a cycle that is as long as possible.
                # A simple heuristic: DFS and try to close the cycle when possible,
                # but prefer longer paths if possible.
                
                stack = [(start, [start], {start}, 0)]
                found_in_this_try = False
                while stack and not found_in_this_try:
                    current, path, visited, depth = stack.pop()
                    
                    if depth >= max_depth:
                        continue
                        
                    # Check if we can close a cycle
                    if len(path) >= 3 and start in adj[current]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                        # We don't break immediately because a longer cycle might exist
                        # but for speed, we often take the first good one or the best found in this try.
                        # Let's continue to see if we can find something even longer in this DFS tree?
                        # Actually, standard DFS finds *a* cycle. To find a *long* one is hard.
                        # Heuristic: Prefer extending the path if possible, only close if stuck.
                        
                    # Extend
                    neighbors = list(adj[current])
                    # Sort neighbors by degree to explore high-degree nodes first? 
                    # Or randomize to avoid bias?
                    random.shuffle(neighbors)
                    
                    for neighbor in neighbors:
                        if neighbor == start:
                            # Only close if we have a valid cycle length
                            if len(path) >= 3:
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_cycle = list(path)
                                    found_in_this_try = True
                                    break
                            continue
                            
                        if neighbor not in visited:
                            new_path = path + [neighbor]
                            new_visited = visited | {neighbor}
                            stack.append((neighbor, new_path, new_visited, depth + 1))
                            
            return best_cycle

        # Main loop: Peel off cycles
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            cycle = find_long_cycle(remaining_edges)
            
            if cycle:
                pieces.append(cycle)
                # Remove cycle edges
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
                u, v = cycle[-1], cycle[0]
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)
            else:
                # No cycle found. This means the remaining graph is a forest (acyclic).
                # All remaining edges must be singletons.
                for e in remaining_edges:
                    pieces.append(list(e))
                remaining_edges.clear()
                break

        # If time ran out, add remaining edges as singletons to ensure validity
        if remaining_edges:
            for e in remaining_edges:
                pieces.append(list(e))

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)

        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)

        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
