import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i + 1]
        edges.add((min(u, v), max(u, v)))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return sorted(edges)

def _find_all_cycles(adj, n, max_len=8, max_count=500):
    """
    Find all simple cycles up to max_len in the graph defined by adj.
    Returns a list of cycles, where each cycle is a list of vertices.
    """
    cycles = []
    # To avoid duplicates, we only start DFS from the smallest vertex in the cycle
    # and ensure the path is sorted in a specific way? 
    # Simpler: Just find all, then deduplicate.
    # Given small graph, deduplication is fine.
    
    def dfs(start, current, path, visited):
        if len(cycles) >= max_count:
            return
        if len(path) > 1 and current == start:
            if len(path) >= 3 and len(path) <= max_len:
                # Canonical form: rotate so min vertex is first
                min_idx = path.index(min(path))
                canonical = tuple(path[min_idx:] + path[:min_idx])
                cycles.append(list(canonical))
            return
        
        if len(path) >= max_len:
            return
            
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 3:
                # Close cycle
                if len(path) <= max_len:
                    min_idx = path.index(min(path))
                    canonical = tuple(path[min_idx:] + path[:min_idx])
                    cycles.append(list(canonical))
            elif neighbor not in visited and neighbor > start:
                # Only visit neighbors greater than start to avoid symmetric cycles
                # This is a standard trick to find each cycle exactly once
                # if the graph is undirected and we fix the start as the min node.
                # But wait, if we fix start as min, we must ensure we don't skip valid cycles
                # where start is not the global min of the path? 
                # Actually, if we enforce neighbor > start for all steps, we only find cycles 
                # where 'start' is the minimum vertex in the cycle.
                # This covers all cycles exactly once.
                visited.add(neighbor)
                dfs(start, neighbor, path + [neighbor], visited)
                visited.remove(neighbor)

    for start in range(n):
        if not adj[start]:
            continue
        visited = {start}
        dfs(start, start, [start], visited)
        
    # Deduplicate just in case
    unique_cycles = []
    seen = set()
    for c in cycles:
        # c is already canonical (starts with min, but direction? )
        # Cycle [1,2,3] is same as [1,3,2].
        # We should canonicalize direction too.
        # Reverse and take min tuple
        rev = c[::-1]
        key = min(tuple(c), tuple(rev))
        if key not in seen:
            seen.add(key)
            unique_cycles.append(list(key))
            
    return unique_cycles

def _solve_set_packing(cycles, edge_set, n):
    """
    Find max weight set of edge-disjoint cycles.
    Weight of cycle is len(cycle) - 1.
    """
    if not cycles:
        return [], 0
    
    # Map edge to index? Or just use set.
    # Number of cycles is small (<500).
    # We can use backtracking with pruning.
    
    # Sort cycles by weight (length) descending to find good solutions early
    cycles_with_weight = []
    for c in cycles:
        w = len(c) - 1
        cycles_with_weight.append((w, c))
    cycles_with_weight.sort(key=lambda x: x[0], reverse=True)
    
    best_weight = 0
    best_set = []
    
    def backtrack(idx, current_weight, current_set, used_edges):
        nonlocal best_weight, best_set
        
        if current_weight > best_weight:
            best_weight = current_weight
            best_set = current_set[:]
            
        if idx == len(cycles_with_weight):
            return
            
        # Pruning: if current_weight + sum of all remaining weights <= best_weight, prune
        # We can precompute suffix sums of weights
        # But simple check: if current_weight + (len(cycles_with_weight) - idx) * max_possible_avg <= best_weight
        # Let's just rely on the order and small N.
        
        w, c = cycles_with_weight[idx]
        
        # Option 1: Include cycle c
        # Check if edges are disjoint
        can_include = True
        cycle_edges = []
        for i in range(len(c)):
            u = c[i]
            v = c[(i+1) % len(c)]
            e = (min(u,v), max(u,v))
            if e in used_edges:
                can_include = False
                break
            cycle_edges.append(e)
            
        if can_include:
            new_used = used_edges | set(cycle_edges)
            current_set.append(c)
            backtrack(idx + 1, current_weight + w, current_set, new_used)
            current_set.pop()
            
        # Option 2: Exclude cycle c
        backtrack(idx + 1, current_weight, current_set, used_edges)

    backtrack(0, 0, [], set())
    
    return best_set, best_weight

def _repack_subgraph(edges_sub, n, time_limit=0.1):
    """
    Decompose edges_sub into min pieces.
    Returns list of pieces.
    """
    if not edges_sub:
        return []
        
    # Build adjacency for subgraph
    adj = [set() for _ in range(n)]
    for u, v in edges_sub:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find all cycles in this subgraph
    # Limit max_len to avoid explosion, e.g., 10
    cycles = _find_all_cycles(adj, n, max_len=10, max_count=200)
    
    if not cycles:
        return [list(e) for e in edges_sub]
        
    # Solve set packing
    best_cycles, _ = _solve_set_packing(cycles, edges_sub, n)
    
    # Construct pieces
    pieces = []
    used_edges = set()
    for c in best_cycles:
        # Add cycle
        pieces.append(c)
        for i in range(len(c)):
            u = c[i]
            v = c[(i+1) % len(c)]
            e = (min(u,v), max(u,v))
            used_edges.add(e)
            
    # Add remaining edges as singletons
    for e in edges_sub:
        if e not in used_edges:
            pieces.append(list(e))
            
    return pieces

def _greedy_decompose(edges, n, rng):
    """
    Initial greedy decomposition.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining = set(edges)
    
    while remaining:
        # Find a cycle
        cycle = None
        # Try to find a cycle starting from high degree nodes
        candidates = [v for v in range(n) if adj[v]]
        candidates.sort(key=lambda v: len(adj[v]), reverse=True)
        rng.shuffle(candidates) # Add randomness
        
        for start in candidates:
            if not adj[start]:
                continue
            # DFS to find cycle
            stack = [(start, [start])]
            visited = {start}
            while stack:
                node, path = stack.pop()
                for neighbor in list(adj[node]):
                    if neighbor == start and len(path) >= 3:
                        cycle = path[:]
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))
                if cycle:
                    break
            if cycle:
                break
                
        if cycle:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = (min(u,v), max(u,v))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            # No cycle, take one edge
            e = next(iter(remaining))
            pieces.append(list(e))
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            
    return pieces

def _solve_one(edges, n, deadline):
    if not edges:
        return []
        
    rng = random.Random(42)
    
    # 1. Initial Greedy
    best_pieces = _greedy_decompose(edges, n, rng)
    best_count = len(best_pieces)
    
    # 2. Local Search
    # Represent solution as list of pieces
    # We need to map pieces to edge sets for fast removal
    
    def get_piece_edges(p):
        if len(p) == 2:
            return {(min(p[0],p[1]), max(p[0],p[1]))}
        else:
            es = set()
            for i in range(len(p)):
                u = p[i]
                v = p[(i+1) % len(p)]
                es.add((min(u,v), max(u,v)))
            return es
            
    while time.time() < deadline - 0.1:
        # Select a random subset of pieces to repack
        # Prefer cycles (len > 2)
        cycle_indices = [i for i, p in enumerate(best_pieces) if len(p) > 2]
        
        if not cycle_indices:
            break
            
        # Pick 1 to 3 cycles
        k = min(len(cycle_indices), rng.choice([1, 1, 2, 3]))
        selected_indices = rng.sample(cycle_indices, k)
        
        # Collect edges
        edges_to_repack = set()
        for idx in selected_indices:
            edges_to_repack |= get_piece_edges(best_pieces[idx])
            
        # Remove these pieces
        # Create a new list without these indices
        new_pieces = [p for i, p in enumerate(best_pieces) if i not in selected_indices]
        
        # Repack
        repacked = _repack_subgraph(edges_to_repack, n, time_limit=0.05)
        
        # Check if better
        if len(repacked) < k:
            # Accept
            new_pieces.extend(repacked)
            best_pieces = new_pieces
            best_count = len(best_pieces)
        # If equal, we might accept to diversify, but let's stick to strict improvement for now
        # Or accept with small probability to escape local minima?
        # Given the exact solver in repack, it's likely optimal for that subgraph.
        # So strict improvement is good.

    return best_pieces

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
