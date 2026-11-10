"""
Strategy: Maximum Weight Cycle Packing via ILP
Instead of greedy extraction, we:
1. Enumerate a diverse set of candidate cycles using randomized DFS.
2. Formulate an Integer Linear Program (ILP) to select a subset of these cycles
   that are edge-disjoint and maximize the total number of edges covered.
3. The remaining edges not covered by any selected cycle become single-edge pieces.
This uses scipy.optimize.milp for global optimization over the candidate set.
"""

import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
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
    
    def decompose_graph_ilp(edges, time_limit):
        if not edges:
            return []
        
        # 1. Build adjacency and edge list
        edge_list = [tuple(sorted(e)) for e in edges]
        edge_set = set(edge_list)
        adj = defaultdict(set)
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # 2. Enumerate candidate cycles
        # We use a randomized DFS to find many distinct cycles.
        # Limiting the number of candidates keeps the ILP size manageable.
        candidates = []
        seen_cycles = set()
        
        # Try to find cycles starting from each vertex with degree >= 2
        # We do this multiple times with different random seeds to get diversity
        max_candidates = 2000  # Cap on number of candidate cycles to keep ILP fast
        num_search_rounds = 50
        
        # Shuffle vertices to randomize start points
        all_vertices = list(adj.keys())
        random.shuffle(all_vertices)
        
        for _ in range(num_search_rounds):
            if time.time() > time_limit - 2:
                break
            if len(candidates) >= max_candidates:
                break
            
            # Pick a random start vertex with degree >= 2
            starts = [v for v in all_vertices if len(adj[v]) >= 2]
            if not starts:
                break
            
            start = random.choice(starts)
            
            # Randomized DFS to find a cycle
            path = [start]
            visited = {start}
            found = False
            
            def dfs(current, depth):
                nonlocal found
                if found or time.time() > time_limit - 2:
                    return
                if depth > 15: # Limit depth to keep cycles reasonable size for enumeration
                    return
                
                neighbors = list(adj[current])
                random.shuffle(neighbors)
                
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle_tuple = tuple(path)
                        # Normalize cycle representation to avoid duplicates (rotation/reversal)
                        # Simple normalization: start with smallest vertex
                        min_idx = cycle_tuple.index(min(cycle_tuple))
                        norm_cycle = tuple(cycle_tuple[min_idx:] + cycle_tuple[:min_idx])
                        # Also check reverse
                        rev_cycle = tuple(reversed(cycle_tuple))
                        min_idx_rev = rev_cycle.index(min(rev_cycle))
                        norm_rev = tuple(rev_cycle[min_idx_rev:] + rev_cycle[:min_idx_rev])
                        
                        key = min(norm_cycle, norm_rev)
                        if key not in seen_cycles:
                            seen_cycles.add(key)
                            candidates.append(list(cycle_tuple))
                            found = True
                            return
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited.discard(nb)
                        if found:
                            return
            
            dfs(start, 0)
            
        if not candidates:
            # No cycles found, return all edges as singles
            return [list(e) for e in edge_list]
        
        # 3. Formulate ILP
        # Variables: x_i for each candidate cycle i (binary)
        # Objective: Maximize sum(len(cycle_i) * x_i)
        # Constraints: For each edge e, sum(x_i for i where e in cycle_i) <= 1
        
        num_cycles = len(candidates)
        num_edges = len(edge_list)
        
        # Map edges to indices
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Build constraint matrix A (num_edges x num_cycles)
        # A[e, c] = 1 if edge e is in cycle c
        A = np.zeros((num_edges, num_cycles))
        for c_idx, cycle in enumerate(candidates):
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                ek = edge_key(u, v)
                if ek in edge_to_idx:
                    e_idx = edge_to_idx[ek]
                    A[e_idx, c_idx] = 1
        
        # Objective: maximize sum(len(cycle) * x)
        # milp minimizes, so we negate
        c_obj = -np.array([len(cycle) for cycle in candidates], dtype=float)
        
        # Constraints: A @ x <= 1
        # Bounds: 0 <= x <= 1 (and integrality)
        
        try:
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A, ub=1),
                integrality=np.ones(num_cycles),
                bounds=Bounds(lb=0, ub=1)
            )
            
            if res.status != 0:
                # Fallback to greedy if ILP fails
                return decompose_graph_greedy(edge_list, time_limit)
            
            selected_cycles = []
            for i in range(num_cycles):
                if res.x[i] > 0.5:
                    selected_cycles.append(candidates[i])
            
            # 4. Construct final piece list
            pieces = []
            used_edges = set()
            
            for cycle in selected_cycles:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    used_edges.add(edge_key(u, v))
            
            # Add remaining edges as singles
            for e in edge_list:
                ek = edge_key(e[0], e[1])
                if ek not in used_edges:
                    pieces.append(list(e))
            
            return pieces
            
        except Exception:
            # Fallback to greedy if ILP fails
            return decompose_graph_greedy(edge_list, time_limit)
    
    def decompose_graph_greedy(edge_list, time_limit):
        """Fallback greedy decomposition if ILP fails or is too slow."""
        remaining = set(tuple(sorted(e)) for e in edge_list)
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
            
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            best_cycle = None
            random.shuffle(candidates)
            
            for start in candidates[:10]:
                if time.time() > time_limit:
                    break
                
                path = [start]
                visited = {start}
                
                def dfs(current, depth):
                    if time.time() > time_limit or depth > 15:
                        return None
                    neighbors = list(adj[current])
                    random.shuffle(neighbors)
                    for nb in neighbors:
                        if nb == start and len(path) >= 3:
                            return path[:]
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            result = dfs(nb, depth + 1)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(nb)
                    return None
                
                cycle = dfs(start, 0)
                if cycle and (not best_cycle or len(cycle) > len(best_cycle)):
                    best_cycle = cycle
                    if len(best_cycle) >= 8:
                        break
            
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                if remaining:
                    u, v = list(remaining)[0]
                    commit_single_edge(u, v)
        
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        return pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Allocate time for this instance
        time_per_instance = min(40, deadline - time.time() - 10)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Try ILP with a time limit
        # ILP can be expensive, so we give it a chunk of time
        # If it fails or times out, we fall back to greedy
        trial_deadline = time.time() + time_per_instance
        
        pieces = decompose_graph_ilp(edges, trial_deadline)
        
        results.append(pieces)
    
    return results
