import time
import random
from collections import defaultdict
import sys

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def find_all_cycles_up_to_len(adj, max_len):
        """
        Find all simple cycles up to max_len.
        Returns a list of cycles (lists of vertices).
        """
        cycles = []
        n = max(adj.keys()) + 1 if adj else 0
        # To avoid duplicates, we can enforce that the smallest vertex in the cycle is the start.
        # Or just deduplicate at the end.
        
        def dfs(start, current, path, visited):
            if len(path) > max_len:
                return
            if len(path) >= 3:
                if current in adj[start]:
                    # Found a cycle
                    # Canonicalize: rotate so min vertex is first
                    cycle = list(path)
                    min_idx = cycle.index(min(cycle))
                    canonical_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                    # Check if we should add it
                    # We can add a tuple representation to a set to dedupe
                    cycles.append(canonical_cycle)
            
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    continue # Already handled above
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        for start in adj.keys():
            # Optimization: only start from the smallest vertex in the potential cycle?
            # Hard to know. Just run from all and dedupe.
            # To reduce work, only start from vertices with degree >= 2
            if len(adj[start]) < 2:
                continue
            visited = {start}
            path = [start]
            dfs(start, start, path, visited)
        
        # Deduplicate
        unique_cycles = list(set(cycles))
        return unique_cycles

    def decompose_greedy(edges):
        if not edges:
            return []
        
        adj = defaultdict(set)
        remaining = set()
        for e in edges:
            c = canonical(e)
            remaining.add(c)
            adj[c[0]].add(c[1])
            adj[c[1]].add(c[0])
        
        pieces = []
        
        # Extract 2-core
        to_remove = set()
        while True:
            changed = False
            for v in list(adj.keys()):
                if len(adj[v]) < 2:
                    to_remove.add(v)
                    changed = True
            if not to_remove:
                break
            for v in to_remove:
                for u in list(adj[v]):
                    adj[u].remove(v)
                    remaining.discard((min(u,v), max(u,v)))
                del adj[v]
            to_remove = set()
        
        # If adj is empty, all remaining edges are bridges
        if not adj:
            for e in remaining:
                pieces.append(list(e))
            return pieces
        
        # Greedy cycle extraction
        while adj:
            if time.time() - t0 > DEADLINE - 5:
                break
                
            best_cycle = None
            
            # Try to find a long cycle
            # Heuristic: Start from high degree vertices
            starts = sorted(adj.keys(), key=lambda x: -len(adj[x]))
            
            for start in starts[:10]: # Limit starts
                if len(adj[start]) < 2:
                    continue
                
                # Multiple random DFS attempts
                for _ in range(20):
                    visited = {start}
                    path = [start]
                    current = start
                    
                    # Randomized DFS
                    # We want to avoid closing too early, so prefer unvisited nodes
                    # but if we can close, we record it.
                    
                    stack = [(current, list(adj[current]))]
                    # Actually, recursive DFS is easier to manage for simple cycles
                    # Let's do a randomized walk that backtracks
                    
                    def random_dfs(u, visited, path):
                        # Shuffle neighbors
                        nbrs = list(adj[u])
                        random.shuffle(nbrs)
                        # Bias towards high degree?
                        # nbrs.sort(key=lambda x: -len(adj[x]))
                        # random.shuffle(nbrs[:3]) # Partial shuffle?
                        
                        for v in nbrs:
                            if v == start and len(path) >= 3:
                                return list(path)
                            if v not in visited:
                                visited.add(v)
                                path.append(v)
                                res = random_dfs(v, visited, path)
                                if res:
                                    return res
                                path.pop()
                                visited.remove(v)
                        return None

                    res = random_dfs(start, {start}, [start])
                    if res:
                        if best_cycle is None or len(res) > len(best_cycle):
                            best_cycle = res
            
            if best_cycle and len(best_cycle) >= 3:
                # Verify and remove
                valid = True
                for i in range(len(best_cycle)):
                    e = canonical((best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]))
                    if e not in remaining:
                        valid = False
                        break
                
                if valid:
                    pieces.append(best_cycle)
                    for i in range(len(best_cycle)):
                        e = canonical((best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]))
                        remaining.remove(e)
                        u, v = e
                        adj[u].remove(v)
                        adj[v].remove(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]
                else:
                    break
            else:
                # No cycle found, remaining edges in 2-core?
                # This shouldn't happen if adj is the 2-core, unless we missed a cycle
                # Fallback: treat as edges
                for e in remaining:
                    pieces.append(list(e))
                adj = {}
                remaining = set()
                break
        
        # Add any remaining bridges (edges not in 2-core or leftover)
        for e in remaining:
            pieces.append(list(e))
            
        return pieces

    def decompose_ilp(edges):
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            if not edges:
                return []
            
            # Build adjacency and remaining
            adj = defaultdict(set)
            remaining = set()
            for e in edges:
                c = canonical(e)
                remaining.add(c)
                adj[c[0]].add(c[1])
                adj[c[1]].add(c[0])
            
            # Extract 2-core
            to_remove = set()
            while True:
                changed = False
                for v in list(adj.keys()):
                    if len(adj[v]) < 2:
                        to_remove.add(v)
                        changed = True
                if not to_remove:
                    break
                for v in to_remove:
                    for u in list(adj[v]):
                        adj[u].remove(v)
                        remaining.discard((min(u,v), max(u,v)))
                    del adj[v]
                to_remove = set()
            
            pieces = []
            
            # Edges outside 2-core are bridges
            bridge_edges = list(remaining)
            for e in bridge_edges:
                pieces.append(list(e))
            
            if not adj:
                return pieces
            
            # Generate candidate cycles
            # 1. All cycles up to length 8
            candidates = find_all_cycles_up_to_len(adj, 8)
            
            # 2. Add some random long cycles
            # Re-use greedy finder logic to get a few long ones
            temp_adj = {k: set(v) for k, v in adj.items()}
            for _ in range(20):
                if not temp_adj: break
                starts = sorted(temp_adj.keys(), key=lambda x: -len(temp_adj[x]))
                if not starts: break
                start = starts[0]
                if len(temp_adj[start]) < 2: continue
                
                # Simple random walk to find a cycle
                visited = {start}
                path = [start]
                current = start
                found = False
                for _ in range(50):
                    nbrs = [n for n in temp_adj[current] if n not in visited or n == start]
                    if not nbrs: break
                    if len(path) >= 3 and start in nbrs:
                        # Close cycle
                        cycle = list(path)
                        # Check validity in original adj
                        valid = True
                        for i in range(len(cycle)):
                            e = canonical((cycle[i], cycle[(i+1)%len(cycle)]))
                            if e not in adj[cycle[i]]: # adj is static
                                valid = False
                                break
                        if valid:
                            candidates.append(tuple(cycle))
                            found = True
                            break
                        else:
                            break
                    # Pick a random neighbor
                    # Prefer unvisited
                    unvisited = [n for n in nbrs if n not in visited]
                    if unvisited:
                        next_node = random.choice(unvisited)
                    else:
                        break
                    visited.add(next_node)
                    path.append(next_node)
                    current = next_node
                
                if found:
                    # Remove cycle from temp_adj to find more
                    for i in range(len(cycle)):
                        e = canonical((cycle[i], cycle[(i+1)%len(cycle)]))
                        u,v = e
                        temp_adj[u].remove(v)
                        temp_adj[v].remove(u)
                        if not temp_adj[u]: del temp_adj[u]
                        if not temp_adj[v]: del temp_adj[v]

            # Deduplicate candidates
            candidates = list(set(candidates))
            
            if not candidates:
                # No cycles found, all edges in 2-core are bridges?
                # This implies no cycle, so 2-core was empty or logic error.
                # If 2-core is not empty, it must have a cycle.
                # Fallback to greedy for the core
                core_edges = []
                for u in adj:
                    for v in adj[u]:
                        if u < v:
                            core_edges.append((u,v))
                core_pieces = decompose_greedy(core_edges)
                pieces.extend(core_pieces)
                return pieces

            # ILP Setup
            # Variables: x_i for each candidate cycle, y_e for each edge in 2-core
            # Edges in 2-core
            core_edges = []
            for u in adj:
                for v in adj[u]:
                    if u < v:
                        core_edges.append((u, v))
            
            n_cycles = len(candidates)
            n_edges = len(core_edges)
            
            # Mapping edge to index
            edge_index = {e: i for i, e in enumerate(core_edges)}
            
            # Objective: Min sum(x) + sum(y)
            c = np.ones(n_cycles + n_edges)
            
            # Constraints: For each edge, sum of cycles containing it + y_edge = 1
            # A = [A_cycles | I]
            # A_cycles[i, j] = 1 if edge i is in cycle j
            A = np.zeros((n_edges, n_cycles + n_edges))
            
            for j, cyc in enumerate(candidates):
                for i in range(len(cyc)):
                    e = canonical((cyc[i], cyc[(i + 1) % len(cyc)]))
                    if e in edge_index:
                        A[edge_index[e], j] = 1
                    else:
                        # Cycle uses an edge not in core? Should not happen.
                        # If it does, this candidate is invalid.
                        pass
            
            # Identity matrix for y variables
            for i in range(n_edges):
                A[i, n_cycles + i] = 1
            
            # Bounds
            # x in [0,1], y in [0,1]
            integrality = np.concatenate([np.ones(n_cycles), np.ones(n_edges)])
            lb = np.zeros(n_cycles + n_edges)
            ub = np.ones(n_cycles + n_edges)
            
            # Constraints: A @ vars = 1
            # LinearConstraint(A, 1, 1)
            constraint = LinearConstraint(A, 1, 1)
            bounds = Bounds(lb, ub)
            
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                x_vars = res.x[:n_cycles]
                y_vars = res.x[n_cycles:]
                
                # Extract cycles
                for j in range(n_cycles):
                    if x_vars[j] > 0.5:
                        pieces.append(list(candidates[j]))
                
                # Extract edges
                for i in range(n_edges):
                    if y_vars[i] > 0.5:
                        pieces.append(list(core_edges[i]))
                
                return pieces
            else:
                # Fallback
                core_edges_list = [list(e) for e in core_edges]
                # We need to decompose the core edges.
                # Use greedy on the core edges
                core_pieces = decompose_greedy(core_edges)
                pieces.extend(core_pieces)
                return pieces
                
        except Exception:
            # Fallback to greedy
            return decompose_greedy(edges)

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            # Try ILP first, fallback to greedy
            try:
                res = decompose_ilp(edges)
                if res is None:
                    res = decompose_greedy(edges)
            except Exception:
                res = decompose_greedy(edges)
            results.append(res)

    return results
