import time
import random
import sys
import itertools
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def find_cycles_in_graph(edges, max_cycle_len=10, max_cycles=500):
        """
        Find a diverse set of simple cycles in the graph.
        We limit the number of cycles to keep the ILP small.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if not adj:
            return []

        cycles = []
        seen_cycles = set()
        
        # Heuristic: Try to find cycles starting from each node
        # To ensure diversity, we can use DFS with backtracking and a limit on depth
        # Since n is small (<=30), and we only need a good set, we can be a bit aggressive
        
        def dfs(start, current, path, visited):
            if len(cycles) >= max_cycles:
                return
            
            # If we return to start and have at least 3 nodes, it's a cycle
            if current == start and len(path) >= 3:
                # Normalize cycle representation
                # Find the smallest vertex in the cycle to start, then direction
                min_idx = path.index(min(path))
                norm_path = path[min_idx:] + path[:min_idx]
                
                # Check reverse
                rev_path = [norm_path[0]] + list(reversed(norm_path[1:]))
                
                key = tuple(norm_path)
                rev_key = tuple(rev_path)
                
                if key not in seen_cycles and rev_key not in seen_cycles:
                    seen_cycles.add(key)
                    seen_cycles.add(rev_key)
                    cycles.append(norm_path)
                    return # Stop this branch to save time? Or continue? 
                           # Stopping helps find diverse cycles faster.
            
            if len(path) >= max_cycle_len:
                return
                
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    # Cycle found
                    dfs(start, start, path, visited)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    dfs(start, neighbor, path + [neighbor], visited)
                    visited.remove(neighbor)

        # We might not find ALL cycles, but a good subset
        # To improve diversity, we can randomize neighbor order or start nodes
        for start_node in sorted(adj.keys()):
            if len(cycles) >= max_cycles:
                break
            dfs(start_node, start_node, [start_node], {start_node})

        return cycles

    def decompose_with_ilp(edges, time_limit):
        if not edges:
            return []

        # 1. Find candidate cycles
        # We want to maximize the number of edges covered by cycles.
        # Each cycle covers len(cycle) edges.
        # Each edge can be covered by at most 1 cycle.
        # Any edge not covered by a cycle becomes a single-edge piece.
        # Total pieces = (Number of edges not covered) + (Number of cycles selected)
        # Let x_c be 1 if cycle c is selected.
        # Let e_c be the set of edges in cycle c.
        # Constraint: sum_{c contains e} x_c <= 1 for all edges e.
        # Objective: Minimize sum_e (1 - sum_{c contains e} x_c) + sum_c x_c
        #           = |E| - sum_e sum_{c contains e} x_c + sum_c x_c
        #           = |E| + sum_c x_c * (1 - len(c))
        #           = |E| - sum_c x_c * (len(c) - 1)
        
        # So maximizing sum_c x_c * (len(c) - 1) is equivalent.
        
        cycles = find_cycles_in_graph(edges, max_cycle_len=12, max_cycles=1000)
        
        if not cycles:
            return [list(e) for e in edges]

        # Map edges to indices
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        num_edges = len(edges)
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            return [list(e) for e in edges]

        # Build constraint matrix A (num_edges x num_cycles)
        # A[i, j] = 1 if edge i is in cycle j, else 0
        # We can use sparse matrix if available, but numpy is fine for small sizes.
        # However, milp expects dense or sparse. Let's build dense for simplicity if small, 
        # but 30 nodes, ~200 edges, 1000 cycles -> 200x1000 matrix is 200k entries, fine.
        
        A = np.zeros((num_edges, num_cycles))
        for j, cycle in enumerate(cycles):
            # Get edges of cycle
            for k in range(len(cycle)):
                u = cycle[k]
                v = cycle[(k + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in edge_to_idx:
                    i = edge_to_idx[e]
                    A[i, j] = 1
        
        # Objective: c_j = -(len(cycle_j) - 1)
        c = np.array([-(len(cycle) - 1) for cycle in cycles])
        
        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, ub=np.ones(num_edges))
        
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x_j is integer (0 or 1)
        integrality = np.ones(num_cycles)
        
        # Solve
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        except Exception:
            # Fallback to greedy if ILP fails
            return decompose_greedy(edges)

        if res.status != 0:
            # Fallback
            return decompose_greedy(edges)

        # Extract solution
        selected_cycles = []
        covered_edges = set()
        
        for j in range(num_cycles):
            if res.x[j] > 0.5:
                selected_cycles.append(cycles[j])
                # Add edges to covered set
                cycle = cycles[j]
                for k in range(len(cycle)):
                    u = cycle[k]
                    v = cycle[(k + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    covered_edges.add(e)
        
        # Construct result
        pieces = []
        # Add cycles
        for cycle in selected_cycles:
            pieces.append(cycle)
        
        # Add remaining edges as single edges
        for e in edges:
            if e not in covered_edges:
                pieces.append(list(e))
                
        return pieces

    def decompose_greedy(edges, time_limit):
        """
        Fallback greedy strategy:
        1. Find longest cycle
        2. Remove its edges
        3. Repeat
        """
        remaining_edges = set(edges)
        pieces = []
        
        while remaining_edges and time.time() < time_limit:
            # Find a cycle
            # Simple DFS to find any cycle
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            found_cycle = None
            # Try to find longest cycle heuristically
            # Just pick a random edge and try to extend
            # Or use the find_cycles function but take the longest one found
            
            # For speed in fallback, let's just do a simple random walk cycle finder
            # or reuse find_cycles and pick the longest
            cycles = find_cycles_in_graph(list(remaining_edges), max_cycle_len=15, max_cycles=50)
            
            if cycles:
                # Pick longest
                cycles.sort(key=len, reverse=True)
                best_cycle = cycles[0]
                pieces.append(best_cycle)
                
                # Remove edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = tuple(sorted((u, v)))
                    remaining_edges.discard(e)
            else:
                # No cycle found, take a single edge
                # To avoid getting stuck, pick an edge that might allow a cycle later?
                # Greedy: pick any edge.
                edge = next(iter(remaining_edges))
                pieces.append(list(edge))
                remaining_edges.discard(edge)
                
        # Add any remaining
        for e in remaining_edges:
            pieces.append(list(e))
            
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed
        if time_remaining < 1.0:
            # Not enough time, just return all edges as single pieces
            results.append([list(e) for e in edges])
            continue
            
        # Allocate time proportional to remaining graphs
        graphs_left = len(graphs) - i
        time_per_graph = time_remaining / graphs_left
        
        # ILP is fast for these sizes, but let's cap it
        t_limit = time.time() + min(time_per_graph * 0.9, 20.0)
        
        # Try ILP first
        try:
            pieces = decompose_with_ilp(edges, t_limit)
        except Exception:
            pieces = decompose_greedy(edges, t_limit)
            
        # Verification step (internal)
        # Ensure validity
        edge_set = set(edges)
        used = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used:
                    valid = False
                    break
                used.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used:
                        valid = False
                        break
                    used.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used:
                    valid = False
                    break
                used.add(e)
            else:
                valid = False
                break
                
        if not valid or used != edge_set:
            # Fallback to greedy if ILP produced invalid result
            pieces = decompose_greedy(edges, t_limit)
            # Re-verify
            used = set()
            for piece in pieces:
                if len(piece) == 2:
                    u, v = piece
                    e = tuple(sorted((u, v)))
                    used.add(e)
                elif len(piece) >= 3:
                    for k in range(len(piece) - 1):
                        u, v = piece[k], piece[k+1]
                        e = tuple(sorted((u, v)))
                        used.add(e)
                    u, v = piece[-1], piece[0]
                    e = tuple(sorted((u, v)))
                    used.add(e)
            if used != edge_set:
                # Last resort: all single edges
                pieces = [list(e) for e in edges]

        results.append(pieces)

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
