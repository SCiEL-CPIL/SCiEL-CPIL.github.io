"""Combined program: Cycle Decomposition via ILP."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # Safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def enumerate_cycles(adj, n, max_cycles=20000):
        """Enumerate simple cycles in the graph."""
        cycles = []
        # Use a DFS approach to find all simple cycles
        # To avoid duplicates, we can enforce that the smallest node in the cycle is the start
        # and we only visit nodes larger than the start? No, that misses cycles.
        # Standard Johnson's algorithm is complex to implement quickly.
        # For small n, a simple DFS from each node, keeping a visited set, 
        # and only closing the cycle if we return to the start, 
        # and ensuring we don't revisit nodes in the path.
        # To avoid finding the same cycle multiple times (rotations), 
        # we can require that the start node is the minimum index in the cycle.
        
        for start in range(n):
            # Only start from the minimum node in the cycle to avoid duplicates
            # But we don't know the other nodes yet.
            # Alternative: Generate all cycles, then deduplicate by sorting the vertex list.
            # Given the small size, deduplication is cheap.
            
            def dfs(current, path, visited):
                if len(cycles) >= max_cycles:
                    return
                for neighbor in adj[current]:
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        cycles.append(path[:])
                    elif neighbor not in visited and neighbor > start:
                        # Optimization: Only visit neighbors with index > start 
                        # to ensure each cycle is found exactly once 
                        # (starting from its smallest node).
                        # Wait, this only works if we enforce start is the min node.
                        # If we enforce start is min node, we only traverse to nodes > start.
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
            
            dfs(start, [start], {start})
            
        return cycles

    def decompose_graph_ilp(edges, n):
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Enumerate cycles
        cycles = enumerate_cycles(adj, n, max_cycles=50000)
        
        if not cycles:
            return [list(e) for e in edge_set]
        
        # Prepare for ILP
        # Variables: x_c for each cycle c, y_e for each edge e
        # We want to minimize sum(x_c) + sum(y_e)
        # Subject to: for each edge e, sum(x_c for c containing e) + y_e = 1
        
        edge_list = sorted(edge_set)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        num_edges = len(edge_list)
        num_cycles = len(cycles)
        
        # Build constraint matrix A for cycles
        # A is (num_edges x num_cycles)
        # A[e_idx, c_idx] = 1 if edge e is in cycle c
        
        # To save memory/time, we can build the matrix sparsely or just as a dense array if small.
        # num_edges ~ 100, num_cycles ~ 1000-5000. Dense is fine (100*5000 = 500k floats).
        
        A = np.zeros((num_edges, num_cycles), dtype=int)
        
        for c_idx, cycle in enumerate(cycles):
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                e_idx = edge_index[e]
                A[e_idx, c_idx] = 1
                
        # Total variables: num_cycles + num_edges
        total_vars = num_cycles + num_edges
        
        # Objective coefficients
        c = np.ones(total_vars)
        
        # Constraints: A @ x + I @ y = 1
        # We combine into one matrix A_full
        # A_full is (num_edges x total_vars)
        A_full = np.hstack((A, np.eye(num_edges)))
        
        # Linear constraint: A_full @ z = 1
        # lower = 1, upper = 1
        lin_con = LinearConstraint(A_full, 1, 1)
        
        # Bounds: 0 <= z <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all variables are binary
        integrality = np.ones(total_vars)
        
        # Solve
        try:
            res = milp(c=c, constraints=lin_con, integrality=integrality, bounds=bounds)
            if res.status == 0: # Optimal
                z = res.x
                x_cycles = z[:num_cycles]
                y_edges = z[num_cycles:]
                
                pieces = []
                
                # Add selected cycles
                for i in range(num_cycles):
                    if x_cycles[i] > 0.5:
                        pieces.append(cycles[i][:])
                        
                # Add selected single edges
                for i in range(num_edges):
                    if y_edges[i] > 0.5:
                        pieces.append(list(edge_list[i]))
                        
                return pieces
            else:
                # Fallback if ILP fails
                pass
        except Exception:
            pass
            
        # Fallback: Greedy heuristic from current program
        return decompose_graph_greedy(edges, n)

    def decompose_graph_greedy(edges, n):
        """Fallback greedy decomposition."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining = set(edge_set)
        pieces = []
        rng = random.Random(42)
        
        # Simple greedy: find any cycle, remove, repeat
        while remaining:
            cycle = None
            # Try to find a cycle using DFS from random nodes
            for _ in range(50):
                if not remaining: break
                start = rng.choice(list(remaining)[0][0] if remaining else 0) # Approx
                # Better: pick a random edge, then try to extend
                # This is slow, so we just use a simple check
                # For fallback, just take edges if no cycle found quickly
                pass
            
            # Find a cycle in remaining
            found_cycle = None
            for start in range(n):
                if len(adj[start]) < 2: continue
                # DFS
                path = [start]
                visited = {start}
                stack = [(start, iter(adj[start]))]
                while stack:
                    node, it = stack[-1]
                    advanced = False
                    for nb in it:
                        e = tuple(sorted((node, nb)))
                        if e not in remaining:
                            continue
                        if nb == start and len(path) >= 3:
                            found_cycle = path[:]
                            break
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            stack.append((nb, iter(adj[nb])))
                            advanced = True
                            break
                    if found_cycle:
                        break
                    if not advanced:
                        stack.pop()
                        if path: path.pop()
                        if visited: visited.discard(node)
                if found_cycle:
                    break
            
            if found_cycle:
                for i in range(len(found_cycle)):
                    u, v = found_cycle[i], found_cycle[(i+1)%len(found_cycle)]
                    e = tuple(sorted((u,v)))
                    remaining.discard(e)
                pieces.append(found_cycle)
            else:
                e = tuple(sorted(rng.choice(list(remaining))))
                remaining.discard(e)
                pieces.append(list(e))
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Check time
        if time.time() > deadline - 10:
            # Fallback to greedy for all remaining
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append(decompose_graph_greedy(edges2, n2))
            break
            
        pieces = decompose_graph_ilp(edges, n)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
