import time
import sys
from collections import defaultdict
from itertools import combinations

# Set recursion limit high
sys.setrecursionlimit(10000)

def construct():
    """
    Decomposes three fixed random graphs into a minimum number of pieces (cycles and edges).
    Strategy: 
    1. Enumerate all simple cycles up to a maximum length (e.g., 15).
    2. Formulate the problem as a Maximum Weight Set Packing:
       - Variables: x_c for each cycle c (1 if selected, 0 otherwise).
       - Objective: Maximize sum((len(c) - 1) * x_c).
       - Constraints: For each edge e, sum(x_c for c containing e) <= 1.
    3. Solve the ILP using scipy.optimize.milp if available, otherwise use a custom 
       branch-and-bound or greedy heuristic with local search.
    4. The selected cycles form the cycle pieces. All other edges are single edge pieces.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_all_cycles_up_to_max_len(n, adj, max_len=15):
        """
        Finds all simple cycles of length up to max_len.
        Returns a list of lists of vertices.
        """
        cycles = []
        # To avoid duplicates, we only consider cycles where the smallest vertex is the start
        # and the second smallest is the second node? 
        # Standard trick: Fix start node u. Find paths from u to neighbors v > u 
        # that don't go through any node < u? 
        # Simpler: Generate all cycles, then canonicalize and deduplicate.
        
        # For efficiency, use DFS from each node, restricting the search space.
        # We will find all simple cycles.
        # Since n is small, we can use a bitmask approach or just DFS with visited set.
        
        # To reduce duplicates, we can enforce that the smallest vertex in the cycle is the start.
        # And the second vertex must be smaller than the last? 
        # Actually, just generating all and using a set of frozensets of edges for dedup is easy.
        
        all_cycles_set = set()
        
        for start in range(n):
            # DFS from start
            # To ensure we don't miss cycles and don't count rotations/reflections multiple times:
            # We can just find all simple cycles containing 'start' where 'start' is the minimum vertex in the cycle.
            # This partitions the set of cycles.
            
            def dfs(u, path, visited):
                nonlocal all_cycles_set
                if len(path) > max_len:
                    return
                
                for v in adj[u]:
                    if v == start and len(path) >= 3:
                        # Found a cycle
                        # Canonical form: sorted tuple of vertices? No, cycles are ordered.
                        # But for dedup, the set of edges is unique.
                        # Let's store the tuple of vertices in the order found, 
                        # but normalize by rotating so the min vertex is first and the second is the smallest neighbor?
                        # Actually, storing the frozenset of edges is the safest dedup key.
                        cycle_edges = set()
                        for i in range(len(path)):
                            u1 = path[i]
                            u2 = path[(i + 1) % len(path)]
                            cycle_edges.add(tuple(sorted((u1, u2))))
                        all_cycles_set.add(frozenset(cycle_edges))
                    elif v > start and v not in visited:
                        # Only visit nodes > start to ensure start is the min
                        # And v != start is handled above
                        visited.add(v)
                        path.append(v)
                        dfs(v, path, visited)
                        path.pop()
                        visited.remove(v)
            
            dfs(start, [start], {start})
            
        # Convert frozensets of edges back to vertex lists for the solver
        # We need to reconstruct the vertex order from the edges.
        # This is tricky. It's better to keep the vertex list from DFS.
        # Let's modify the DFS to store the vertex list directly, and use a set of normalized vertex tuples for dedup.
        
        # Re-implementing with vertex list dedup
        cycles_list = []
        seen_cycles = set()
        
        for start in range(n):
            def dfs2(u, path, visited):
                if len(path) > max_len:
                    return
                
                for v in adj[u]:
                    if v == start and len(path) >= 3:
                        # Found a cycle
                        # Normalize: rotate path so min vertex is first.
                        # The path is [start, ..., v, start]? No, path is [start, ..., v].
                        # The cycle vertices are path.
                        # To normalize, find index of min vertex (which is start).
                        # So path already starts with start.
                        # To avoid reverse duplicates: ensure path[1] < path[-1]
                        if path[1] < path[-1]:
                            key = tuple(path)
                            if key not in seen_cycles:
                                seen_cycles.add(key)
                                cycles_list.append(path)
                    elif v > start and v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs2(v, path, visited)
                        path.pop()
                        visited.remove(v)
            
            dfs2(start, [start], {start})
            
        return cycles_list

    def solve_set_packing(cycles, edges_set, n):
        """
        Solves the max weight set packing problem.
        Returns a list of selected cycles (vertex lists).
        """
        if not cycles:
            return []
        
        # Map edges to indices
        edge_list = sorted(edges_set)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        num_edges = len(edge_list)
        
        # Precompute for each cycle:
        # - weight (len - 1)
        # - list of edge indices it uses
        cycle_data = []
        for c in cycles:
            edge_indices = []
            valid = True
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                e = tuple(sorted((u, v)))
                if e in edge_to_idx:
                    edge_indices.append(edge_to_idx[e])
                else:
                    valid = False
                    break
            if valid and len(c) >= 3:
                cycle_data.append((len(c) - 1, edge_indices, c))
        
        if not cycle_data:
            return []
            
        # Try scipy.optimize.milp
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            num_cycles = len(cycle_data)
            
            # Objective: maximize sum(weight_i * x_i)
            # milp minimizes c @ x. So we use c = -weights.
            c_obj = np.array([-cd[0] for cd in cycle_data])
            
            # Constraints: sum(x_j for j containing e) <= 1 for each edge e
            # A matrix of shape (num_edges, num_cycles)
            A = np.zeros((num_edges, num_cycles))
            for j, (w, e_indices, verts) in enumerate(cycle_data):
                for e_idx in e_indices:
                    A[e_idx, j] = 1.0
            
            constraints = [
                LinearConstraint(A, ub=np.ones(num_edges)),
                Bounds(lb=np.zeros(num_cycles), ub=np.ones(num_cycles))
            ]
            
            # Integers
            integrality = np.ones(num_cycles)
            
            res = milp(c=c_obj, constraints=constraints, integrality=integrality)
            
            if res.success:
                selected = [cycle_data[i][2] for i in range(num_cycles) if res.x[i] > 0.5]
                return selected
            else:
                # Fallback to greedy if milp fails
                pass
                
        except Exception as e:
            pass
            
        # Fallback: Greedy with Local Search
        # Sort by weight descending
        sorted_cycles = sorted(cycle_data, key=lambda x: x[0], reverse=True)
        
        selected = []
        used_edges = set()
        
        # Greedy pass
        for w, e_indices, verts in sorted_cycles:
            if not any(e in used_edges for e in e_indices):
                selected.append(verts)
                used_edges.update(e_indices)
                
        # Local Search: Try to improve by swapping
        # This is a simple improvement. We can try to add a cycle if it frees up more weight?
        # Or remove a low weight cycle to add two higher weight ones?
        # Given time constraints, a single pass greedy is often decent, but let's try a simple swap.
        
        # Actually, for these small graphs, the greedy might be close to optimal.
        # Let's try a random restart greedy to pick the best.
        
        best_selected = selected
        best_weight = sum(cd[0] for cd in cycle_data if cd[2] in [s for s in best_selected])
        
        # A few random restarts
        import random
        rng = random.Random(42)
        
        for _ in range(10):
            # Perturb the order
            shuffled = sorted_cycles.copy()
            rng.shuffle(shuffled)
            # But keep high weight ones prioritized? 
            # Let's just do random greedy
            sel = []
            used = set()
            for w, e_indices, verts in shuffled:
                if not any(e in used for e in e_indices):
                    sel.append(verts)
                    used.update(e_indices)
            
            wgt = sum(cd[0] for cd in cycle_data if cd[2] in sel)
            if wgt > best_weight:
                best_weight = wgt
                best_selected = sel
                
        return best_selected

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: return all edges as pieces to ensure valid output
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        edges_set = set(edges)
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find cycles
        # Limit cycle length to 15 to keep enumeration fast
        # For n=30, p=0.3, avg degree is ~9. Cycles are short.
        cycles = find_all_cycles_up_to_max_len(n, adj, max_len=15)
        
        # Solve packing
        selected_cycles = solve_set_packing(cycles, edges_set, n)
        
        # Construct pieces
        pieces = []
        used_edges = set()
        
        for cycle in selected_cycles:
            # Add cycle piece
            pieces.append(cycle)
            # Mark edges as used
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                used_edges.add(e)
                
        # Add remaining edges
        for e in edges_set:
            if e not in used_edges:
                pieces.append(list(e))
                
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} pieces")
