import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    # Fast deterministic HC finder using Warnsdorff's rule with limited backtracking
    def find_hc_fast(adj_remaining, start=0, max_nodes=50000):
        visited = [False] * n
        path = [start]
        visited[start] = True
        nodes_explored = [0]
        
        def dfs(current, depth):
            if nodes_explored[0] > max_nodes:
                return False
            if depth == n:
                if start in adj_remaining[current]:
                    return True
                return False
            
            # Warnsdorff: prefer neighbors with fewest unvisited neighbors
            unvisited = [v for v in adj_remaining[current] if not visited[v]]
            if not unvisited:
                return False
            
            unvisited.sort(key=lambda v: len([x for x in adj_remaining[v] if not visited[x]]))
            
            for nxt in unvisited:
                nodes_explored[0] += 1
                if nodes_explored[0] > max_nodes:
                    return False
                visited[nxt] = True
                path.append(nxt)
                
                if dfs(nxt, depth + 1):
                    return True
                
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(start, 1):
            return list(path)
        return None
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    deadline = time.time() + 80  # Safety margin for 90s limit
    best_cycles = []
    
    # Strategy 1: Greedy with multiple random orderings
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        current_adj = [set(s) for s in adj]
        cycles = []
        success = True
        
        for i in range(3):
            # Try different starting vertices
            found = False
            for start_v in range(n):
                if time.time() > deadline:
                    break
                hc = find_hc_fast(current_adj, start=start_v, max_nodes=200000)
                if hc is not None:
                    found = True
                    break
            
            if not found:
                success = False
                break
            
            cycles.append(hc)
            for j in range(len(hc)):
                u = hc[j]
                v = hc[(j + 1) % len(hc)]
                current_adj[u].discard(v)
                current_adj[v].discard(u)
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:
                # Verify and return immediately
                break
    
    # If we found 3 cycles, validate and return
    if len(best_cycles) >= 3:
        result = []
        used_edges = set()
        valid = True
        for cycle in best_cycles[:3]:
            if len(cycle) != n or len(set(cycle)) != n:
                valid = False
                break
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ek = (min(u, v), max(u, v))
                if ek not in edge_set:
                    valid = False
                    break
                if ek in used_edges:
                    valid = False
                    break
                used_edges.add(ek)
            if not valid:
                break
        
        if valid:
            return [[int(v) for v in cycle] for cycle in best_cycles[:3]]
    
    # Strategy 2: MILP-based approach using scipy
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            m = len(edges)
            # Variables: x[i][j] = 1 if edge i is in cycle j, for j in 0,1,2
            # Total variables: 3 * m
            # Constraints:
            # 1. For each vertex v and each cycle j: sum of edges incident to v in cycle j = 2
            # 2. For each edge i: sum over j of x[i][j] <= 1 (edge-disjoint)
            # 3. Subtour elimination (hard to do fully, but we can add some)
            
            # Build incidence: for each vertex, which edges are incident
            vertex_edges = defaultdict(list)
            for idx, (u, v) in enumerate(edges):
                vertex_edges[u].append(idx)
                vertex_edges[v].append(idx)
            
            # Objective: maximize number of edges used (should be 3*n = 63 if all 3 cycles found)
            c = np.zeros(3 * m)
            c[:] = -1.0  # Minimize negative = maximize
            
            # Constraint 1: degree constraint for each vertex and each cycle
            # For vertex v, cycle j: sum_{e incident to v} x[e][j] = 2
            A_rows = []
            b_lower = []
            b_upper = []
            
            for v in range(n):
                for j in range(3):
                    row = np.zeros(3 * m)
                    for e_idx in vertex_edges[v]:
                        row[j * m + e_idx] = 1.0
                    A_rows.append(row)
                    b_lower.append(2.0)
                    b_upper.append(2.0)
            
            # Constraint 2: edge-disjointness: for each edge, sum over cycles <= 1
            for e_idx in range(m):
                row = np.zeros(3 * m)
                for j in range(3):
                    row[j * m + e_idx] = 1.0
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(1.0)
            
            A = np.array(A_rows)
            b_lo = np.array(b_lower)
            b_hi = np.array(b_upper)
            
            constraints = LinearConstraint(A, b_lo, b_hi)
            integrality = np.ones(3 * m)
            bounds = Bounds(0, 1)
            
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0 and res.fun < -62.5:  # At least 63 edges used
                cycles = []
                for j in range(3):
                    cycle_edges = []
                    for e_idx in range(m):
                        if res.x[j * m + e_idx] > 0.5:
                            cycle_edges.append(edges[e_idx])
                    
                    if len(cycle_edges) == n:
                        # Reconstruct cycle from edge list
                        # Build adjacency from these edges
                        cyc_adj = [set() for _ in range(n)]
                        for u, v in cycle_edges:
                            cyc_adj[u].add(v)
                            cyc_adj[v].add(u)
                        
                        # Verify it's a single cycle
                        if all(len(cyc_adj[v]) == 2 for v in range(n)):
                            # Traverse the cycle
                            start = 0
                            path = [start]
                            prev = -1
                            curr = start
                            for _ in range(n - 1):
                                nexts = list(cyc_adj[curr])
                                nexts = [x for x in nexts if x != prev]
                                if not nexts:
                                    break
                                prev = curr
                                curr = nexts[0]
                                path.append(curr)
                            
                            if len(path) == n and path[0] in cyc_adj[path[-1]]:
                                cycles.append(path)
                
                if len(cycles) == 3:
                    # Validate
                    used_edges = set()
                    valid = True
                    for cycle in cycles:
                        if len(cycle) != n or len(set(cycle)) != n:
                            valid = False
                            break
                        for j in range(len(cycle)):
                            u = cycle[j]
                            v = cycle[(j + 1) % len(cycle)]
                            ek = (min(u, v), max(u, v))
                            if ek not in edge_set:
                                valid = False
                                break
                            if ek in used_edges:
                                valid = False
                                break
                            used_edges.add(ek)
                        if not valid:
                            break
                    
                    if valid:
                        return [[int(v) for v in cycle] for cycle in cycles]
        except Exception:
            pass
    
    # Fallback: return best found
    if best_cycles:
        result = []
        used_edges = set()
        valid = True
        for cycle in best_cycles:
            if len(cycle) != n or len(set(cycle)) != n:
                valid = False
                break
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ek = (min(u, v), max(u, v))
                if ek not in edge_set:
                    valid = False
                    break
                if ek in used_edges:
                    valid = False
                    break
                used_edges.add(ek)
            if not valid:
                break
        
        if valid:
            return [[int(v) for v in cycle] for cycle in best_cycles]
    
    # Last resort: return empty (should not happen)
    return []

# Test the program
if __name__ == '__main__':
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Found {len(result)} cycles in {elapsed:.2f}s")
    for i, cycle in enumerate(result):
        print(f"Cycle {i+1}: {cycle}")
