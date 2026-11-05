"""
Strategy: Use Integer Linear Programming (ILP) to find the longest simple path.
The graph is sparse (G(100, 0.018)), so the number of edges is small (~90).
We define binary variables for each edge (directional) and use flow conservation
to ensure the selected edges form a single simple path.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    """Build adjacency list and set for fast lookups."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = P['n']
    adj = _build_adjacency()
    
    # 1. Define directed edges
    # For each undirected edge (u, v), we have two directed variables: x_{u,v} and x_{v,u}
    directed_edges = []
    edge_index = {}
    idx = 0
    for u, v in P['edges']:
        directed_edges.append((u, v))
        directed_edges.append((v, u))
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx + 1
        idx += 2
        
    m = len(directed_edges) # Number of directed edges
    
    # 2. Objective: Maximize sum of x_{u,v}
    # c = [-1, -1, ..., -1] because milp minimizes
    c = np.ones(m) * -1.0
    
    # 3. Constraints
    
    # A) Flow conservation:
    # For each vertex i: sum_{j} x_{j,i} - sum_{j} x_{i,j} = 0
    # This ensures that if a vertex is internal, 1 edge enters and 1 leaves.
    # Wait, this allows cycles. To prevent cycles and ensure a single path, 
    # we typically need subtour elimination or a specific formulation.
    # A simpler approach for "simple path" (not necessarily Hamiltonian) 
    # is harder to constrain purely with flow conservation without big-M or MTZ.
    #
    # Alternative: Use the "degree" constraint + "no subtour" is complex.
    # Let's use a different formulation:
    # Let x_{u,v} be 1 if edge u->v is in the path.
    # Let y_i be 1 if vertex i is in the path.
    #
    # Constraints:
    # 1. Degree constraint: For each vertex i, sum_{j} x_{j,i} + sum_{j} x_{i,j} <= 2 * y_i
    #    Actually, for a simple path, degree is 0, 1, or 2.
    #    If y_i=1, degree can be 1 (endpoint) or 2 (internal).
    #    Sum of x_{j,i} (in-degree) + Sum of x_{i,j} (out-degree) <= 2.
    #    Also, we must ensure connectivity.
    #
    # A robust ILP for Longest Simple Path often uses:
    # Minimize -sum(x)
    # Subject to:
    # 1. x_{u,v} + x_{v,u} <= 1 (cannot use both directions)
    # 2. Flow conservation with a "source" and "sink" or just ensure no cycles.
    #    Since we don't know source/sink, we can use the MTZ (Miller-Tucker-Zemlin) formulation 
    #    to eliminate cycles, but it requires a root. 
    #    Alternatively, we can just solve for the maximum matching? No, path is ordered.
    #
    # Let's try a simpler heuristic-guided ILP or just a strong heuristic if ILP is too slow.
    # Given N=100, M~180. ILP might be slow if not careful.
    #
    # Let's use a different strategy: **Branch and Bound with strong pruning** or **Iterative Deepening**.
    # Actually, the current program's score is 2.88. A path of length 288? No, max vertices is 100.
    # The score is `value/seed`. Seed is 1.0? No, "1.0 equals the seed program".
    # If score is 2.88, and seed is 1.0, the value is 2.88 * seed_value?
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # If seed program returns a path of length L_seed, and score is 1.0, then L_seed / L_seed = 1.
    # So "value" is the length of the path.
    # Current score 2.8889 implies the current program finds a path of length ~2.89 * L_seed?
    # Wait, if the seed program is just a baseline, maybe it returns a short path.
    # If the seed returns length 10, and score is 1.0, value=10.
    # If current score is 2.88, value = 28.8? No, path length is integer.
    # Maybe the seed value is not 1.0 in the denominator?
    # "1.0 equals the seed program". So Score = Value / Seed_Value.
    # If Score = 2.88, Value = 2.88 * Seed_Value.
    # What is the Seed_Value?
    # The prompt says: "construct() -> ... Maximize path vertex count."
    # If the seed program is a simple greedy, it might find a path of length ~20-30.
    # If Seed_Value = 10, then Value = 28.
    # If Seed_Value = 1, then Value = 2.88 (impossible, must be integer).
    # Let's assume the "seed" is a specific reference solution provided by the system, 
    # and its length is S. The score is L / S.
    # If the graph has 100 nodes, the max path length is 100.
    # If S is small (e.g., 10), a path of 30 gives 3.0.
    # So the goal is to find the longest path, ideally close to 100 (Hamiltonian path).
    
    # Given the sparsity (p=0.018), the graph is likely disconnected or has small components.
    # Average degree = 1.8.
    # In a random graph G(n, p), the giant component appears when np > 1.
    # Here np = 1.8 > 1, so there is a giant component.
    # The size of the giant component is roughly 1 - 1/np = 1 - 1/1.8 = 0.44? No, that's for p near critical.
    # For p=0.018, n=100, expected edges = 90.
    # The largest component will contain most of the nodes.
    # A Hamiltonian path exists in the largest component with high probability if the component is sufficiently connected.
    # However, finding the LONGEST path in a general graph is NP-hard.
    
    # Let's stick to a very efficient heuristic that is likely better than the current one,
    # or a fast ILP solver call.
    
    # Since we have 90 seconds, we can afford a more expensive search than the current one.
    # The current one uses greedy + random + backtrack.
    
    # New Strategy: **Simulated Annealing** or **Genetic Algorithm** on the path representation?
    # Or **A* Search** with a good heuristic?
    
    # Let's try **Iterative Deepening DFS (IDDFS)** with strong pruning, but optimized.
    # Actually, for N=100, DFS is too slow.
    
    # Let's try **2-Opt Local Search** on a set of initial paths, which is very fast and effective for TSP-like problems.
    # We can generate many random paths (greedy, random walk) and then apply 2-opt improvements.
    # 2-opt swaps segments to improve the path. For "Longest Path", we want to MAXIMIZE length, 
    # but 2-opt usually minimizes cost. We can invert the logic: 
    # We want to find a path that uses as many edges as possible.
    # Actually, if we just want to maximize the number of vertices, we can use a "repair" heuristic.
    
    # Let's implement a **Greedy + Local Search (2-Opt style extension)** strategy.
    # 1. Generate multiple candidate paths using different heuristics (Nearest Neighbor, Farthest First, etc.).
    # 2. For each candidate, try to extend it at both ends by finding neighbors not in the path.
    # 3. Try "insertion" moves: remove a vertex from the path and try to insert it elsewhere if it connects better?
    #    No, removing a vertex shortens the path. We want to ADD vertices.
    #    So the primary move is "Extension".
    #    Secondary move: "Reordering" to allow extension.
    #    Example: Path A-B-C-D. If C has a neighbor E not in path, but D is a dead end, 
    #    maybe reordering to A-B-D-C allows C to connect to E?
    
    # Let's implement a robust **Greedy Construction + Iterative Extension** loop.
    
    best_path = []
    
    # Helper to check if a path is valid and get its length
    def path_len(p):
        return len(p)
    
    def is_valid_path(p, adj):
        if not p:
            return False
        seen = set(p)
        if len(seen) != len(p):
            return False
        for i in range(len(p) - 1):
            if p[i+1] not in adj[p[i]]:
                return False
        return True
        
    def try_extend_path(p, adj, max_iter=1000):
        """Try to extend the path at either end repeatedly."""
        p = list(p)
        extended = True
        while extended and max_iter > 0:
            extended = False
            max_iter -= 1
            
            # Try extending at the end
            end = p[-1]
            for nxt in adj[end]:
                if nxt not in set(p):
                    p.append(nxt)
                    extended = True
                    break
                    
            # Try extending at the front
            if not extended:
                front = p[0]
                for nxt in adj[front]:
                    if nxt not in set(p):
                        p.insert(0, nxt)
                        extended = True
                        break
                        
        return p

    def two_opt_improve(p, adj, max_iter=500):
        """
        Try to improve the path by reordering segments to allow further extension.
        This is heuristic. We look for pairs of edges (i, i+1) and (j, j+1) 
        such that swapping the segment between them creates a longer path or 
        allows extension.
        For simplicity, we'll just do random perturbations and re-extension.
        """
        p = list(p)
        current_len = len(p)
        
        for _ in range(max_iter):
            # Randomly pick two indices i < j
            if len(p) < 4:
                break
            i = np.random.randint(1, len(p) - 2)
            j = np.random.randint(i + 1, len(p) - 1)
            
            # Reverse the segment p[i+1..j]
            # Original: ... A - B ... C - D ...
            # New: ... A - C ... B - D ...
            # Check if A-C and B-D are edges.
            a = p[i]
            b = p[i+1]
            c = p[j]
            d = p[j+1]
            
            if c in adj[a] and b in adj[d]:
                # Perform reversal
                new_p = p[:i+1] + p[i+1:j+1][::-1] + p[j+1:]
                # Validate
                if is_valid_path(new_p, adj):
                    # Try to extend this new path
                    extended_p = try_extend_path(new_p, adj)
                    if len(extended_p) > current_len:
                        p = extended_p
                        current_len = len(p)
                        
        return p

    # Phase 1: Generate diverse initial paths
    candidates = []
    
    # 1a. Greedy from every node with different tie-breaking
    for start in range(n):
        # Min degree neighbor
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            nbrs = [x for x in adj[cur] if x not in seen]
            if not nbrs:
                break
            # Pick neighbor with min degree (unvisited)
            nbrs.sort(key=lambda x: len(adj[x] - seen))
            nxt = nbrs[0]
            path.append(nxt)
            seen.add(nxt)
        candidates.append(path)
        
        # Max degree neighbor
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            nbrs = [x for x in adj[cur] if x not in seen]
            if not nbrs:
                break
            nbrs.sort(key=lambda x: -len(adj[x] - seen))
            nxt = nbrs[0]
            path.append(nxt)
            seen.add(nxt)
        candidates.append(path)

    # 1b. Random walks
    rng = np.random.default_rng(42)
    for _ in range(50):
        start = rng.integers(0, n)
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            nbrs = [x for x in adj[cur] if x not in seen]
            if not nbrs:
                break
            nxt = rng.choice(nbrs)
            path.append(nxt)
            seen.add(nxt)
        candidates.append(path)

    # Phase 2: Improve candidates
    for p in candidates:
        if time.time() > deadline - 5:
            break
        
        # First, extend greedily
        p_ext = try_extend_path(p, adj)
        
        # Then, try 2-opt improvements
        p_imp = two_opt_improve(p_ext, adj)
        
        if len(p_imp) > len(best_path):
            best_path = p_imp
            
        # Check time
        if time.time() > deadline - 10:
            break

    # Phase 3: If we have time, try a more exhaustive search on the best component
    # Find the largest connected component
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            stack = [i]
            comp = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nb in adj[node]:
                    if nb not in visited:
                        stack.append(nb)
            components.append(comp)
            
    # Sort components by size
    components.sort(key=len, reverse=True)
    
    # If the best path is not using the entire largest component, try to force it
    # This is hard to force in ILP, but we can try DFS on the largest component if it's small enough (< 30 nodes)
    # If the largest component is > 30, we rely on the heuristics above.
    
    if not best_path:
        # Fallback
        if components:
            best_path = components[0][:2] if len(components[0]) > 1 else components[0]
        else:
            best_path = [0]
            
    return best_path

# EVOLVE-BLOCK-END
