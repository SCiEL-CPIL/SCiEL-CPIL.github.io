import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import heapq
import time

N = 21

def construct():
    best_value = -np.inf
    best_solution = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.42857, 0.5, 2.0, 0.6667, 1.5]
    
    for w in aspect_ratios:
        h = 2.0 - w
        if w < 0.15 or h < 0.15:
            continue
        
        # Grid resolution - fine enough to capture good solutions
        # Number of grid points in x and y
        nx = max(10, int(np.round(15 * w / max(w, h))))
        ny = max(10, int(np.round(15 * h / max(w, h))))
        
        # Ensure we have enough grid points
        nx = max(nx, 12)
        ny = max(ny, 12)
        
        # Grid positions
        xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
        ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
        
        # All candidate centers and their max radii
        candidates = []
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                # Max radius at this position
                r_max = min(x, w - x, y, h - y)
                if r_max > 1e-6:
                    candidates.append((x, y, r_max, xi, yi))
        
        n_cand = len(candidates)
        if n_cand < N:
            continue
        
        # Build conflict graph
        # Two candidates conflict if circles with max radii would overlap
        # We use a bipartite approach: color by (xi + yi) % 2
        # For each pair, check if they conflict
        
        # For efficiency, only check nearby points
        conflict = np.zeros((n_cand, n_cand), dtype=bool)
        
        for i in range(n_cand):
            xi, yi, ri, xi_idx, yi_idx = candidates[i]
            for j in range(i + 1, n_cand):
                xj, yj, rj, xj_idx, yj_idx = candidates[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < ri + rj - 1e-9:
                    conflict[i, j] = True
                    conflict[j, i] = True
        
        # Bipartite coloring: (xi + yi) % 2
        left_nodes = []
        right_nodes = []
        for i, (x, y, r, xi_idx, yi_idx) in enumerate(candidates):
            if (xi_idx + yi_idx) % 2 == 0:
                left_nodes.append(i)
            else:
                right_nodes.append(i)
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Build bipartite graph for max-flow
        # Source -> left nodes (capacity = weight)
        # Left -> right (capacity = inf if conflict)
        # Right -> sink (capacity = weight)
        
        # We want maximum weight independent set
        # MWIS = total_weight - min_weight_vertex_cover
        # Min vertex cover in bipartite graph = max matching (Kőnig's theorem)
        # But we need weighted version
        
        # For weighted bipartite min vertex cover, we use max-flow:
        # Source -> left_i (cap = weight_i)
        # left_i -> right_j (cap = inf) if edge exists
        # right_j -> sink (cap = weight_j)
        # Min cut = min vertex cover weight
        # MWIS = total_weight - min_cut
        
        weights = np.array([c[2] for c in candidates])
        total_weight = np.sum(weights)
        
        # Build flow network
        # Nodes: 0=source, 1..n_left = left nodes, n_left+1..n_left+n_right = right nodes, n_left+n_right+1 = sink
        n_nodes = 2 + n_left + n_right
        source = 0
        sink = n_nodes - 1
        
        # Adjacency list for max-flow (Edmonds-Karp or Dinic)
        # Use scipy's max_flow if available, otherwise implement
        
        # Build capacity matrix
        cap = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        
        # Source to left nodes
        for i, node_idx in enumerate(left_nodes):
            cap[source, 1 + i] = weights[node_idx]
        
        # Right nodes to sink
        for j, node_idx in enumerate(right_nodes):
            cap[1 + n_left + j, sink] = weights[node_idx]
        
        # Left to right edges (infinite capacity if conflict)
        INF = 1e9
        for i, left_idx in enumerate(left_nodes):
            for j, right_idx in enumerate(right_nodes):
                if conflict[left_idx, right_idx]:
                    cap[1 + i, 1 + n_left + j] = INF
        
        # Run max-flow using scipy
        from scipy.sparse.csgraph import maximum_bipartite_matching
        # Actually, scipy doesn't have weighted max-flow directly in csgraph
        # Let's implement a simple max-flow using Edmonds-Karp or use networkx if available
        # For now, let's use a simple augmenting path algorithm
        
        def max_flow(cap_matrix, src, dst):
            n = len(cap_matrix)
            flow = np.zeros((n, n), dtype=np.float64)
            total_flow = 0.0
            
            while True:
                # BFS to find augmenting path
                visited = [-1] * n
                visited[src] = -2  # mark source
                queue = [src]
                
                while queue:
                    u = queue.pop(0)
                    if u == dst:
                        break
                    for v in range(n):
                        if visited[v] == -1 and cap_matrix[u, v] - flow[u, v] > 1e-9:
                            visited[v] = u
                            queue.append(v)
                
                if visited[dst] == -1:
                    break  # no augmenting path
                
                # Find bottleneck
                v = dst
                bottleneck = float('inf')
                while v != src:
                    u = visited[v]
                    bottleneck = min(bottleneck, cap_matrix[u, v] - flow[u, v])
                    v = u
                
                # Update flows
                v = dst
                while v != src:
                    u = visited[v]
                    flow[u, v] += bottleneck
                    flow[v, u] -= bottleneck
                    v = u
                
                total_flow += bottleneck
            
            return total_flow
        
        try:
            min_cut_weight = max_flow(cap, source, sink)
            mwis_weight = total_weight - min_cut_weight
            
            # Now we need to extract the actual independent set
            # The min cut partitions nodes into S (source side) and T (sink side)
            # Independent set = (left_nodes not in S) union (right_nodes in S)
            # Wait, let me reconsider.
            
            # Actually, for the standard reduction:
            # Min vertex cover = (left nodes in T) union (right nodes in S)
            # where S is the set reachable from source in residual graph
            
            # Let me re-derive. The max-flow min-cut theorem gives us a min cut.
            # The min vertex cover in bipartite graph G=(L,R,E) with weights:
            # V_cover = (L ∩ T) ∪ (R ∩ S) where (S,T) is min cut
            # Independent set = V \ V_cover = (L ∩ S) ∪ (R ∩ T)
            
            # Find S: nodes reachable from source in residual graph
            residual = cap - flow
            S = set()
            queue = [source]
            S.add(source)
            while queue:
                u = queue.pop(0)
                for v in range(n_nodes):
                    if v not in S and residual[u, v] > 1e-9:
                        S.add(v)
                        queue.append(v)
            
            # Independent set nodes
            indep_set = []
            for i, node_idx in enumerate(left_nodes):
                if (1 + i) in S:  # left node in S
                    indep_set.append(node_idx)
            for j, node_idx in enumerate(right_nodes):
                if (1 + n_left + j) not in S:  # right node in T
                    indep_set.append(node_idx)
            
            # We need exactly N circles. If we have more, pick the N largest.
            # If we have fewer, we need a different approach.
            if len(indep_set) >= N:
                # Pick the N with largest radii
                radii_in_set = [(candidates[idx][2], idx) for idx in indep_set]
                radii_in_set.sort(reverse=True)
                selected = [idx for _, idx in radii_in_set[:N]]
                
                # Verify no overlaps among selected
                valid = True
                for a in range(len(selected)):
                    for b in range(a + 1, len(selected)):
                        ia, ib = selected[a], selected[b]
                        xa, ya, ra = candidates[ia][0], candidates[ia][1], candidates[ia][2]
                        xb, yb, rb = candidates[ib][0], candidates[ib][1], candidates[ib][2]
                        d = np.sqrt((xa - xb)**2 + (ya - yb)**2)
                        if d < ra + rb - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    sol = np.array([[candidates[idx][0], candidates[idx][1], candidates[idx][2]] for idx in selected])
                    val = np.sum(sol[:, 2])
                    if val > best_value:
                        best_value = val
                        best_solution = sol
        
        except Exception:
            continue
    
    if best_solution is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.001
                    cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_solution
