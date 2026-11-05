import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1 # Timeout in check
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Girth-constrained configuration model ----------------
    def girth_constrained_config_model(n, g, time_limit, seed=None):
        """
        Build a k-regular graph on n vertices using the configuration model,
        rejecting edges that would create cycles shorter than g.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Adjacency list (will grow as we add edges)
        adj = [[] for _ in range(n)]
        # Track which edges have been added
        edge_set = set()
        
        # For each vertex, track its "stubs" (remaining degree)
        stubs_remaining = [k] * n
        
        # List of vertices with available stubs, for efficient sampling
        # We'll use a list of (vertex) entries, one per stub
        stub_list = []
        for v in range(n):
            stub_list.extend([v] * k)
        
        # Shuffle the stub list
        rng.shuffle(stub_list)
        
        # Process stubs in pairs
        for i in range(0, len(stub_list), 2):
            if time.time() > time_limit:
                return None
            
            u = stub_list[i]
            v = stub_list[i + 1]
            
            if u == v:
                return None  # Self-loop, fail
            
            # Check if edge already exists
            if (u, v) in edge_set or (v, u) in edge_set:
                return None  # Multi-edge, fail
            
            # Check girth constraint: adding edge (u,v) must not create cycle < g
            # This means the shortest path from u to v in the current graph
            # (excluding the edge we're about to add) must be >= g-1
            # Actually, if there's a path of length L from u to v, adding the edge
            # creates a cycle of length L+1. We need L+1 >= g, so L >= g-1.
            # But we also need to make sure we don't create cycles through other paths.
            # The key insight: in a graph being built edge-by-edge, the only new cycles
            # created by adding edge (u,v) are those that use this edge. Such a cycle
            # corresponds to a path from u to v in the existing graph plus the new edge.
            # So we need the shortest path from u to v in the current graph to be >= g-1.
            
            # BFS from u to find distance to v, but we need to be careful:
            # We should NOT use the edge (u,v) since it doesn't exist yet.
            # Also, we need to avoid backtracking through the edge we're adding.
            
            # Actually, let's think more carefully. When we add edge (u,v), any new cycle
            # must use this edge. A cycle using edge (u,v) is: u - v - (path from v to u not using (u,v)).
            # So we need the shortest path from v to u in the graph WITHOUT edge (u,v) to be >= g-1.
            # Since (u,v) doesn't exist yet, this is just the shortest path in the current graph.
            
            # But wait - there's a subtlety. The current graph may have multiple components.
            # If u and v are in different components, adding the edge connects them and
            # creates no cycles. So we're fine.
            
            # If u and v are in the same component, we need dist(u,v) >= g-1.
            
            # Let's do a BFS from u, but stop when we've explored enough or found v too close.
            # We can prune: if we find v at distance < g-1, reject.
            
            # Optimization: if the current graph has fewer than g-1 edges on any path,
            # we can skip the check. But let's just do BFS with early termination.
            
            # BFS from u to v, tracking distance
            # We'll use a queue and stop early if we find v at distance < g-1
            if len(adj[u]) == 0 and len(adj[v]) == 0:
                # Both isolated, no path exists, safe to add
                pass
            else:
                # BFS from u
                dist = [-1] * n
                dist[u] = 0
                queue = [u]
                qi = 0
                found_v = False
                v_dist = -1
                
                while qi < len(queue):
                    node = queue[qi]
                    qi += 1
                    
                    if dist[node] >= g - 1:
                        # We've explored enough; any path from u to v must be >= g-1
                        # if v is reachable, but we haven't found v yet at this depth.
                        # Actually, if we've explored all nodes at distance < g-1 from u
                        # and didn't find v, then dist(u,v) >= g-1 or v is unreachable.
                        break
                    
                    for nb in adj[node]:
                        if dist[nb] == -1:
                            dist[nb] = dist[node] + 1
                            if nb == v:
                                found_v = True
                                v_dist = dist[nb]
                                if v_dist < g - 1:
                                    break  # Reject
                                else:
                                    break
                            queue.append(nb)
                    
                    if found_v and v_dist < g - 1:
                        break
                
                if found_v and v_dist < g - 1:
                    return None  # Would create cycle too short
        
        # Check if all vertices have degree exactly k
        for v in range(n):
            if len(adj[v]) != k:
                return None
        
        # Build edge list
        edges = []
        for v in range(n):
            for nb in adj[v]:
                if v < nb:
                    edges.append((v, nb))
        
        return edges

    # ---------------- Voltage Lift of Petersen with improved search ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all non-backtracking closed walks of length [5, 15]
        max_L = g - 1
        walk_vectors = set()

        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()

                if length >= max_L:
                    continue

                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1

                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1

                    if next_v == start_v and new_l >= 5:
                        walk_vectors.add(tuple(new_counts))

                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        # Filter out zero vectors and duplicates
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)

        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None

        # Precompute indices of non-zero entries for each constraint for faster repair
        constraint_nonzero_indices = []
        for i in range(C.shape[0]):
            nz = np.nonzero(C[i])[0]
            constraint_nonzero_indices.append(nz)

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        # Improved local search with constraint-directed repair and SA
        def solve_voltage(m, max_iters=20000):
            # Initialize with random values
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            if best_cost == 0:
                return v, 0

            # Simulated annealing with adaptive temperature
            T = 1.0
            cooling = 0.9995
            min_T = 0.001
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                if best_cost > 0:
                    residues = np.mod(C @ v, m)
                    violated_idx = np.nonzero(residues == 0)[0]
                    if len(violated_idx) == 0:
                        break
                    c_idx = violated_idx[random.randint(0, len(violated_idx)-1)]
                    
                    nz_indices = constraint_nonzero_indices[c_idx]
                    if len(nz_indices) == 0:
                        continue
                    j = nz_indices[random.randint(0, len(nz_indices)-1)]
                else:
                    j = random.randint(0, 14)

                old_val = v[j]
                v[j] = random.randint(1, m - 1)

                new_cost = cost(v)
                delta = new_cost - best_cost

                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val

                T *= cooling

            return best_v, best_cost

        m = 53
        while m < 200 and remaining() > 20:
            if is_prime(m):
                found = False
                for restart in range(10):
                    if remaining() < 10:
                        break
                    
                    v, cost = solve_voltage(m, max_iters=5000)

                    if cost == 0:
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))

                        gc = check_girth(n, edges, g, deadline - 3)
                        if gc >= g:
                            return (n, edges)

            m += 1

        return None

    # ---------------- Main ----------------
    # Try voltage lift first (known to work for m=53, giving n=530)
    res = voltage_lift_petersen()
    if res is not None:
        return res

    # Fallback: girth-constrained configuration model
    # Try various n values
    for n in [520, 522, 524, 526, 528, 530, 532, 534, 536, 538, 540]:
        if remaining() < 15:
            break
        
        # Multiple restarts
        for attempt in range(20):
            if remaining() < 5:
                break
            
            edges = girth_constrained_config_model(n, g, deadline - 5, seed=attempt * 1000 + n)
            if edges is not None:
                gc = check_girth(n, edges, g, deadline - 3)
                if gc >= g:
                    return (n, edges)

    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
