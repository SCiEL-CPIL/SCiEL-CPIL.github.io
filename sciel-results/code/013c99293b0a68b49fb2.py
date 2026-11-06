"""Rewritten program for R(3,10) triangle-free graph construction."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds; leave safety margin before 120s limit

    best_n = 0
    best_edges = []

    def check_triangle_free(n, adj):
        """Check if graph is triangle-free using adjacency sets."""
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        return True

    def max_independent_set_size(n, adj):
        """Exact branch and bound for maximum independent set size."""
        best = [0]

        def branch(remaining, size):
            if time.time() - start_time > DEADLINE - 5:
                return
            if not remaining:
                if size > best[0]:
                    best[0] = size
                return
            if size + len(remaining) <= best[0]:
                return
            # Pick vertex with max degree in remaining subgraph
            r = list(remaining)
            v = max(r, key=lambda x: sum(1 for y in r if y != x and y in adj[x]))
            # Branch 1: include v
            new_remaining = set(r) - {v} - adj[v]
            branch(new_remaining, size + 1)
            # Branch 2: exclude v
            new_remaining2 = set(r) - {v}
            branch(new_remaining2, size)

        branch(set(range(n)), 0)
        return best[0]

    def is_valid(n, edges):
        """Check both conditions: no triangle and no independent set of 10."""
        if time.time() - start_time > DEADLINE - 5:
            return False
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        if not check_triangle_free(n, adj):
            return False
        alpha = max_independent_set_size(n, adj)
        return alpha < 10

    def edges_to_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # --- Candidate 1: Andrasfai graph And(9) on 26 vertices ---
    n1 = 26
    edges1 = [[i, j] for i in range(n1) for j in range(i + 1, n1) if (j - i) % n1 % 3 == 1]
    if is_valid(n1, edges1):
        best_n = n1
        best_edges = edges1
        print(f"Found valid graph with n={n1}")

    # --- Try to extend by adding vertices with careful edge choices ---
    def try_extend(base_n, base_edges, max_add=15):
        """Try to extend a valid graph by adding new vertices."""
        n = base_n
        edges = list(base_edges)
        adj = edges_to_adj(n, edges)
        
        for _ in range(max_add):
            if time.time() - start_time > DEADLINE - 10:
                break
            new_v = n
            found = False
            
            # Try multiple strategies for connecting new vertex
            strategies = []
            
            # Strategy 1: Connect to all vertices that form an independent set with each other
            # (i.e., no two connected vertices are adjacent)
            connected_to = set()
            for u in range(n):
                can_connect = all(v not in adj[u] for v in connected_to)
                if can_connect:
                    connected_to.add(u)
            if connected_to:
                strategies.append(connected_to.copy())
            
            # Strategy 2: Connect to a random subset that's triangle-free
            for trial in range(5):
                random.seed(int(time.time() * 1000) + trial + n)
                conn = set()
                for u in range(n):
                    if random.random() < 0.5:
                        can_connect = all(v not in adj[u] for v in conn)
                        if can_connect:
                            conn.add(u)
                if conn:
                    strategies.append(conn.copy())
            
            # Strategy 3: Connect to every k-th vertex (cyclic pattern)
            for k in range(2, min(n, 8) + 1):
                conn = set(range(0, n, k))
                # Verify triangle-free
                valid_conn = True
                for u in conn:
                    for v in conn:
                        if u != v and v in adj[u]:
                            valid_conn = False
                            break
                    if not valid_conn:
                        break
                if valid_conn and conn:
                    strategies.append(conn.copy())
            
            # Try each strategy
            for conn in strategies:
                if time.time() - start_time > DEADLINE - 10:
                    break
                new_edges = edges + [[new_v, u] for u in conn]
                new_adj = edges_to_adj(n + 1, new_edges)
                if check_triangle_free(n + 1, new_adj):
                    if max_independent_set_size(n + 1, new_adj) < 10:
                        n += 1
                        edges = new_edges
                        adj = new_adj
                        found = True
                        break
            
            if not found:
                break
        
        return n, edges

    n2, edges2 = try_extend(n1, edges1, max_add=15)
    if n2 > best_n:
        best_n = n2
        best_edges = edges2
        print(f"Extended to n={n2}")

    # --- Spectral construction + simulated annealing ---
    def spectral_sa_search(target_n, time_budget):
        """
        Build a candidate graph using spectral methods, then use simulated annealing
        to eliminate triangles while keeping alpha < 10.
        """
        if time.time() - start_time > DEADLINE - 10:
            return 0, []
        
        n = target_n
        # Initial graph: use a structured pattern based on modular arithmetic
        # Inspired by Paley-type designs
        adj = [set() for _ in range(n)]
        
        # Start with a sparse structured graph
        # Use multiple modular distances for good expansion
        for i in range(n):
            for d in [2, 3, 5, 7, 11]:
                j = (i + d) % n
                if j > i and j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Simulated annealing to optimize
        # Objective: minimize triangles + penalty if alpha >= 10
        # We want: 0 triangles AND alpha < 10
        
        def count_triangles(n, adj):
            cnt = 0
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        for k in adj[j]:
                            if k > j and k in adj[i]:
                                cnt += 1
            return cnt
        
        # Check if current state is valid
        def is_state_valid(n, adj):
            if count_triangles(n, adj) > 0:
                return False
            alpha = max_independent_set_size(n, adj)
            return alpha < 10
        
        # Annealing parameters
        T_start = 5.0
        T_end = 0.01
        cooling = 0.995
        
        current_triangles = count_triangles(n, adj)
        current_alpha = max_independent_set_size(n, adj)
        
        T = T_start
        step = 0
        max_steps = 50000
        
        while T > T_end and step < max_steps:
            if time.time() - start_time > DEADLINE - 10:
                break
            step += 1
            
            # Choose a random edge to flip (add or remove)
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            adding = (j not in adj[i])
            
            # Check triangle impact if adding
            if adding:
                # Adding (i,j) creates triangles with common neighbors
                common = adj[i] & adj[j]
                new_triangles = current_triangles + len(common)
                # Also check if this creates new independent set issues
                # (hard to compute quickly, so we'll check later)
                
                # Accept with probability based on energy change
                delta_E = len(common)  # increase in triangles
                if delta_E == 0 or random.random() < math.exp(-delta_E / T):
                    adj[i].add(j)
                    adj[j].add(i)
                    current_triangles = new_triangles
                    # Periodically check alpha
                    if step % 100 == 0:
                        current_alpha = max_independent_set_size(n, adj)
            else:
                # Removing edge (i,j)
                # Check if it's part of any triangle
                common = adj[i] & adj[j]
                new_triangles = current_triangles - len(common)
                
                # Removing edges can increase alpha, so be careful
                # Accept if it doesn't make things worse
                if new_triangles <= current_triangles:
                    adj[i].discard(j)
                    adj[j].discard(i)
                    current_triangles = new_triangles
                    if step % 100 == 0:
                        current_alpha = max_independent_set_size(n, adj)
            
            T *= cooling
        
        # Final check
        if current_triangles == 0 and current_alpha < 10:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        edges.append([i, j])
            return n, edges
        return 0, []

    # Try spectral SA for larger n
    for target_n in range(max(best_n + 1, 28), 41):
        if time.time() - start_time > DEADLINE - 15:
            break
        n_sa, edges_sa = spectral_sa_search(target_n, 30)
        if n_sa > best_n:
            best_n = n_sa
            best_edges = edges_sa
            print(f"Spectral SA found n={n_sa}")
            break

    # --- Random search with varied densities (kept as fallback) ---
    def random_search(target_n, iterations=3000):
        """Randomly generate triangle-free graphs and check alpha."""
        for it in range(iterations):
            if time.time() - start_time > DEADLINE - 10:
                break
            n = target_n
            adj = [set() for _ in range(n)]
            edges = []
            
            random.seed(12345 + it)
            # Vary the edge probability
            p = 0.2 + 0.4 * (it % 5) / 4.0  # 0.2, 0.3, 0.4, 0.5, 0.6
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        # Check if adding (i,j) creates a triangle
                        creates_triangle = False
                        for k in range(n):
                            if k != i and k != j and k in adj[i] and k in adj[j]:
                                creates_triangle = True
                                break
                        if not creates_triangle:
                            adj[i].add(j)
                            adj[j].add(i)
                            edges.append([i, j])
            
            if check_triangle_free(n, adj):
                alpha = max_independent_set_size(n, adj)
                if alpha < 10:
                    return n, edges
        return 0, []

    # Try to find graphs with n > best_n
    for target_n in range(best_n + 1, 41):
        if time.time() - start_time > DEADLINE - 15:
            break
        n_rand, edges_rand = random_search(target_n, iterations=2000)
        if n_rand > best_n:
            best_n = n_rand
            best_edges = edges_rand
            print(f"Random search found n={n_rand}")
            break

    if best_n == 0:
        # Fallback to And(9)
        best_n = n1
        best_edges = edges1

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
