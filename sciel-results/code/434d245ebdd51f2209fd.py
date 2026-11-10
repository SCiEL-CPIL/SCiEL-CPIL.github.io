import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    n = 31
    target_edges = 91

    # Build polarity graph of PG(2,5) as initial structure
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1

    # Precompute A2 = adj @ adj for fast C4 counting
    def compute_A2(adj_matrix):
        return adj_matrix @ adj_matrix

    def count_c4s_from_A2(A2, adj_matrix):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i,j] == 0:
                    k = A2[i,j]
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    A2 = compute_A2(adj)
    current_c4s = count_c4s_from_A2(A2, adj)

    # Fast incremental update of A2 when adding/removing edge (u,v)
    def update_A2_add_edge(A2_mat, adj_matrix, u, v):
        # Adding edge (u,v): A2[i][j] changes for all i,j
        # A2_new[i][j] = A2[i][j] + adj[i][u]*adj[v][j] + adj[i][v]*adj[u][j]
        # But since we're adding (u,v), adj[u][v] and adj[v][u] become 1
        # For the purpose of C4 counting, we need A2 for non-edges
        # Simpler: just recompute affected entries
        # A2[i][j] += adj[i][u] * 1 * adj[v][j] + adj[i][v] * 1 * adj[u][j]
        # But adj[u][v] is now 1, so:
        # A2[i][j] += adj[i][u]*adj[v][j] + adj[i][v]*adj[u][j]
        # Note: this includes i=u or j=u etc.
        u_col = adj_matrix[u].copy()
        v_col = adj_matrix[v].copy()
        # Set the new edge
        u_col[v] = 1
        v_col[u] = 1
        # A2[i][j] += u_col[i] * v_col[j] + v_col[i] * u_col[j]
        A2_mat += np.outer(u_col, v_col) + np.outer(v_col, u_col)
        return A2_mat

    def update_A2_remove_edge(A2_mat, adj_matrix, u, v):
        u_col = adj_matrix[u].copy()
        v_col = adj_matrix[v].copy()
        u_col[v] = 1
        v_col[u] = 1
        A2_mat -= np.outer(u_col, v_col) + np.outer(v_col, u_col)
        return A2_mat

    def count_new_c4s_add_fast(u, v, adj_matrix, A2_mat):
        """Count C4s created by adding edge (u,v).
        A C4 using edge (u,v) is u-v-a-b-u where a~v, b~u, a~b, a!=b, a!=u, b!=v.
        This equals the number of edges in the bipartite-like structure between N(v)\{u} and N(u)\{v}.
        Using A2: for each a in N(v)\{u}, count neighbors b of a that are in N(u)\{v,a}.
        Equivalent: sum over a in N(v)\{u} of |N(a) ∩ N(u)| - [a in N(u)] - 1 (for b=v if a~v... no)
        
        Actually: new C4s = number of pairs (a,b) with a~v, b~u, a~b, a≠b, a≠u, b≠v
        = sum_{a: a~v, a≠u} |{b: b~u, b~a, b≠v, b≠a}|
        For fixed a: |N(a) ∩ N(u)| - [v in N(a) and v in N(u)] - [a in N(u)] (since b≠a)
        Since a~v, v in N(a). Is v in N(u)? Yes if u~v, but u~v is what we're adding, so currently no.
        So: sum_{a: a~v, a≠u} (|N(a) ∩ N(u)| - [a in N(u)])
        
        |N(a) ∩ N(u)| = A2[a][u] (number of common neighbors of a and u)
        [a in N(u)] = adj[a][u]
        
        So new_c4s = sum_{a: a~v, a≠u} (A2[a][u] - adj[a][u])
        """
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if adj_matrix[v, a] == 1:
                count += A2_mat[a, u] - adj_matrix[a, u]
        return count

    def count_c4s_removed_fast(u, v, adj_matrix, A2_mat):
        """Count C4s that use edge (u,v) and would be removed.
        Same formula as count_new_c4s_add_fast since the structure is symmetric.
        """
        return count_new_c4s_add_fast(u, v, adj_matrix, A2_mat)

    # Greedy: add edges to reach 91, minimizing new C4s
    edge_count = np.sum(adj) // 2
    while edge_count < target_edges:
        if time.time() > deadline:
            break

        best_edge = None
        best_cost = float('inf')

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    cost = count_new_c4s_add_fast(i, j, adj, A2)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (i, j)

        if best_edge is None:
            break

        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        A2 = update_A2_add_edge(A2, adj, u, v)
        edge_count += 1

    # If we have too many edges, remove greedily
    while edge_count > target_edges:
        if time.time() > deadline:
            break

        best_edge = None
        best_saving = -1

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    saving = count_c4s_removed_fast(i, j, adj, A2)
                    if saving > best_saving:
                        best_saving = saving
                        best_edge = (i, j)

        if best_edge is None:
            break

        u, v = best_edge
        adj[u, v] = 0
        adj[v, u] = 0
        A2 = update_A2_remove_edge(A2, adj, u, v)
        edge_count -= 1

    # Local search: edge swaps to reduce C4s
    # Use simulated annealing style: accept worse moves with some probability
    improved = True
    current_c4s = count_c4s_from_A2(A2, adj)
    best_c4s = current_c4s
    best_adj = adj.copy()
    best_A2 = A2.copy()
    temperature = 5.0
    cooling = 0.995

    while time.time() < deadline:
        # Build lists of edges and non-edges
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
        non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]

        if not edges_list or not non_edges_list:
            break

        # Try a subset of swaps per iteration to save time
        num_swaps_to_try = min(len(edges_list) * len(non_edges_list), 500)
        import random
        random.seed(42)
        swap_indices = random.sample(range(len(edges_list) * len(non_edges_list)), min(num_swaps_to_try, len(edges_list) * len(non_edges_list)))

        for idx in swap_indices:
            if time.time() > deadline:
                break
            ei = idx // len(non_edges_list)
            ni = idx % len(non_edges_list)
            if ei >= len(edges_list) or ni >= len(non_edges_list):
                continue

            (u, v) = edges_list[ei]
            (x, y) = non_edges_list[ni]

            # Remove edge (u,v), add edge (x,y)
            adj[u, v] = 0
            adj[v, u] = 0
            A2 = update_A2_remove_edge(A2, adj, u, v)

            adj[x, y] = 1
            adj[y, x] = 1
            A2 = update_A2_add_edge(A2, adj, x, y)

            new_c4s = count_c4s_from_A2(A2, adj)
            delta_c4 = new_c4s - current_c4s

            # Accept if better, or with probability exp(-delta/T) if worse
            accept = False
            if delta_c4 < 0:
                accept = True
            elif delta_c4 == 0:
                accept = True  # Accept neutral moves for diversity
            else:
                import math
                if temperature > 0.1:
                    prob = math.exp(-delta_c4 / temperature)
                    if random.random() < prob:
                        accept = True

            if accept:
                current_c4s = new_c4s
                if new_c4s < best_c4s:
                    best_c4s = new_c4s
                    best_adj = adj.copy()
                    best_A2 = A2.copy()
            else:
                # Revert
                adj[x, y] = 0
                adj[y, x] = 0
                A2 = update_A2_remove_edge(A2, adj, x, y)

                adj[u, v] = 1
                adj[v, u] = 1
                A2 = update_A2_add_edge(A2, adj, u, v)

        temperature *= cooling

        if temperature < 0.1:
            # Reset temperature for another round
            temperature = 3.0

    # Restore best solution found
    if best_c4s < current_c4s:
        adj = best_adj
        A2 = best_A2
        current_c4s = best_c4s

    # Final polish: try single edge swaps with stricter acceptance
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
        non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]

        for (u, v) in edges_list:
            if time.time() > deadline:
                break
            for (x, y) in non_edges_list:
                if (u, v) == (x, y) or (u, v) == (y, x):
                    continue

                adj[u, v] = 0
                adj[v, u] = 0
                A2 = update_A2_remove_edge(A2, adj, u, v)

                adj[x, y] = 1
                adj[y, x] = 1
                A2 = update_A2_add_edge(A2, adj, x, y)

                new_c4s = count_c4s_from_A2(A2, adj)

                if new_c4s < current_c4s:
                    current_c4s = new_c4s
                    improved = True
                else:
                    adj[x, y] = 0
                    adj[y, x] = 0
                    A2 = update_A2_remove_edge(A2, adj, x, y)

                    adj[u, v] = 1
                    adj[v, u] = 1
                    A2 = update_A2_add_edge(A2, adj, u, v)

        if not improved:
            break

    # Extract edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])

    return result
