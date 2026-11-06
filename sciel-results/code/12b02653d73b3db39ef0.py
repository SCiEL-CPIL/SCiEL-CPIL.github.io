import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Bit-mask helpers ---
    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def is_add_valid(u, v, adj):
        """
        Check if adding edge (u,v) keeps the graph K3,3-free.
        After adding (u,v):
          - For any pair i,j in N(v): triple {u,i,j} gains v as common neighbor.
            Need |N(u)\{v} ∩ N(i) ∩ N(j)| >= 2 to create K3,3.
          - For any pair i,j in N(u): triple {v,i,j} gains u as common neighbor.
            Need |N(v)\{u} ∩ N(i) ∩ N(j)| >= 2 to create K3,3.
        """
        # Triples containing u: i, j in N(v)
        mask_v = adj[v]
        i_list = []
        m = mask_v
        while m:
            lsb = m & -m
            i_list.append(lsb.bit_length() - 1)
            m -= lsb
        for idx_i in range(len(i_list)):
            i = i_list[idx_i]
            mask_i = adj[i]
            for idx_j in range(idx_i + 1, len(i_list)):
                j = i_list[idx_j]
                common = adj[u] & mask_i & adj[j]
                common &= ~(1 << v)
                if common.bit_count() >= 2:
                    return False

        # Triples containing v: i, j in N(u)
        mask_u = adj[u]
        i_list = []
        m = mask_u
        while m:
            lsb = m & -m
            i_list.append(lsb.bit_length() - 1)
            m -= lsb
        for idx_i in range(len(i_list)):
            i = i_list[idx_i]
            mask_i = adj[i]
            for idx_j in range(idx_i + 1, len(i_list)):
                j = i_list[idx_j]
                common = adj[v] & mask_i & adj[j]
                common &= ~(1 << u)
                if common.bit_count() >= 2:
                    return False
        return True

    def is_valid_graph(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def greedy_fill(adj):
        """Add all safe edges in order."""
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_add_valid(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    def find_k33_edges(adj):
        """Return a list of edges in some K3,3, or None."""
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj[i] & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        # Found K3,3 with parts {i,j,k} and 3 from common_ijk
                        cl = []
                        m = common_ijk
                        while m and len(cl) < 3:
                            lsb = m & -m
                            cl.append(lsb.bit_length() - 1)
                            m -= lsb
                        edges = []
                        for a in (i, j, k):
                            for b in cl:
                                edges.append((a, b))
                        return edges
        return None

    def repair(adj):
        """Remove edges until K3,3-free."""
        while True:
            k33 = find_k33_edges(adj)
            if k33 is None:
                break
            u, v = random.choice(k33)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        return adj

    # --- Deterministic 5x4 group construction ---
    def build_group_construction():
        """
        Partition 20 vertices into 5 groups of 4.
        Connect groups in a pattern that avoids K3,3.
        Use a 5x5 block matrix where block (a,b) has a fixed edge pattern.
        """
        adj = [0] * n
        groups = [[4 * g + r for r in range(4)] for g in range(5)]

        # Define a 5x5 pattern of edge densities between groups
        # Use a structured pattern: connect group g to group h if (g - h) mod 5 in {1, 2, 3, 4}
        # but with a specific intra-block pattern to avoid K3,3
        # Simpler: use a known K3,3-free bipartite structure between groups
        # Each pair of groups: connect with a 4x4 bipartite graph that is K3,3-free
        # A 4x4 grid (C4 x C4) is K3,3-free and has 8 edges
        # Use a fixed 4x4 pattern: row r connects to col c if (r + c) % 4 in {0, 1}
        # That gives 8 edges per block, and the whole graph should be K3,3-free

        for g in range(5):
            for h in range(5):
                if g == h:
                    continue
                # Only add edges once (g < h)
                if g > h:
                    continue
                # 4x4 bipartite between groups g and h
                # Pattern: connect (r, c) if (r + c) % 4 in {0, 1}
                for r in range(4):
                    for c in range(4):
                        if (r + c) % 4 in (0, 1):
                            u = groups[g][r]
                            v = groups[h][c]
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj

    # --- Simulated Annealing with 1-edge swaps ---
    def sa_search(adj_init):
        adj = adj_init[:]
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges

        T = 2.0
        cooling = 0.9997
        min_T = 0.01
        steps_since_improve = 0
        max_steps_no_improve = 8000

        while time.time() < deadline and T > min_T:
            # Move: swap a random edge (remove one, add another)
            u, v = random.sample(range(n), 2)
            if adj[u] & (1 << v):
                # Remove edge (u,v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                # Try to add a random non-edge
                u2, v2 = random.sample(range(n), 2)
                if not (adj[u2] & (1 << v2)):
                    if is_add_valid(u2, v2, adj):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        current_edges += 1
                    else:
                        # Revert the removal
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        T *= cooling
                        continue
            else:
                # Add edge (u,v) if valid
                if is_add_valid(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                else:
                    # Try removing a random edge and adding (u,v)
                    all_edges = []
                    for i in range(n):
                        m = adj[i]
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            if i < j:
                                all_edges.append((i, j))
                            m -= lsb
                    if not all_edges:
                        T *= cooling
                        continue
                    ru, rv = random.choice(all_edges)
                    # Remove (ru, rv)
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    current_edges -= 1
                    # Try to add (u, v)
                    if not (adj[u] & (1 << v)):
                        if is_add_valid(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            current_edges += 1
                        else:
                            # Revert
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
                            current_edges += 1
                            T *= cooling
                            continue
                    else:
                        # Revert
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        current_edges += 1
                        T *= cooling
                        continue

            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1

            T *= cooling

            # Periodic greedy fill
            if steps_since_improve > 500 or random.random() < 0.03:
                greedy_fill(adj)
                new_edges = count_edges(adj)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
                steps_since_improve = 0

            if steps_since_improve > max_steps_no_improve:
                break

        return best_adj, best_edges

    # --- Main search loop with multiple restarts ---
    best_global_adj = None
    best_global_edges = 0

    # Start with deterministic group construction
    adj_init = build_group_construction()
    repair(adj_init)
    greedy_fill(adj_init)
    e0 = count_edges(adj_init)
    best_global_adj = adj_init[:]
    best_global_edges = e0

    # SA from the group construction
    adj_sa, e_sa = sa_search(adj_init[:])
    if e_sa > best_global_edges:
        best_global_edges = e_sa
        best_global_adj = adj_sa

    # Additional restarts with random initializations
    for r in range(50):
        if time.time() > deadline:
            break
        adj_r = [0] * n
        p = 0.3 + 0.04 * (r % 5)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_r[i] |= (1 << j)
                    adj_r[j] |= (1 << i)
        repair(adj_r)
        greedy_fill(adj_r)
        e_r = count_edges(adj_r)
        if e_r > best_global_edges:
            best_global_edges = e_r
            best_global_adj = adj_r[:]
        # SA from this random start
        adj_sa2, e_sa2 = sa_search(adj_r[:])
        if e_sa2 > best_global_edges:
            best_global_edges = e_sa2
            best_global_adj = adj_sa2

    # Final greedy fill on best
    greedy_fill(best_global_adj)
    final_edges = count_edges(best_global_adj)
    if final_edges > best_global_edges:
        best_global_edges = final_edges

    return get_edges_from_adj(best_global_adj)
