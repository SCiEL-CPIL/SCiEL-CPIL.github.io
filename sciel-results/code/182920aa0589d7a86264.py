import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def get_edges_list(adj):
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
        When adding (u,v), new K3,3s can only involve u or v as part of a triple
        with two others, where the 4th vertex completes the bipartite structure.
        
        Key insight: A K3,3 violation from adding (u,v) means there exist
        vertices a1,a2,b1,b2 such that:
        - {u,v,a1} and {b1,b2,a2} form the bipartition, OR
        - {u,v,b1} and {a1,a2,b2} form the bipartition, etc.
        
        More precisely: after adding (u,v), we need no triple {x,y,z} to have
        3 common neighbors. The new edge (u,v) creates new common neighbors:
        - v becomes common neighbor of u and any neighbor of u
        - u becomes common neighbor of v and any neighbor of v
        
        So we check: for any pair (i,j) in N(v), does {u,i,j} have >=3 common neighbors?
        Common neighbors of {u,i,j} after adding (u,v) = {v} ∪ (N(u)∩N(i)∩N(j) \ {v})
        So we need |N(u)∩N(i)∩N(j) \ {v}| >= 2, i.e., (adj[u]&adj[i]&adj[j]&~(1<<v)).bit_count() >= 2
        
        Similarly for pairs in N(u) with v.
        """
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
                mask_j = adj[j]
                common = adj[u] & mask_i & mask_j
                common &= ~(1 << v)
                if common.bit_count() >= 2:
                    return False

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
                mask_j = adj[j]
                common = adj[v] & mask_i & mask_j
                common &= ~(1 << u)
                if common.bit_count() >= 2:
                    return False

        return True

    def is_valid_graph(adj):
        """
        Full check for K3,3: no three vertices have three common neighbors.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a K3,3 violation if exists, return list of 9 edges or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        # Get 3 common neighbors
                        cn = list(common_ijk)
                        b1, b2, b3 = cn[0], cn[1], cn[2]
                        return [(i, b1), (i, b2), (i, b3),
                                (j, b1), (j, b2), (j, b3),
                                (k, b1), (k, b2), (k, b3)]
        return None

    def repair(adj):
        """Remove edges to make graph valid."""
        attempts = 0
        while not is_valid_graph(adj) and attempts < 5000:
            viol = find_violation(adj)
            if viol is None:
                break
            e = random.choice(viol)
            u, v = e
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            attempts += 1
        return adj

    def greedy_add(adj):
        """Try to add edges greedily until no more can be added."""
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

    def simulated_annealing(adj_init, T_start=2.0, cooling=0.9997, min_T=0.03, max_steps=200000):
        """
        Simulated annealing with bitset-based fast validation.
        Moves: add edge (if valid), remove edge (always accepted with Metropolis),
        swap: remove edge e1, add edge e2 if valid.
        """
        adj = adj_init[:]
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges
        T = T_start
        steps = 0
        steps_since_improve = 0
        max_no_improve = 8000

        while time.time() < deadline and T > min_T and steps < max_steps:
            steps += 1
            r = random.random()

            if r < 0.35:
                # Try to add a random edge
                u, v = random.sample(range(n), 2)
                if adj[u] & (1 << v):
                    continue
                if is_add_valid(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    steps_since_improve += 1
            elif r < 0.65:
                # Try to remove a random edge (kick)
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
                    continue
                u, v = random.choice(all_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                # Metropolis: accept with probability exp(-1/T)
                if random.random() > (0.3679 ** (1.0 / T)):  # exp(-1/T)
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                steps_since_improve += 1
            else:
                # Swap: remove one edge, try to add another
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
                    continue
                u1, v1 = random.choice(all_edges)
                # Remove
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                current_edges -= 1
                
                # Try to add a different edge
                u2, v2 = random.sample(range(n), 2)
                if adj[u2] & (1 << v2):
                    # Revert removal
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    current_edges += 1
                    continue
                if is_add_valid(u2, v2, adj):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    current_edges += 1
                    # Net change: 0 (removed one, added one)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    # Revert removal
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    current_edges += 1
                    steps_since_improve += 1

            T *= cooling

            # Periodic greedy fill to escape local optima
            if steps % 2000 == 0 or steps_since_improve > max_no_improve:
                greedy_add(adj)
                new_edges = count_edges(adj)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve = 0

            if steps_since_improve > max_no_improve * 2:
                break

        return best_adj, best_edges

    # Multiple restarts with different initializations
    best_global_adj = [0] * n
    best_global_edges = 0

    random.seed(42)

    # Start with empty graph and greedy fill
    adj = [0] * n
    adj = greedy_add(adj)
    best_global_adj = adj[:]
    best_global_edges = count_edges(adj)

    # Run SA from various starting points
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r == 0:
            adj_init = best_global_adj[:]
        else:
            adj_init = [0] * n
            # Vary density
            p = 0.2 + 0.04 * (r % 7)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj_init[i] |= (1 << j)
                        adj_init[j] |= (1 << i)
            adj_init = repair(adj_init)
            adj_init = greedy_add(adj_init)

        best_adj, best_edges = simulated_annealing(adj_init)
        if best_edges > best_global_edges:
            best_global_edges = best_edges
            best_global_adj = best_adj

    # Final greedy polish
    final_adj = greedy_add(best_global_adj[:])
    final_edges = count_edges(final_adj)
    if final_edges > best_global_edges:
        best_global_adj = final_adj
        best_global_edges = final_edges

    # Convert to list of edges
    result = get_edges_list(best_global_adj)
    return result
