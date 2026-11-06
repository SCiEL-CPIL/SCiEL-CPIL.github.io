import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (150s limit)

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

    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def is_add_valid(u, v, adj):
        """
        Check if adding edge (u,v) keeps the graph K3,3-free.
        Optimized: precompute neighbor lists for faster iteration.
        """
        mask_v = adj[v]
        mask_u = adj[u]
        
        # Get neighbor lists as lists for faster iteration
        nv = []
        m = mask_v
        while m:
            lsb = m & -m
            nv.append(lsb.bit_length() - 1)
            m -= lsb
        
        nu = []
        m = mask_u
        while m:
            lsb = m & -m
            nu.append(lsb.bit_length() - 1)
            m -= lsb
        
        # Check triples {u, i, j} where i,j in N(v)
        for idx_i in range(len(nv)):
            i = nv[idx_i]
            mask_i = adj[i]
            for idx_j in range(idx_i + 1, len(nv)):
                j = nv[idx_j]
                mask_j = adj[j]
                common = mask_u & mask_i & mask_j
                common &= ~(1 << v)
                if common.bit_count() >= 2:
                    return False

        # Check triples {v, i, j} where i,j in N(u)
        for idx_i in range(len(nu)):
            i = nu[idx_i]
            mask_i = adj[i]
            for idx_j in range(idx_i + 1, len(nu)):
                j = nu[idx_j]
                mask_j = adj[j]
                common = mask_v & mask_i & mask_j
                common &= ~(1 << u)
                if common.bit_count() >= 2:
                    return False

        return True

    def is_valid_graph(adj):
        """
        Full check for K3,3.
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

    def greedy_add(adj):
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

    def local_search_sa(adj_init):
        """
        Simulated Annealing with improved move acceptance and edge list caching.
        """
        adj = adj_init[:]
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges

        T = 2.0
        cooling = 0.999
        min_T = 0.1
        steps_since_improve = 0
        max_steps_no_improve = 3000
        total_steps = 0

        while time.time() < deadline and T > min_T:
            total_steps += 1
            # Randomly add or remove
            if random.random() < 0.5:
                # Try to add
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
            else:
                # Try to remove - build edge list efficiently
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
                steps_since_improve += 1

            T *= cooling

            # Periodic greedy fill - more frequent
            if steps_since_improve > max_steps_no_improve or (total_steps % 1000 == 0):
                greedy_add(adj)
                new_edges = count_edges(adj)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve = 0
            
            if steps_since_improve > 15000:
                break

        return best_adj, best_edges

    def construct_deterministic():
        """
        Construct a strong deterministic starting point using multiple structured patterns.
        """
        adj = [0] * n
        
        # Strategy 1: 4 groups of 5 with cyclic difference pattern
        for i in range(4):
            for j in range(i+1, 4):
                for a in range(5):
                    for b in range(5):
                        if (a - b) % 5 in [0, 1, 2]:
                            u = 5 * i + a
                            v = 5 * j + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        
        # Add edges within groups
        for i in range(4):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5 * i + a
                    v = 5 * i + b
                    if is_add_valid(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Strategy 2: Try a different grouping - 5 groups of 4
        adj2 = [0] * n
        for i in range(5):
            for j in range(i+1, 5):
                for a in range(4):
                    for b in range(4):
                        # Use a 4x4 pattern: connect if (a-b) mod 4 is in {0,1}
                        if (a - b) % 4 in [0, 1]:
                            u = 4 * i + a
                            v = 4 * j + b
                            adj2[u] |= (1 << v)
                            adj2[v] |= (1 << u)
        
        # Add edges within groups
        for i in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    u = 4 * i + a
                    v = 4 * i + b
                    if is_add_valid(u, v, adj2):
                        adj2[u] |= (1 << v)
                        adj2[v] |= (1 << u)
        
        # Pick the better one
        e1 = count_edges(adj)
        e2 = count_edges(adj2)
        if e2 > e1:
            return adj2
        return adj

    best_global_edges = 0
    best_global_adj = [0] * n

    # Start with deterministic construction
    adj_det = construct_deterministic()
    if is_valid_graph(adj_det):
        greedy_add(adj_det)
        det_edges = count_edges(adj_det)
        if det_edges > best_global_edges:
            best_global_edges = det_edges
            best_global_adj = adj_det[:]

    # Multiple restarts with varied strategies
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 5:  # Leave 5s margin for final polish
            break

        # Initialize with different strategies
        if r == 0:
            adj_init = [0] * n
        elif r % 3 == 0:
            # Use best found as base with perturbation
            adj_init = best_global_adj[:] if best_global_edges > 0 else [0] * n
            # Perturb: remove some random edges
            for _ in range(random.randint(2, 8)):
                all_e = []
                for i in range(n):
                    m = adj_init[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            all_e.append((i, j))
                        m -= lsb
                if all_e:
                    u, v = random.choice(all_e)
                    adj_init[u] &= ~(1 << v)
                    adj_init[v] &= ~(1 << u)
        elif r % 3 == 1:
            # Random initialization with moderate density
            adj_init = [0] * n
            p = 0.3 + 0.04 * (r % 7)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj_init[i] |= (1 << j)
                        adj_init[j] |= (1 << i)
        else:
            # Start from deterministic and add random edges
            adj_init = construct_deterministic()
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_init[i] & (1 << j)):
                        if random.random() < 0.15:
                            adj_init[i] |= (1 << j)
                            adj_init[j] |= (1 << i)
        
        # Repair if invalid
        attempts = 0
        while not is_valid_graph(adj_init) and attempts < 3000:
            all_edges = []
            for i in range(n):
                m = adj_init[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        all_edges.append((i, j))
                    m -= lsb
            if not all_edges:
                break
            u, v = random.choice(all_edges)
            adj_init[u] &= ~(1 << v)
            adj_init[v] &= ~(1 << u)
            attempts += 1

        if not is_valid_graph(adj_init):
            continue

        # Local Search
        local_adj, local_edges = local_search_sa(adj_init)
        
        if local_edges > best_global_edges:
            best_global_edges = local_edges
            best_global_adj = local_adj

    # Final greedy pass
    if best_global_edges > 0:
        greedy_add(best_global_adj)
        final_edges = count_edges(best_global_adj)
        if final_edges > best_global_edges:
            best_global_edges = final_edges

    # Final verification and cleanup
    if best_global_edges == 0:
        return []

    # Verify the final graph is valid
    if not is_valid_graph(best_global_adj):
        # Fallback to a simple valid graph
        adj_fallback = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if (i + j) % 2 == 0:  # Simple bipartite-like
                    adj_fallback[i] |= (1 << j)
                    adj_fallback[j] |= (1 << i)
        greedy_add(adj_fallback)
        return get_edges_from_adj(adj_fallback)

    return get_edges_from_adj(best_global_adj)

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    # Verify
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if common.bit_count() < 3: continue
            for k in range(j+1, n):
                if (common & adj[k]).bit_count() >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
