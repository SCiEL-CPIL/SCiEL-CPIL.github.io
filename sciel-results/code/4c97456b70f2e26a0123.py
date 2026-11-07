import time
import random
from math import log, exp

def construct():
    n = 15
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    # Precompute all potential edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Bitmask adjacency representation
    # adj_mask[v] is a bitmask of neighbors of v

    def check_valid(adj_mask, n):
        """Check if no simple 3-edge path has endpoints with equal degree."""
        deg = [0] * n
        for v in range(n):
            deg[v] = bin(adj_mask[v]).count('1')
        
        # Precompute degree masks: deg_masks[d] has bit v set if deg[v] == d
        deg_masks = [0] * n
        for v in range(n):
            deg_masks[deg[v]] |= (1 << v)
            
        for u in range(n):
            mask_u = adj_mask[u]
            for v in range(u + 1, n):
                if not (mask_u >> v) & 1:
                    continue
                mask_v = adj_mask[v]
                # s in N(u), s != v
                s_mask = mask_u & ~(1 << v)
                # t in N(v), t != u
                t_mask_base = mask_v & ~(1 << u)
                
                while s_mask:
                    lb = s_mask & (-s_mask)
                    s = lb.bit_length() - 1
                    s_mask ^= lb
                    
                    # t in N(v) \ {u, s}
                    t_mask = t_mask_base & ~(1 << s)
                    
                    # Check if any t in t_mask has deg[t] == deg[s]
                    if t_mask & deg_masks[deg[s]]:
                        return False
        return True

    def count_edges(adj_mask, n):
        total = 0
        for v in range(n):
            total += bin(adj_mask[v]).count('1')
        return total // 2

    def make_bipartite(adj_mask, n, split):
        for v in range(n):
            adj_mask[v] = 0
        for i in range(split):
            for j in range(split, n):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def try_add_edges_greedy(adj_mask, n, deadline):
        """Greedy: try to add each missing edge if it maintains validity."""
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if (adj_mask[i] >> j) & 1:
                    continue
                # Try adding edge (i, j)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                if check_valid(adj_mask, n):
                    pass  # keep it
                else:
                    adj_mask[i] ^= (1 << j)
                    adj_mask[j] ^= (1 << i)
        return adj_mask

    # Local search with edge swaps
    def local_search(adj_mask, n, deadline):
        """Try edge swaps to improve edge count."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Get current edge set
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            # Try 1-for-1 swaps
            for e1 in edges:
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                for e2 in non_edges:
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    # Try swap: remove e1, add e2
                    i1, j1 = e1
                    i2, j2 = e2
                    adj_mask[i1] ^= (1 << j1)
                    adj_mask[j1] ^= (1 << i1)
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask, n):
                        improved = True
                        break
                    else:
                        # Revert
                        adj_mask[i1] |= (1 << j1)
                        adj_mask[j1] |= (1 << i1)
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)

            # Try adding a single edge (no removal)
            if not improved and time.time() < deadline - 5:
                for e2 in non_edges:
                    if time.time() > deadline - 5:
                        break
                    i2, j2 = e2
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask, n):
                        improved = True
                        break
                    else:
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)

        return adj_mask

    # 2-for-2 swaps for more exploration
    def two_for_two_search(adj_mask, n, deadline):
        """Try 2-edge swaps to improve edge count."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            max_edges_to_try = min(len(edges), 60)
            max_non_edges_to_try = min(len(non_edges), 60)
            
            for idx1 in range(max_edges_to_try):
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                for idx2 in range(idx1 + 1, max_edges_to_try):
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    e1 = edges[idx1]
                    e2 = edges[idx2]
                    for idx3 in range(max_non_edges_to_try):
                        if time.time() > deadline - 5:
                            break
                        if improved:
                            break
                        for idx4 in range(idx3 + 1, max_non_edges_to_try):
                            if time.time() > deadline - 5:
                                break
                            if improved:
                                break
                            f1 = non_edges[idx3]
                            f2 = non_edges[idx4]
                            # Remove e1, e2; add f1, f2
                            for (a, b) in [e1, e2]:
                                adj_mask[a] ^= (1 << b)
                                adj_mask[b] ^= (1 << a)
                            for (a, b) in [f1, f2]:
                                adj_mask[a] |= (1 << b)
                                adj_mask[b] |= (1 << a)
                            if check_valid(adj_mask, n):
                                improved = True
                                break
                            else:
                                for (a, b) in [f1, f2]:
                                    adj_mask[a] ^= (1 << b)
                                    adj_mask[b] ^= (1 << a)
                                for (a, b) in [e1, e2]:
                                    adj_mask[a] |= (1 << b)
                                    adj_mask[b] |= (1 << a)

        return adj_mask

    # Random restart with perturbation
    def perturb(adj_mask, n, k):
        """Remove k random edges."""
        adj_mask = list(adj_mask)
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
        if len(edges_list) < k:
            return adj_mask
        random.shuffle(edges_list)
        for (i, j) in edges_list[:k]:
            adj_mask[i] ^= (1 << j)
            adj_mask[j] ^= (1 << i)
        return adj_mask

    # Main search
    best_adj = [0] * n
    best_count = 0

    # Try different bipartite splits
    for split in range(4, 12):
        if time.time() > deadline - 20:
            break
        adj_mask = [0] * n
        make_bipartite(adj_mask, n, split)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        adj_mask = local_search(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # 2-for-2 search from best
    if time.time() < deadline - 15:
        adj_mask = list(best_adj)
        adj_mask = two_for_two_search(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Random restarts with perturbation
    restarts = 0
    while time.time() < deadline - 10 and restarts < 8:
        restarts += 1
        adj_mask = perturb(best_adj, n, random.randint(3, 6))
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        adj_mask = local_search(adj_mask, n, deadline)
        if time.time() < deadline - 8:
            adj_mask = two_for_two_search(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Final greedy fill
    if time.time() < deadline - 3:
        adj_mask = list(best_adj)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Convert to edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([int(i), int(j)])

    return result

