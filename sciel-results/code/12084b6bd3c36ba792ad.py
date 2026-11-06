import random
import time
import itertools
from collections import defaultdict

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all triples for K3,3 checking
    triples = list(itertools.combinations(range(n), 3))

    def count_violations(adj):
        """Returns list of (u1, u2, u3, common_mask)."""
        violations = []
        for u1, u2, u3 in triples:
            common = adj[u1] & adj[u2] & adj[u3]
            if common.bit_count() >= 3:
                violations.append((u1, u2, u3, common))
        return violations

    def violation_edge_freq(adj):
        """Count how many K3,3 violations each edge participates in."""
        freq = defaultdict(int)
        for u1, u2, u3, common in count_violations(adj):
            for c in range(n):
                if (common >> c) & 1:
                    for u in (u1, u2, u3):
                        key = (min(u, c), max(u, c))
                        freq[key] += 1
        return freq

    def is_add_safe(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        A K3,3 is created iff there exist b1,b2 in N(u) and a1,a2 in N(v)∩N(b1)∩N(b2)
        such that all 6 are distinct.
        Optimized: precompute list of neighbors for faster iteration.
        """
        nu = adj[u]
        nv = adj[v]
        # Get neighbor lists as Python lists for faster iteration
        nu_list = []
        tmp = nu
        while tmp:
            bit = (tmp & (-tmp)).bit_length() - 1
            nu_list.append(bit)
            tmp &= tmp - 1
        
        nv_list = []
        tmp = nv
        while tmp:
            bit = (tmp & (-tmp)).bit_length() - 1
            nv_list.append(bit)
            tmp &= tmp - 1
        
        # Check all pairs in N(u)
        for i in range(len(nu_list)):
            b1 = nu_list[i]
            for j in range(i + 1, len(nu_list)):
                b2 = nu_list[j]
                common = nv & adj[b1] & adj[b2]
                mask = (1 << u) | (1 << v) | (1 << b1) | (1 << b2)
                common &= ~mask
                if common.bit_count() >= 2:
                    return False
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def non_edges(adj):
        """Return list of all non-edges (i, j) with i < j."""
        res = []
        for i in range(n):
            mi = adj[i]
            for j in range(i + 1, n):
                if not (mi >> j) & 1:
                    res.append((i, j))
        return res

    def all_edges(adj):
        """Return list of all edges (i, j) with i < j."""
        res = []
        for i in range(n):
            mi = adj[i]
            for j in range(i + 1, n):
                if (mi >> j) & 1:
                    res.append((i, j))
        return res

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def greedy_add(adj):
        """Repeatedly add any safe non-edge until none remain."""
        improved = True
        while improved:
            improved = False
            ne = non_edges(adj)
            random.shuffle(ne)
            for u, v in ne:
                if is_add_safe(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        return adj

    def greedy_add_optimized(adj):
        """Faster greedy: check each non-edge once per pass, no shuffle."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                mi = adj[i]
                for j in range(i + 1, n):
                    if not (mi >> j) & 1:
                        if is_add_safe(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        return adj

    def fix_violations(adj):
        """Remove edges to eliminate all K3,3 violations, prioritizing high-frequency edges."""
        max_iter = 200  # Safety bound
        for _ in range(max_iter):
            freq = violation_edge_freq(adj)
            if not freq:
                break
            best_edge = max(freq, key=freq.get)
            remove_edge(adj, best_edge[0], best_edge[1])
        return adj

    def is_k33_free(adj):
        """Check if graph is K3,3-free."""
        for u1, u2, u3 in triples:
            common = adj[u1] & adj[u2] & adj[u3]
            if common.bit_count() >= 3:
                return False
        return True

    def local_search(adj_init, max_time=None):
        """Full local search: fix, add, 1-edge swaps, then 2-edge swaps."""
        if max_time is None:
            max_time = deadline
        adj = list(adj_init)

        # Phase 1: Fix violations
        adj = fix_violations(adj)

        # Phase 2: Greedy add
        adj = greedy_add(adj)

        edge_count = count_edges(adj)

        # Phase 2.5: Try optimized greedy to see if we can do better
        adj2 = list(adj)
        adj2 = greedy_add_optimized(adj2)
        ec2 = count_edges(adj2)
        if ec2 > edge_count:
            adj = adj2
            edge_count = ec2

        # Phase 3: 1-edge swap local search (remove 1, add 1)
        swap1_improved = True
        while swap1_improved and time.time() < max_time:
            swap1_improved = False
            edges = all_edges(adj)
            random.shuffle(edges)
            for e1 in edges:
                if time.time() >= max_time:
                    break
                remove_edge(adj, e1[0], e1[1])
                ne = non_edges(adj)
                random.shuffle(ne)
                added = None
                for u, v in ne:
                    if is_add_safe(adj, u, v):
                        add_edge(adj, u, v)
                        added = (u, v)
                        break
                if added is not None:
                    new_count = count_edges(adj)
                    if new_count > edge_count:
                        edge_count = new_count
                        swap1_improved = True
                    else:
                        remove_edge(adj, added[0], added[1])
                        add_edge(adj, e1[0], e1[1])
                else:
                    add_edge(adj, e1[0], e1[1])
            # After each swap round, try greedy add
            if time.time() < max_time:
                adj = greedy_add(adj)
                new_count = count_edges(adj)
                if new_count > edge_count:
                    edge_count = new_count
                    swap1_improved = True

        # Phase 4: 2-edge swap local search (limited iterations for speed)
        swap2_improved = True
        max_swap2_rounds = 5
        for _ in range(max_swap2_rounds):
            if not swap2_improved or time.time() >= max_time:
                break
            swap2_improved = False
            edges = all_edges(adj)
            random.shuffle(edges)
            # Limit number of 2-edge swaps tried per round
            max_pairs = min(len(edges) * (len(edges) - 1) // 2, 800)
            pair_list = []
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    pair_list.append((edges[i], edges[j]))
                    if len(pair_list) >= max_pairs:
                        break
                if len(pair_list) >= max_pairs:
                    break
            random.shuffle(pair_list)
            for e1, e2 in pair_list:
                if time.time() >= max_time:
                    break
                remove_edge(adj, e1[0], e1[1])
                remove_edge(adj, e2[0], e2[1])
                ne = non_edges(adj)
                random.shuffle(ne)
                added = []
                for u, v in ne:
                    if is_add_safe(adj, u, v):
                        add_edge(adj, u, v)
                        added.append((u, v))
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    new_count = count_edges(adj)
                    if new_count > edge_count:
                        edge_count = new_count
                        swap2_improved = True
                    else:
                        for u, v in added:
                            remove_edge(adj, u, v)
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])
                else:
                    for u, v in added:
                        remove_edge(adj, u, v)
                    add_edge(adj, e1[0], e1[1])
                    add_edge(adj, e2[0], e2[1])
            adj = greedy_add(adj)
            new_count = count_edges(adj)
            if new_count > edge_count:
                edge_count = new_count
                swap2_improved = True

        return adj, edge_count

    def structured_init():
        """Denser structured init: 5 groups of 4, connect groups i to (i+1)%5 and (i+3)%5,
        plus intra-group edges to boost density."""
        adj = [0] * n
        groups = [[i * 4 + j for j in range(4)] for i in range(5)]
        # Connect group i to groups (i+1)%5 and (i+3)%5 (skip adjacent in cycle for more structure)
        for i in range(5):
            for j in [(i + 1) % 5, (i + 3) % 5]:
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        # Add some intra-group edges (complete within each group of 4)
        for i in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = groups[i][a]
                    v = groups[i][b]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_init2():
        """Alternative: 4 groups of 5, connect i to (i+1)%4 and (i+2)%4, complete within groups."""
        adj = [0] * n
        groups = [[i * 5 + j for j in range(5)] for i in range(4)]
        for i in range(4):
            for j in [(i + 1) % 4, (i + 2) % 4]:
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        # Complete within each group
        for i in range(4):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = groups[i][a]
                    v = groups[i][b]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def random_init(p):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def perturb(adj, k=3):
        """Randomly flip k edges."""
        adj = list(adj)
        all_e = all_edges(adj)
        ne = non_edges(adj)
        for _ in range(k // 2):
            if all_e:
                e = random.choice(all_e)
                remove_edge(adj, e[0], e[1])
        for _ in range(k - k // 2):
            if ne:
                e = random.choice(ne)
                add_edge(adj, e[0], e[1])
        return adj

    best_edges = 0
    best_graph = None

    def update_best(adj):
        nonlocal best_edges, best_graph
        ec = count_edges(adj)
        if ec > best_edges:
            best_edges = ec
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            best_graph = edges

    # Try structured initializations
    for init_func in [structured_init, structured_init2]:
        if time.time() >= deadline:
            break
        adj_struct = init_func()
        adj_ls, ec_ls = local_search(adj_struct)
        update_best(adj_ls)

    # Try random restarts with various densities (reduced count to allow more time per run)
    for p in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        if time.time() >= deadline - 10:
            break
        for _ in range(8):
            if time.time() >= deadline - 10:
                break
            adj_rand = random_init(p)
            adj_ls, ec_ls = local_search(adj_rand)
            update_best(adj_ls)

    # Try perturbations of the best found so far
    if best_graph is not None:
        best_adj = [0] * n
        for e in best_graph:
            u, v = e
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)

        for _ in range(30):
            if time.time() >= deadline - 5:
                break
            adj_pert = perturb(best_adj, k=random.randint(2, 6))
            adj_ls, ec_ls = local_search(adj_pert)
            update_best(adj_ls)

    # Final polish: intensive local search on best graph with tight deadline
    if best_graph is not None and time.time() < deadline - 5:
        best_adj = [0] * n
        for e in best_graph:
            u, v = e
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
        # Run a few more rounds of 1-edge and 2-edge swaps
        for _ in range(5):
            if time.time() >= deadline - 3:
                break
            adj_polish, ec_polish = local_search(best_adj, max_time=deadline - 3)
            if ec_polish > best_edges:
                update_best(adj_polish)
                best_adj = [0] * n
                for e in best_graph:
                    u, v = e
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)

    if best_graph is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    # Final verification and cleanup
    final_adj = [0] * n
    for e in best_graph:
        u, v = e
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    # One last fix pass to ensure no violations
    final_adj = fix_violations(final_adj)
    final_adj = greedy_add(final_adj)
    
    # Verify K3,3-free
    if not is_k33_free(final_adj):
        # Fallback: try to fix more aggressively
        for _ in range(50):
            freq = violation_edge_freq(final_adj)
            if not freq:
                break
            best_edge = max(freq, key=freq.get)
            remove_edge(final_adj, best_edge[0], best_edge[1])
        final_adj = greedy_add(final_adj)
    
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (final_adj[i] >> j) & 1:
                final_edges.append([i, j])
    
    if len(final_edges) > len(best_graph):
        return final_edges
    
    return best_graph
