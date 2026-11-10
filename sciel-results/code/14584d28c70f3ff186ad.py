import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n = 20

    def popcount(x):
        return bin(x).count('1')

    def count_common_neighbors(adj, u, v):
        # adj[u] is a bitmask of neighbors of u
        common = adj[u] & adj[v]
        return popcount(common)

    def is_k23_free(adj):
        # Check all pairs: no two vertices share 3+ common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True

    def edge_count(adj):
        # adj is list of bitmasks; count edges as sum of popcounts / 2
        total = sum(popcount(adj[u]) for u in range(n))
        return total // 2

    def can_add_edge(adj, u, v):
        # Check if adding edge (u,v) creates a K2,3
        # After adding, u and v become neighbors.
        # We need to check: for any pair (a,b), do they now have 3+ common neighbors?
        # The only pairs whose common-neighbor count changes are those involving u or v
        # because only adj[u] and adj[v] change.
        # Actually, adding edge (u,v) changes adj[u] |= (1<<v) and adj[v] |= (1<<u).
        # Pairs affected: (u, x) for all x, and (v, x) for all x, and (u,v) itself.
        # For pair (u,v): common neighbors = adj[u] & adj[v] after update.
        #   Before: adj[u] and adj[v] don't include each other.
        #   After: adj[u] has v, adj[v] has u. Common = (old_adj_u | (1<<v)) & (old_adj_v | (1<<u))
        #   = (old_adj_u & old_adj_v) | (old_adj_u & (1<<u)) | ((1<<v) & old_adj_v) | ((1<<v)&(1<<u))
        #   Since u is not in old_adj_u (simple graph, no self-loops), old_adj_u & (1<<u) = 0
        #   Similarly (1<<v) & old_adj_v = 0 if v not already neighbor of v (no self-loop)
        #   So common = old_adj_u & old_adj_v, same as before! Wait, that's not right.
        #   Actually after adding: new_adj_u = old_adj_u | (1<<v), new_adj_v = old_adj_v | (1<<u)
        #   common = new_adj_u & new_adj_v = (old_adj_u | (1<<v)) & (old_adj_v | (1<<u))
        #   = (old_adj_u & old_adj_v) | (old_adj_u & (1<<u)) | ((1<<v) & old_adj_v) | ((1<<v) & (1<<u))
        #   old_adj_u & (1<<u) = 0 (no self-loop), (1<<v) & old_adj_v: is v a neighbor of v? No. So 0.
        #   (1<<v) & (1<<u) = 0 since u != v.
        #   So common = old_adj_u & old_adj_v. Same as before!
        #   So adding edge (u,v) does NOT change the common neighbor count for pair (u,v).
        #
        # For pair (u, x) where x != v: 
        #   new common = new_adj_u & new_adj_x = (old_adj_u | (1<<v)) & old_adj_x
        #   = (old_adj_u & old_adj_x) | ((1<<v) & old_adj_x)
        #   So it increases by 1 iff v is a neighbor of x (i.e., x in old_adj_v).
        #   So if old count for (u,x) was 2, and v is neighbor of x, then new count = 3 -> violation.
        #
        # For pair (v, x) where x != u:
        #   Similarly, increases by 1 iff u is neighbor of x (x in old_adj_u).
        #
        # So can_add_edge check:
        #   1. For each x in neighbors of v (x != u): if count_common(old_adj, u, x) >= 2, violation.
        #   2. For each x in neighbors of u (x != v): if count_common(old_adj, v, x) >= 2, violation.
        
        # Check pairs (u, x) for x in adj[v] (neighbors of v), x != u
        for x in range(n):
            if x == u:
                continue
            if adj[v] & (1 << x):  # x is neighbor of v
                c = count_common_neighbors(adj, u, x)
                if c >= 2:
                    return False
        # Check pairs (v, x) for x in adj[u] (neighbors of u), x != v
        for x in range(n):
            if x == v:
                continue
            if adj[u] & (1 << x):  # x is neighbor of u
                c = count_common_neighbors(adj, v, x)
                if c >= 2:
                    return False
        return True

    def make_seed():
        """Return adj bitmask list for 5 disjoint K4s."""
        adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def random_dense_start():
        """Create a random graph and greedily remove K2,3 violations."""
        adj = [0] * n
        # Start with a moderately dense random graph
        p = 0.6
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Remove violations
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if count_common_neighbors(adj, u, v) >= 3:
                        # Remove a random common neighbor edge to fix
                        common = adj[u] & adj[v]
                        # Pick a random vertex w from common neighbors
                        w_list = [w for w in range(n) if common & (1 << w)]
                        if w_list:
                            w = random.choice(w_list)
                            # Remove edge (u, w)
                            adj[u] &= ~(1 << w)
                            adj[w] &= ~(1 << u)
                            changed = True
        return adj

    def local_search(adj, max_time):
        """Improve adj via edge additions and swaps. Returns improved adj."""
        improved = True
        while improved and (time.time() < max_time):
            improved = False
            # Try adding random missing edges
            for _ in range(200):
                if time.time() >= max_time:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            # Try swaps: remove one edge, try to add two
            for _ in range(100):
                if time.time() >= max_time:
                    break
                # Pick a random existing edge to remove
                u = random.randint(0, n - 1)
                # Pick a random neighbor of u
                neighbors = [x for x in range(n) if adj[u] & (1 << x)]
                if not neighbors:
                    continue
                v = random.choice(neighbors)
                # Remove edge (u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Try to add two new edges
                added = 0
                for _ in range(50):
                    a = random.randint(0, n - 1)
                    b = random.randint(0, n - 1)
                    if a == b:
                        continue
                    if adj[a] & (1 << b):
                        continue
                    if can_add_edge(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        added += 1
                        if added == 2:
                            break
                if added < 1:
                    # Revert: put back (u,v) if possible
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    improved = True

        return adj

    # Start with seed
    best_adj = make_seed()
    best_count = edge_count(best_adj)

    # Try multiple random starts and local search
    random.seed(42)
    num_trials = 50
    for trial in range(num_trials):
        if time.time() >= deadline:
            break
        # Alternate between seed and random starts
        if trial % 2 == 0:
            adj = make_seed()
        else:
            adj = random_dense_start()
        
        # Local search with time budget
        t_budget = min(deadline, time.time() + 10.0)
        adj = local_search(adj, t_budget)
        
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [x for x in adj]
            # Try further improvement from this best
            if time.time() < deadline - 5:
                adj2 = [x for x in best_adj]
                t_budget2 = min(deadline, time.time() + 10.0)
                adj2 = local_search(adj2, t_budget2)
                cnt2 = edge_count(adj2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best_adj = [x for x in adj2]

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])

    return edges
