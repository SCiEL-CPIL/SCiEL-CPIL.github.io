import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Precompute all edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    m_total = len(all_edges)  # 190

    def popcount(x):
        return x.bit_count()

    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Check 1: |N(u) ∩ N(v)| <= 2 (the pair u,v themselves)
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # Check 2: For each w in N(u) (old), after adding v to N(u):
        #   |N(v) ∩ N(w)| + 1 <= 2, i.e., |N(v) ∩ N(w)| <= 1 for all w in N(u)
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
        
        # Check 3: For each w in N(v) (old), after adding u to N(v):
        #   |N(u) ∩ N(w)| + 1 <= 2, i.e., |N(u) ∩ N(w)| <= 1 for all w in N(v)
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == u:
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False
        
        return True

    def edge_count(adj):
        c = 0
        for u in range(n):
            c += popcount(adj[u])
        return c // 2

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Greedy construction with a given edge order
    def greedy_build(order):
        adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    # Local search: try to add any missing edge, then try swaps
    def local_search(adj, edges_count, max_rounds=50):
        adj = list(adj)
        changed = True
        rounds = 0
        while changed and rounds < max_rounds:
            changed = False
            rounds += 1
            # Try adding edges
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        edges_count += 1
                        changed = True
            if time.time() > deadline:
                break
            # Try swaps: remove (u,v), add (x,y)
            for i, (u, v) in enumerate(all_edges):
                if ((adj[u] >> v) & 1):
                    for j in range(i + 1, len(all_edges)):
                        x, y = all_edges[j]
                        if ((adj[x] >> y) & 1):
                            continue
                        # Try swap
                        new_adj = list(adj)
                        remove_edge(new_adj, u, v)
                        if is_valid_add(new_adj, x, y):
                            add_edge(new_adj, x, y)
                            # Same edge count, but may enable further additions
                            # Accept if it leads to improvement
                            adj2 = list(new_adj)
                            # Quick check: can we add more edges?
                            improved = False
                            for a, b in all_edges:
                                if not ((adj2[a] >> b) & 1):
                                    if is_valid_add(adj2, a, b):
                                        improved = True
                                        break
                            if improved:
                                adj = new_adj
                                changed = True
                                break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
        return adj, edges_count

    # Simulated annealing
    def simulated_annealing(adj, edges_count, steps=5000):
        adj = list(adj)
        curr_edges = edges_count
        best_adj = list(adj)
        best_edges = curr_edges
        T = 1.0
        cool = 0.99995
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Add a random edge
                u, v = random.choice(all_edges)
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        curr_edges += 1
                        if curr_edges > best_edges:
                            best_edges = curr_edges
                            best_adj = list(adj)
                    else:
                        T *= 0.95
            elif move_type < 0.85:
                # Remove a random edge
                # Find a random present edge
                present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    curr_edges -= 1
            else:
                # Swap: remove one, add another
                present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
                if present:
                    u, v = random.choice(present)
                    x, y = random.choice(all_edges)
                    if ((adj[x] >> y) & 1) or (x == u and y == v) or (x == v and y == u):
                        T *= 0.99
                        continue
                    new_adj = list(adj)
                    remove_edge(new_adj, u, v)
                    if is_valid_add(new_adj, x, y):
                        add_edge(new_adj, x, y)
                        # Same edge count
                        adj = new_adj
                    else:
                        T *= 0.95
            
            T *= cool
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_edges

    # Run multiple strategies
    best_adj = [0] * n
    best_edges = 0

    # Strategy 1: Greedy with random edge orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(all_edges)
        random.shuffle(order)
        adj, count = greedy_build(order)
        adj, count = local_search(adj, count)
        if count > best_edges:
            best_edges = count
            best_adj = list(adj)

    # Strategy 2: Start from structured initializations
    # 5 disjoint K4s
    adj = [0] * n
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                add_edge(adj, base + a, base + b)
    count = edge_count(adj)
    adj, count = local_search(adj, count)
    if count > best_edges:
        best_edges = count
        best_adj = list(adj)

    # Complete graph minus a perfect matching (10 edges removed from 190)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if (u // 2) != (v // 2) or (u % 2) != (v % 2):
                pass
    # Actually let's just do: all edges except matching
    adj = [0] * n
    for u in range(0, n, 2):
        for v in range(n):
            if v != u and v != u + 1:
                add_edge(adj, u, v)
    count = edge_count(adj)
    # This might not be K2,3-free, so we need to verify and fix
    # Let's use a different structured start
    # Random dense start
    for trial in range(10):
        if time.time() > deadline:
            break
        adj = [0] * n
        for u, v in all_edges:
            if random.random() < 0.4:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
        count = edge_count(adj)
        adj, count = local_search(adj, count)
        if count > best_edges:
            best_edges = count
            best_adj = list(adj)

    # Strategy 3: Simulated annealing from best solution
    for trial in range(5):
        if time.time() > deadline:
            break
        adj2, edges2 = simulated_annealing(best_adj, best_edges, steps=5000)
        if edges2 > best_edges:
            best_edges = edges2
            best_adj = list(adj2)

    # Final local search polish
    best_adj, best_edges = local_search(best_adj, best_edges)

    # Convert to edge list
    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
    return result
