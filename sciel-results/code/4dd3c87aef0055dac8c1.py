import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    DEADLINE = start + 135.0
    N = 16
    FULL = (1 << N) - 1

    def time_ok():
        return time.time() < DEADLINE

    def is_3_colorable_fast(adj, edge_count):
        """
        Optimized 3-colorability check using bitmask backtracking with memoization.
        """
        if edge_count <= 2 * N - 3:
            return True

        adj_mask = list(adj)
        deg = [bin(adj_mask[i]).count('1') for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: -deg[i])

        pos_in_order = [0] * N
        for i, v in enumerate(order):
            pos_in_order[v] = i

        ordered_adj = [0] * N
        for i, v in enumerate(order):
            m = 0
            for u in range(N):
                if (adj_mask[v] >> u) & 1:
                    m |= (1 << pos_in_order[u])
            ordered_adj[i] = m

        memo = {}

        def dfs(i, colors):
            if i == N:
                return True

            key = (i, colors[0], colors[1], colors[2])
            if key in memo:
                return memo[key]

            v = order[i]
            mask = ordered_adj[i]

            for c in range(3):
                if (colors[c] & mask) == 0:
                    newc = colors[c] | (1 << i)
                    if c == 0:
                        nc = (newc, colors[1], colors[2])
                    elif c == 1:
                        nc = (colors[0], newc, colors[2])
                    else:
                        nc = (colors[0], colors[1], newc)
                    if dfs(i + 1, nc):
                        memo[key] = True
                        return True

            memo[key] = False
            return False

        result = dfs(0, (0, 0, 0))
        return result

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count_of(adj):
        cnt = 0
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    cnt += 1
        return cnt

    def edges_from_adj(adj):
        return [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]

    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    best_edges = None
    best_count = -1

    # Start with the Clebsch graph: 16 vertices, 40 edges, triangle-free, not 3-colorable
    clebsch_vertices = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    clebsch_adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            xor_val = clebsch_vertices[i] ^ clebsch_vertices[j]
            if bin(xor_val).count('1') == 4:
                add_edge(clebsch_adj, i, j)

    if not is_3_colorable_fast(clebsch_adj, 40):
        best_count = 40
        best_edges = edges_from_adj(clebsch_adj)

    def local_search_strict(adj, max_iter=200):
        """Local search that only accepts edges preserving non-3-colorability."""
        nonlocal best_count, best_edges
        ec = edge_count_of(adj)
        for _ in range(max_iter):
            if not time_ok():
                break
            improved = False
            for u in range(N):
                for v in range(u + 1, N):
                    if not ((adj[u] >> v) & 1):
                        if is_triangle_free_add(adj, u, v):
                            add_edge(adj, u, v)
                            if not is_3_colorable_fast(adj, ec + 1):
                                ec += 1
                                improved = True
                                if ec > best_count:
                                    best_count = ec
                                    best_edges = edges_from_adj(adj)
                            else:
                                remove_edge(adj, u, v)
            if not improved:
                break
        return adj

    def simulated_annealing(adj, max_steps=5000):
        """
        Simulated annealing that can temporarily accept 3-colorable graphs.
        Objective: maximize (edge_count, -is_3_colorable) with soft constraint.
        We use a temperature schedule to escape local optima.
        """
        nonlocal best_count, best_edges
        
        ec = edge_count_of(adj)
        is_nc = not is_3_colorable_fast(adj, ec)
        
        # State: adj, ec, is_nc
        cur_adj = adj[:]
        cur_ec = ec
        cur_is_nc = is_nc
        
        # Track best valid (non-3-colorable) solution
        sa_best_adj = cur_adj[:]
        sa_best_ec = cur_ec
        sa_best_is_nc = cur_is_nc
        
        T0 = 1.0
        T_end = 0.01
        cooling = (T_end / T0) ** (1.0 / max_steps)
        
        for step in range(max_steps):
            if not time_ok():
                break
            
            T = T0 * (cooling ** step)
            
            # Generate a random move: add or remove a random non-edge/edge
            # that preserves triangle-freeness
            all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            rng_move = random.Random(step * 7919 + 12345)
            rng_move.shuffle(all_pairs)
            
            move_applied = False
            for u, v in all_pairs:
                has_edge = (cur_adj[u] >> v) & 1
                
                if has_edge:
                    # Try removing edge
                    remove_edge(cur_adj, u, v)
                    new_ec = cur_ec - 1
                    new_is_nc = not is_3_colorable_fast(cur_adj, new_ec)
                    
                    # Evaluate move
                    # Reward: want high edge count and non-3-colorable
                    # Score = edge_count * 10 + (5 if non-3-colorable else 0)
                    old_score = cur_ec * 10 + (5 if cur_is_nc else 0)
                    new_score = new_ec * 10 + (5 if new_is_nc else 0)
                    delta = new_score - old_score
                    
                    if delta >= 0 or random.random() < (2.718281828 ** (delta / max(T, 1e-10))):
                        # Accept
                        cur_ec = new_ec
                        cur_is_nc = new_is_nc
                        move_applied = True
                        
                        if new_is_nc and new_ec > sa_best_ec:
                            sa_best_ec = new_ec
                            sa_best_adj = cur_adj[:]
                            if new_ec > best_count:
                                best_count = new_ec
                                best_edges = edges_from_adj(cur_adj)
                    else:
                        # Reject: undo
                        add_edge(cur_adj, u, v)
                    
                    break
                else:
                    # Try adding edge (must preserve triangle-freeness)
                    if not is_triangle_free_add(cur_adj, u, v):
                        continue
                    
                    add_edge(cur_adj, u, v)
                    new_ec = cur_ec + 1
                    new_is_nc = not is_3_colorable_fast(cur_adj, new_ec)
                    
                    old_score = cur_ec * 10 + (5 if cur_is_nc else 0)
                    new_score = new_ec * 10 + (5 if new_is_nc else 0)
                    delta = new_score - old_score
                    
                    if delta >= 0 or random.random() < (2.718281828 ** (delta / max(T, 1e-10))):
                        # Accept
                        cur_ec = new_ec
                        cur_is_nc = new_is_nc
                        move_applied = True
                        
                        if new_is_nc and new_ec > sa_best_ec:
                            sa_best_ec = new_ec
                            sa_best_adj = cur_adj[:]
                            if new_ec > best_count:
                                best_count = new_ec
                                best_edges = edges_from_adj(cur_adj)
                    else:
                        # Reject: undo
                        remove_edge(cur_adj, u, v)
                    
                    break
            
            if not move_applied:
                # No valid move found; try a random restart within SA
                if random.random() < 0.1:
                    # Perturb: randomly add/remove a few edges
                    for _ in range(3):
                        pu, pv = random.choice(all_pairs)
                        if (cur_adj[pu] >> pv) & 1:
                            remove_edge(cur_adj, pu, pv)
                            cur_ec -= 1
                        elif is_triangle_free_add(cur_adj, pu, pv):
                            add_edge(cur_adj, pu, pv)
                            cur_ec += 1
                    cur_is_nc = not is_3_colorable_fast(cur_adj, cur_ec)
        
        # After SA, try strict local search from the best SA state
        if sa_best_is_nc and sa_best_ec > 0:
            final_adj = local_search_strict(sa_best_adj, max_iter=100)
            final_ec = edge_count_of(final_adj)
            if final_ec > best_count:
                best_count = final_ec
                best_edges = edges_from_adj(final_adj)
        
        return cur_adj

    # Multi-start with simulated annealing
    rng = random.Random(42)
    attempts = 0

    while time_ok() and attempts < 3000:
        attempts += 1

        if attempts % 5 == 1:
            # Start from Clebsch graph with perturbation
            adj = clebsch_adj[:]
            # Randomly remove some edges to create diversity
            for _ in range(rng.randint(0, 8)):
                edges_list = edges_from_adj(adj)
                if edges_list:
                    u, v = rng.choice(edges_list)
                    remove_edge(adj, u, v)
            
            # Run simulated annealing
            adj = simulated_annealing(adj, max_steps=3000)
            
            # Follow up with strict local search
            ec = edge_count_of(adj)
            if not is_3_colorable_fast(adj, ec):
                adj = local_search_strict(adj, max_iter=150)
                ec = edge_count_of(adj)
                if ec > best_count:
                    best_count = ec
                    best_edges = edges_from_adj(adj)
        
        elif attempts % 5 == 2:
            # Random greedy construction then SA
            adj = [0] * N
            pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            rng.shuffle(pairs)
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if is_triangle_free_add(adj, u, v):
                        add_edge(adj, u, v)
            
            ec = edge_count_of(adj)
            if not is_3_colorable_fast(adj, ec):
                adj = simulated_annealing(adj, max_steps=2000)
                ec = edge_count_of(adj)
                if not is_3_colorable_fast(adj, ec):
                    adj = local_search_strict(adj, max_iter=100)
                    ec = edge_count_of(adj)
                    if ec > best_count:
                        best_count = ec
                        best_edges = edges_from_adj(adj)
        
        elif attempts % 5 == 3:
            # Start from current best with perturbation
            if best_edges is not None:
                adj = [0] * N
                for u, v in best_edges:
                    add_edge(adj, u, v)
                
                # Perturb
                for _ in range(rng.randint(1, 5)):
                    edges_list = edges_from_adj(adj)
                    if edges_list:
                        u, v = rng.choice(edges_list)
                        remove_edge(adj, u, v)
                
                adj = simulated_annealing(adj, max_steps=2500)
                ec = edge_count_of(adj)
                if not is_3_colorable_fast(adj, ec):
                    adj = local_search_strict(adj, max_iter=150)
                    ec = edge_count_of(adj)
                    if ec > best_count:
                        best_count = ec
                        best_edges = edges_from_adj(adj)
        
        elif attempts % 5 == 4:
            # Pure local search from Clebsch
            adj = clebsch_adj[:]
            adj = local_search_strict(adj, max_iter=200)
            ec = edge_count_of(adj)
            if ec > best_count:
                best_count = ec
                best_edges = edges_from_adj(adj)
        
        else:
            # Random greedy + local search
            adj = [0] * N
            pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            rng.shuffle(pairs)
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if is_triangle_free_add(adj, u, v):
                        add_edge(adj, u, v)
            
            ec = edge_count_of(adj)
            if not is_3_colorable_fast(adj, ec):
                adj = local_search_strict(adj, max_iter=200)
                ec = edge_count_of(adj)
                if ec > best_count:
                    best_count = ec
                    best_edges = edges_from_adj(adj)

    if best_edges is not None:
        return best_edges

    # Fallback: return Clebsch graph (40 edges)
    return edges_from_adj(clebsch_adj)
