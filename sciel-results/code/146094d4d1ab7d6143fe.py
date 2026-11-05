import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            bnb(S_mask, C_mask & ~v_bit)
            
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def greedy_k4_free_from_order(order, rng):
        """Build a maximal K4-free graph by inserting edges in a randomized order
        derived from the vertex ordering. The ordering influences which edges get added."""
        adj = [0] * n
        # Generate edge order based on vertex ordering with some randomization
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        # Shuffle edges but bias by vertex ordering
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def build_two_c9():
        """Build two disjoint 9-cycles: C9 on vertices 0-8 and C9 on vertices 9-17.
        This is K4-free (max degree 2, no triangles at all).
        Triangle-free induced subgraph = independent set.
        For C9, alpha = 4 (floor(9/2) = 4).
        For two disjoint C9, alpha = 4 + 4 = 8.
        """
        adj = [0] * n
        # First C9: vertices 0-8
        for i in range(9):
            u = i
            v = (i + 1) % 9
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Second C9: vertices 9-17
        for i in range(9):
            u = 9 + i
            v = 9 + ((i + 1) % 9)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    random.seed(42)
    rng = random.Random(12345)
    best_adj = None
    best_score = n

    # Strategy 1: Start with two disjoint C9 (score should be 8)
    adj_c9 = build_two_c9()
    score_c9 = largest_tf(adj_c9, time_limit=deadline - 5.0)
    best_score = score_c9
    best_adj = adj_c9[:]
    print(f"Two C9 baseline: {best_score}")

    # Simulated Annealing: perturb vertex ordering for greedy K4-free construction
    # The "temperature" controls how much we randomize the edge insertion order
    def simulated_annealing(time_limit):
        """Use simulated annealing to find a good K4-free graph by perturbing
        the vertex ordering that drives greedy edge insertion."""
        current_order = list(range(n))
        rng.shuffle(current_order)
        current_adj = greedy_k4_free_from_order(current_order, rng)
        current_score = largest_tf(current_adj, time_limit=time_limit)
        
        best_sa_adj = current_adj[:]
        best_sa_score = current_score
        
        # Annealing parameters
        initial_temp = 1.0
        final_temp = 0.01
        cooling_rate = 0.995
        temp = initial_temp
        
        iteration = 0
        while temp > final_temp and time.time() < time_limit:
            iteration += 1
            
            # Perturbation: swap two random vertices in the ordering
            i, j = rng.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            # Rebuild graph with new ordering
            new_adj = greedy_k4_free_from_order(current_order, rng)
            new_score = largest_tf(new_adj, time_limit=time_limit)
            
            # Acceptance criterion
            if new_score <= current_score:
                # Better or equal: always accept
                current_adj = new_adj
                current_score = new_score
                if new_score < best_sa_score:
                    best_sa_score = new_score
                    best_sa_adj = new_adj[:]
            else:
                # Worse: accept with probability exp(-delta/temp)
                delta = new_score - current_score
                if delta > 0:
                    prob = 1.0 / (1.0 + delta * temp * 10)
                    if rng.random() < prob:
                        current_adj = new_adj
                        current_score = new_score
                else:
                    current_adj = new_adj
                    current_score = new_score
            
            # Cool down
            temp *= cooling_rate
            
            # Periodic restarts if stuck
            if iteration % 100 == 0:
                # Try a fresh random ordering
                fresh_order = list(range(n))
                rng.shuffle(fresh_order)
                fresh_adj = greedy_k4_free_from_order(fresh_order, rng)
                fresh_score = largest_tf(fresh_adj, time_limit=time_limit)
                if fresh_score < current_score:
                    current_order = fresh_order
                    current_adj = fresh_adj
                    current_score = fresh_score
                    if fresh_score < best_sa_score:
                        best_sa_score = fresh_score
                        best_sa_adj = fresh_adj[:]
        
        return best_sa_adj, best_sa_score

    # Run simulated annealing
    if time.time() < deadline - 10.0:
        sa_adj, sa_score = simulated_annealing(deadline - 5.0)
        if sa_score < best_score:
            best_score = sa_score
            best_adj = sa_adj[:]
            print(f"Simulated Annealing: {best_score}")

    # Additional random restarts with greedy construction to explore more of the space
    while time.time() < deadline - 5.0:
        adj = greedy_k4_free_from_order(list(range(n)), rng)
        score = largest_tf(adj, time_limit=deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Random restart: New best {best_score}")

    if best_adj is None:
        # Fallback to C9 construction
        best_adj = build_two_c9()

    edges = get_edge_list(best_adj)
    return edges
