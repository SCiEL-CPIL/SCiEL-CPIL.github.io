import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18
    full_mask = (1 << n) - 1

    # Precompute edge list for all pairs
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_independent(mask, adj):
        """Check if the set of vertices in mask forms an independent set."""
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        """Check if adding edge (u, v) keeps the graph K4-free."""
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        # For the graph to remain K4-free, the common neighbors of u and v
        # must form an independent set (no edges among them), otherwise
        # any edge (x,y) among common neighbors would form a K4 with u,v.
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        """
        Exact Branch and Bound for the size of the largest triangle-free induced subgraph.
        A set S is triangle-free if and only if for every v in S, N(v) intersect S is an independent set.
        Alternatively, we can view it as: S is triangle-free iff S contains no triangle.
        The B&B below builds S incrementally. When adding v to S, we must ensure that
        N(v) ∩ S is an independent set (so no new triangles are formed).
        """
        best = [0]
        
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
            # Upper bound: current size + remaining candidates
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C to branch on (heuristic for better pruning)
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
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v
            # Check if S U {v} is triangle-free
            # This requires N(v) & S to be an independent set
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

    # --- Initialization ---
    # Start with a random maximal K4-free graph
    random.seed(42)
    best_adj = [0] * n
    best_score = n  # Max possible is n
    
    # Generate a random maximal K4-free graph
    adj = [0] * n
    random.shuffle(all_edges)
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Evaluate initial graph
    t_start = time.time()
    score = largest_tf(adj, time_limit=deadline - 2.0)
    t_end = time.time()
    
    if score < best_score:
        best_score = score
        best_adj = adj[:]
    
    # --- Simulated Annealing ---
    # We want to minimize the largest triangle-free induced subgraph.
    # Moves:
    # 1. Add an edge (if it keeps K4-free)
    # 2. Remove an edge
    # 3. Swap an edge (remove one, add another)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Local search / Simulated Annealing step
        # Try a few random moves
        moves_made = 0
        while moves_made < 20 and time.time() < deadline - 5.0:
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Find a non-edge that can be added
                candidates = []
                for u, v in all_edges:
                    if not ((adj[u] >> v) & 1):
                        if can_add_edge(current_adj, u, v):
                            candidates.append((u, v))
                
                if not candidates:
                    moves_made += 1
                    continue
                
                u, v = random.choice(candidates)
                
                # Tentatively add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                # Evaluate
                new_score = largest_tf(current_adj, time_limit=deadline - 2.0)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_score - current_score
                    if delta > 0:
                        if random.random() < pow(2.718281828, -delta / T):
                            current_score = new_score
                        else:
                            # Revert
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                    else:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = current_adj[:]
                
                moves_made += 1
                
            elif move_type == 'remove':
                # Find an edge to remove
                edges_in_graph = []
                for u, v in all_edges:
                    if (current_adj[u] >> v) & 1:
                        edges_in_graph.append((u, v))
                
                if not edges_in_graph:
                    moves_made += 1
                    continue
                
                u, v = random.choice(edges_in_graph)
                
                # Tentatively remove
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Evaluate
                new_score = largest_tf(current_adj, time_limit=deadline - 2.0)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                else:
                    delta = new_score - current_score
                    if delta > 0:
                        if random.random() < pow(2.718281828, -delta / T):
                            current_score = new_score
                        else:
                            # Revert
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
                    else:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = current_adj[:]
                
                moves_made += 1
                
            elif move_type == 'swap':
                # Remove one edge, add another
                edges_in_graph = []
                for u, v in all_edges:
                    if (current_adj[u] >> v) & 1:
                        edges_in_graph.append((u, v))
                
                if not edges_in_graph:
                    moves_made += 1
                    continue
                
                u1, v1 = random.choice(edges_in_graph)
                
                # Find a non-edge that can be added
                candidates = []
                for u, v in all_edges:
                    if not ((current_adj[u] >> v) & 1) and (u, v) != (u1, v1) and (u, v) != (v1, u1):
                        if can_add_edge(current_adj, u, v):
                            candidates.append((u, v))
                
                if not candidates:
                    moves_made += 1
                    continue
                
                u2, v2 = random.choice(candidates)
                
                # Tentatively remove (u1, v1) and add (u2, v2)
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                # Evaluate
                new_score = largest_tf(current_adj, time_limit=deadline - 2.0)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                else:
                    delta = new_score - current_score
                    if delta > 0:
                        if random.random() < pow(2.718281828, -delta / T):
                            current_score = new_score
                        else:
                            # Revert
                            current_adj[u1] |= (1 << v1)
                            current_adj[v1] |= (1 << u1)
                            current_adj[u2] &= ~(1 << v2)
                            current_adj[v2] &= ~(1 << u2)
                    else:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = current_adj[:]
                
                moves_made += 1
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    # Final polish: try to add any remaining edges that don't worsen the score
    # (This is a greedy pass to ensure we're in a local optimum for edge addition)
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for u, v in all_edges:
            if time.time() > deadline - 1.0:
                break
            if not ((best_adj[u] >> v) & 1):
                if can_add_edge(best_adj, u, v):
                    # Tentatively add
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    
                    new_score = largest_tf(best_adj, time_limit=deadline - 0.5)
                    
                    if new_score <= best_score:
                        best_score = new_score
                        improved = True
                    else:
                        # Revert
                        best_adj[u] &= ~(1 << v)
                        best_adj[v] &= ~(1 << u)

    edges = get_edge_list(best_adj)
    return edges
