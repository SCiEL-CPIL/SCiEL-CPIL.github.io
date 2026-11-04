import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Degree of each node in the conflict graph
    degrees = np.sum(conflict, axis=1)

    # Compatibility graph: compatible if intersection size <= 2 (i.e., NOT conflict)
    # We want a maximum set of mutually compatible supports
    # This is equivalent to finding a maximum clique in the compatibility graph
    # or a maximum independent set in the conflict graph

    # For the branch-and-bound maximum clique approach:
    # Build the compatibility adjacency matrix
    compat = ~conflict
    np.fill_diagonal(compat, False)

    # Tomita's algorithm for maximum clique with pivot-based coloring
    # We work on the compatibility graph

    def max_clique_tomita(adj, n, time_limit=None, start_time=None):
        """
        Find maximum clique in graph using Tomita's algorithm with coloring.
        adj: n x n boolean adjacency matrix
        Returns: list of nodes in maximum clique
        """
        if start_time is None:
            start_time = time.time()
        
        best_clique = []
        best_size = 0
        
        # Convert to list of neighbors for each node
        neighbors = [np.where(adj[i])[0] for i in range(n)]
        
        # State for backtracking
        def expand(R, P):
            nonlocal best_clique, best_size
            
            if time_limit is not None and time.time() > time_limit:
                return
            
            R = list(R)
            P = list(P)
            
            if len(P) == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R[:]
                return
            
            # Coloring: greedily color nodes in P to get upper bound
            # Use a simple greedy coloring
            uncolored = set(P)
            color_class = []
            color = 0
            colored = []
            
            while uncolored:
                color += 1
                # Find maximal independent set in uncolored subgraph
                available = set(uncolored)
                class_nodes = []
                while available:
                    # Pick node with highest degree in available
                    node = max(available, key=lambda x: len([y for y in available if adj[x, y]]))
                    class_nodes.append(node)
                    available.discard(node)
                    # Remove neighbors of node from available
                    available = {x for x in available if not adj[node, x]}
                
                uncolored -= set(class_nodes)
                colored.append((color, class_nodes))
            
            # Process in reverse color order
            for color_idx in range(len(colored) - 1, -1, -1):
                c, nodes = colored[color_idx]
                if len(R) + c <= best_size:
                    break
                
                for v in nodes:
                    if v not in P:
                        continue
                    # Check if v is still in P
                    if v in P:
                        new_R = R + [v]
                        new_P = [x for x in P if adj[v, x] and x != v]
                        # Only consider nodes with color >= current color
                        # Actually, we should filter to maintain the bound
                        expand(new_R, new_P)
                        P.remove(v)
        
        # Initial call with all nodes
        expand([], list(range(n)))
        
        return best_clique
    
    # Use branch-and-bound with Tomita coloring
    # But first, let's try a faster approach: greedy + local search with better heuristics
    
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization with different orderings
    def greedy_init(order):
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, np.sum(x)

    # Local search: try to improve by swapping
    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / max(num_iterations, 1))
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Compute blocked nodes
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = np.where(x)[0]
            
            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break
                
            move_type = py_rng.random()
            if move_type < 0.5 and len(add_candidates) > 0:
                # Add a random non-conflicting node
                v = py_rng.choice(add_candidates)
                if not x[v]:
                    x[v] = True
                    new_size = current_size + 1
                    if new_size > best_size_local:
                        best_size_local = new_size
                        best_x = x.copy()
                    current_size = new_size
            elif move_type < 0.7 and len(remove_candidates) > 0:
                # Remove a random selected node
                v = py_rng.choice(remove_candidates)
                if x[v]:
                    x[v] = False
                    new_size = current_size - 1
                    if py_rng.random() < np.exp(-1.0 / T):
                        current_size = new_size
                    else:
                        x[v] = True
            else:
                # Swap: remove one, add one
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    v_rem = py_rng.choice(remove_candidates)
                    if v_add != v_rem and not x[v_add] and x[v_rem]:
                        temp_x = x.copy()
                        temp_x[v_rem] = False
                        if not np.any(conflict[v_add, temp_x]):
                            x[v_rem] = False
                            x[v_add] = True
            
            T *= cooling

        return best_x, best_size_local

    # Multiple strategies for initialization
    strategies = []
    
    # Strategy 1: degree-ordered greedy
    order_deg = np.argsort(degrees)
    strategies.append(order_deg)
    
    # Strategy 2: reverse degree-ordered
    strategies.append(order_deg[::-1])
    
    # Strategy 3-10: random orderings
    for _ in range(7):
        order = np.arange(N_S)
        rng.shuffle(order)
        strategies.append(order)

    num_restarts = 50
    iterations_per_restart = 3000
    T0 = 1.0
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Use different initialization strategies
        if restart < len(strategies):
            order = strategies[restart]
        else:
            order = np.arange(N_S)
            rng.shuffle(order)
        
        x, size = greedy_init(order)
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Also try branch-and-bound for maximum clique on a subset of high-potential nodes
    # Focus on nodes with low degree (more compatible)
    if time.time() < deadline - 30:
        # Get top 150 lowest-degree nodes
        low_degree_nodes = np.argsort(degrees)[:150]
        sub_adj = compat[np.ix_(low_degree_nodes, low_degree_nodes)]
        
        # Run a simplified branch-and-bound
        def max_clique_bnb(adj, n, time_limit):
            best_clique = []
            best_size = 0
            
            def expand(R, P):
                nonlocal best_clique, best_size
                
                if time.time() > time_limit:
                    return
                
                R = list(R)
                P = list(P)
                
                if len(P) == 0:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = R[:]
                    return
                
                if len(R) + len(P) <= best_size:
                    return
                
                # Greedy coloring for upper bound
                uncolored = set(P)
                color = 0
                while uncolored:
                    color += 1
                    available = set(uncolored)
                    while available:
                        node = max(available, key=lambda x: len([y for y in available if adj[x, y]]))
                        available.discard(node)
                        available = {x for x in available if not adj[node, x]}
                    uncolored -= available if False else set()  # This is wrong, let me fix
                    
                    # Actually, let's use a simpler bound
                    break
                
                # Simple bound: len(R) + len(P)
                if len(R) + len(P) <= best_size:
                    return
                
                # Pick a pivot node
                pivot = max(P, key=lambda x: len([y for y in P if adj[x, y]]))
                
                # Branch on neighbors of pivot in P
                candidates = [x for x in P if adj[pivot, x]]
                
                if not candidates:
                    # Try non-neighbors
                    candidates = [x for x in P if not adj[pivot, x]]
                
                for v in candidates:
                    new_R = R + [v]
                    new_P = [x for x in P if adj[v, x] and x != v]
                    if len(new_R) + len(new_P) > best_size:
                        expand(new_R, new_P)
                    P.remove(v)
                    if len(R) + len(P) <= best_size:
                        break
            
            expand([], list(range(n)))
            return best_clique
        
        bnb_time_limit = deadline - 10
        clique_nodes = max_clique_bnb(sub_adj, 150, bnb_time_limit)
        
        if len(clique_nodes) > best_size:
            # Map back to original indices
            best_support_set = set(low_degree_nodes[clique_nodes].tolist())
            best_size = len(clique_nodes)

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
