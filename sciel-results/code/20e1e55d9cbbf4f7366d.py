import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 → C(11,4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict graph: edge if M[i,j] > 2 (incompatible)
    # We want maximum independent set: no two selected have M[i,j] > 2
    
    # Build adjacency as boolean matrix (True = conflict/incompatible)
    conflict = (M > 2).astype(np.int32)
    np.fill_diagonal(conflict, 0)
    
    # Get adjacency lists
    adj = [np.where(conflict[i] == 1)[0] for i in range(N)]
    
    # Greedy initialization with multiple orderings
    rng = np.random.default_rng(12345)
    
    compat = (M <= 2).astype(np.int32)
    compat_count = compat.sum(axis=1)
    
    orderings = [np.argsort(-compat_count), np.arange(N)]
    for _ in range(30):
        orderings.append(rng.permutation(N))
    
    best_set = []
    best_size = 0
    
    for oi, order in enumerate(orderings):
        if time.time() > deadline - 10:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Tomita-style branch and bound for Maximum Independent Set
    # Using the conflict graph, we want to find a large independent set
    # which is equivalent to finding a large clique in the complement graph
    
    # Complement adjacency: no conflict means compatible
    # For Tomita, we work with the graph where edges = compatible (no conflict)
    # Actually, let's work directly: we want max set with no conflicts
    # This is MIS on the conflict graph
    
    # Tomita's BB-MCS algorithm
    # We'll implement a simplified version with coloring bound
    
    def tomita_mis(adj_list, initial_set=None, time_limit=10.0):
        """Find maximum independent set using Tomita's algorithm.
        adj_list[i] = set of neighbors (conflicts) of i in the conflict graph.
        We want to find largest set with no internal edges (independent set).
        """
        start_t = time.time()
        n = len(adj_list)
        
        # Convert to sets for fast lookup
        adj_sets = [set(adj_list[i]) for i in range(n)]
        
        best_size = 0
        best_solution = []
        
        # Greedy coloring for upper bound
        def color_sort(P):
            """Greedy coloring of subgraph induced by P, returns (coloring, order).
            Returns list where coloring[i] is the color of vertex P[i],
            and order is the permutation that sorts by color.
            """
            if not P:
                return [], []
            
            # Greedy sequential coloring
            color = {}
            order = list(P)
            
            # Sort by degree in P for better coloring
            P_set = set(P)
            deg_in_P = {v: len(adj_sets[v] & P_set) for v in P}
            order.sort(key=lambda v: -deg_in_P[v])
            
            used_colors = {}
            for v in order:
                # Find smallest color not used by neighbors in P
                neighbor_colors = set()
                for u in adj_sets[v]:
                    if u in P_set and u in color:
                        neighbor_colors.add(color[u])
                
                c = 1
                while c in neighbor_colors:
                    c += 1
                color[v] = c
                used_colors[c] = used_colors.get(c, 0) + 1
            
            # Return coloring and the order sorted by color
            order_by_color = sorted(P, key=lambda v: color[v])
            return color, order_by_color
        
        # Recursive branch and bound
        def bb(P, Q, depth):
            nonlocal best_size, best_solution
            
            if time.time() - start_t > time_limit:
                return
            
            if not P:
                if len(Q) > best_size:
                    best_size = len(Q)
                    best_solution = Q[:]
                return
            
            # Upper bound: |Q| + max color in coloring of P
            color, order_by_color = color_sort(P)
            if not color:
                if len(Q) > best_size:
                    best_size = len(Q)
                    best_solution = Q[:]
                return
            
            max_color = max(color.values()) if color else 0
            if len(Q) + max_color <= best_size:
                return
            
            # Process vertices in reverse color order
            for v in reversed(order_by_color):
                if v not in P:
                    continue
                if time.time() - start_t > time_limit:
                    return
                
                # Branch: include v
                Q.append(v)
                
                # New P: vertices in P that are compatible with v (not in adj_sets[v])
                P_new = [u for u in P if u != v and u not in adj_sets[v]]
                
                bb(P_new, Q, depth + 1)
                
                Q.pop()
                P.remove(v)
                
                # If best_size >= len(Q) + max possible, prune
                if len(Q) + max_color <= best_size:
                    return
        
        # Start with empty Q and all vertices in P
        all_vertices = list(range(n))
        bb(all_vertices, [], 0)
        
        return best_solution, best_size
    
    # Try Tomita MIS on the full graph if we have time
    if time.time() < deadline - 15:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 5:
            # Use a subset for Tomita if N is too large
            # For N=5302, Tomita might be slow. Let's try on a subset first.
            # Actually, let's try on the full graph with a time limit
            try:
                mis_solution, mis_size = tomita_mis(adj, time_limit=min(remaining_time, 30.0))
                if mis_size > best_size:
                    best_size = mis_size
                    best_set = mis_solution
            except Exception:
                pass
    
    # Local search improvement: try to add more vectors
    if time.time() < deadline - 5 and best_set:
        best_set_arr = np.array(best_set)
        in_set = np.zeros(N, dtype=bool)
        in_set[best_set_arr] = True
        
        # Try adding vectors that are compatible with all current selected
        # This is already done by greedy, but try different orderings for expansion
        unselected = np.where(~in_set)[0]
        
        # Sort unselected by compatibility with current set
        if len(unselected) > 0:
            # For each unselected, count how many selected it's compatible with
            compat_with_selected = M[np.ix_(unselected, best_set_arr)]
            # A vector can be added if it's compatible with ALL selected
            # i.e., all values <= 2
            can_add = np.all(compat_with_selected <= 2, axis=1)
            addable = unselected[can_add]
            
            # Add as many as possible
            for idx in addable:
                if time.time() > deadline - 3:
                    break
                dots = M[idx, in_set]
                if np.all(dots <= 2):
                    best_set.append(idx)
                    in_set[idx] = True
                    best_size = len(best_set)
    
    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = candidates[best_set].astype(np.int64)
    
    # Verify validity
    n = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    return result
