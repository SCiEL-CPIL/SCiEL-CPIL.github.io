import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n = 16
    # Vertices 0..15

    # Build Grötzsch graph (Mycielski of C5) on vertices 0..10
    def build_grotzsch():
        # C5: 0-1-2-3-4-0
        c5_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        edges = list(c5_edges)
        # Mycielski: copies 5..9, top 10
        for u, v in c5_edges:
            edges.append((u, 5 + v))
            edges.append((v, 5 + u))
        for i in range(5):
            edges.append((10, 5 + i))
        return edges

    core_edges = build_grotzsch()
    core_n = 11
    new_vertices = list(range(core_n, n))
    num_new = 5

    # Build core adjacency
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)

    # Enumerate all independent sets in Grötzsch
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind and mask != 0:
            independent_sets.append(mask)

    # Sort by size descending
    independent_sets.sort(key=lambda x: -bin(x).count("1"))

    # Precompute disjointness matrix between independent sets
    num_sets = len(independent_sets)
    disjoint = [[False] * num_sets for _ in range(num_sets)]
    for i in range(num_sets):
        for j in range(i, num_sets):
            d = (independent_sets[i] & independent_sets[j]) == 0
            disjoint[i][j] = d
            disjoint[j][i] = d

    # Precompute size of each independent set
    set_sizes = [bin(s).count("1") for s in independent_sets]

    # Fast 3-colorability check using bitmasks
    def is_3_colorable(adj_full):
        # adj_full: list of bitmasks for 16 vertices
        # Order vertices by degree descending
        degrees = [bin(adj_full[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = [False] * 3
            # Check neighbors already colored
            for u in range(n):
                if color[u] != -1 and (adj_full[v] & (1 << u)):
                    used[color[u]] = True
            
            for c in range(3):
                if not used[c]:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    # Evaluate a configuration: assignment of independent sets to new vertices
    def evaluate(assignment):
        # assignment: list of num_new indices into independent_sets
        # Build full adjacency
        adj_full = [0] * n
        # Core edges
        for u, v in core_edges:
            adj_full[u] |= (1 << v)
            adj_full[v] |= (1 << u)
        
        # Edges from new vertices to core
        for i in range(num_new):
            v_new = new_vertices[i]
            s_mask = independent_sets[assignment[i]]
            # Set adjacency between v_new and core vertices in s_mask
            for u in range(core_n):
                if s_mask & (1 << u):
                    adj_full[v_new] |= (1 << u)
                    adj_full[u] |= (1 << v_new)
        
        # Determine possible edges between new vertices
        possible_new_edges = []
        for i in range(num_new):
            for j in range(i + 1, num_new):
                if disjoint[assignment[i]][assignment[j]]:
                    possible_new_edges.append((i, j))
        
        # Find max triangle-free subset of possible_new_edges
        m = len(possible_new_edges)
        best_mask = 0
        max_cnt = 0
        for mask in range(1 << m):
            adj_new = [0] * num_new
            for idx in range(m):
                if mask & (1 << idx):
                    u, v = possible_new_edges[idx]
                    adj_new[u] |= (1 << v)
                    adj_new[v] |= (1 << u)
            # Check triangle-free
            tf = True
            for u in range(num_new):
                for v in range(u + 1, num_new):
                    if adj_new[u] & (1 << v):
                        if adj_new[u] & adj_new[v]:
                            tf = False
                            break
                if not tf:
                    break
            if tf:
                cnt = bin(mask).count("1")
                if cnt > max_cnt:
                    max_cnt = cnt
                    best_mask = mask
        
        # Add best new-new edges
        for idx in range(m):
            if best_mask & (1 << idx):
                u, v = possible_new_edges[idx]
                adj_full[new_vertices[u]] |= (1 << new_vertices[v])
                adj_full[new_vertices[v]] |= (1 << new_vertices[u])
        
        # Check 3-colorability
        if is_3_colorable(adj_full):
            return -1, adj_full
        
        # Count edges
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_full[i] & (1 << j):
                    edge_count += 1
        
        return edge_count, adj_full

    # Branch and bound over assignments of independent sets to new vertices
    # We'll limit to top K independent sets for tractability
    K = min(30, num_sets)
    top_sets = list(range(K))
    
    best_score = -1
    best_assignment = None

    # DFS with branch and bound
    def dfs(k, assignment_prefix, current_core_edges):
        nonlocal best_score, best_assignment, deadline
        
        if time.time() > deadline:
            return
        
        if k == num_new:
            # Evaluate this assignment
            score, adj_full = evaluate(assignment_prefix)
            if score > best_score:
                best_score = score
                best_assignment = list(assignment_prefix)
            return
        
        # Upper bound: current_core_edges + max possible from remaining
        # Each new vertex can contribute at most max_set_size edges to core
        # Plus at most C(5,2)=10 edges between new vertices
        max_remaining_core = 0
        for r in range(k, num_new):
            max_remaining_core += max(set_sizes[s] for s in top_sets)
        max_possible = current_core_edges + max_remaining_core + 10
        
        if max_possible <= best_score:
            return
        
        for s_idx in top_sets:
            new_core_edges = current_core_edges + set_sizes[s_idx]
            dfs(k + 1, assignment_prefix + [s_idx], new_core_edges)

    dfs(0, [], 0)

    if best_assignment is None:
        # Fallback: return Grötzsch + isolated vertices
        return core_edges

    # Reconstruct edges from best_assignment
    final_edges = list(core_edges)
    for i in range(num_new):
        v_new = new_vertices[i]
        s_mask = independent_sets[best_assignment[i]]
        for u in range(core_n):
            if s_mask & (1 << u):
                final_edges.append([v_new, u])

    # Add new-new edges
    possible_new_edges = []
    for i in range(num_new):
        for j in range(i + 1, num_new):
            if disjoint[best_assignment[i]][best_assignment[j]]:
                possible_new_edges.append((i, j))

    m = len(possible_new_edges)
    max_new_edges = 0
    best_mask = 0
    for mask in range(1 << m):
        adj_subset = [0] * num_new
        for idx in range(m):
            if mask & (1 << idx):
                u, v = possible_new_edges[idx]
                adj_subset[u] |= (1 << v)
                adj_subset[v] |= (1 << u)

        tf = True
        for u in range(num_new):
            for v in range(u + 1, num_new):
                if adj_subset[u] & (1 << v):
                    if adj_subset[u] & adj_subset[v]:
                        tf = False
                        break
            if not tf:
                break

        if tf:
            cnt = bin(mask).count("1")
            if cnt > max_new_edges:
                max_new_edges = cnt
                best_mask = mask

    for idx in range(m):
        if best_mask & (1 << idx):
            u, v = possible_new_edges[idx]
            final_edges.append([new_vertices[u], new_vertices[v]])

    return final_edges
