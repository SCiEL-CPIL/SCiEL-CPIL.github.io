import time
import random
from math import gcd
from collections import defaultdict

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    deadline = time.time() + 85.0  # Safety margin

    # Precompute GCD table
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad triples
    # bad_triples is a list of sets of 3 elements
    # We also need a mapping: for each vertex v, list of pairs (u, w) such that {u, v, w} is bad
    # This allows us to quickly update E_forbidden when we select v
    
    pairs_for_vertex = defaultdict(list)
    all_bad_triples = []
    
    for i in range(1, 129):
        for j in range(i + 1, 129):
            g_ij = gcd_table[i][j]
            for k in range(j + 1, 129):
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    all_bad_triples.append((i, j, k))
                    # Add to adjacency lists for each vertex in the triple
                    # For vertex i, the forbidden pair is (j, k)
                    # For vertex j, the forbidden pair is (i, k)
                    # For vertex k, the forbidden pair is (i, j)
                    pairs_for_vertex[i].append((j, k))
                    pairs_for_vertex[j].append((i, k))
                    pairs_for_vertex[k].append((i, j))

    # Sort vertices by degree (number of bad triples they are in) to improve B&B
    vertex_degree = [0] * 129
    for v in range(1, 129):
        vertex_degree[v] = len(pairs_for_vertex[v])
    
    # We will use bitmasks for V_rem
    # 128 bits fit in Python int
    
    best_size = 0
    best_set = set()
    
    # Greedy initialization to get a good lower bound quickly
    def greedy_init():
        S = set()
        # Order by degree ascending? Or descending?
        # Let's try random restarts of greedy
        for _ in range(50):
            order = list(range(1, 129))
            random.shuffle(order)
            S = set()
            for v in order:
                # Check if v can be added
                # v can be added if no pair in S forms a bad triple with v
                # And no pair in S is "forbidden" by previous additions? 
                # In a simple greedy, we just check if adding v creates a bad triple with any two in S.
                conflict = False
                s_list = list(S)
                for i in range(len(s_list)):
                    a = s_list[i]
                    g_va = gcd_table[v][a]
                    for j in range(i + 1, len(s_list)):
                        b = s_list[j]
                        if g_va == gcd_table[v][b] == gcd_table[a][b]:
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    S.add(v)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                
    greedy_init()
    
    # Branch and Bound
    # State: (V_rem_mask, E_forbidden_set, S_size)
    # We don't need to track S explicitly, just its size, because we only care about max size.
    # But we need to reconstruct the set if we want to return it.
    # For B&B, we just track size. At the end, we can run a local search on the best found set
    # or store the path. Storing the path is easy: pass a list of selected vertices.
    
    # To save memory and time, we only keep the size and the current set if it improves best.
    
    def greedy_color_bound(V_rem_mask, E_forbidden):
        """
        Compute an upper bound on the size of the MIS in the graph (V_rem, E_forbidden).
        Uses greedy coloring.
        """
        if V_rem_mask == 0:
            return 0
        
        # Extract vertices
        vertices = []
        m = V_rem_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertices.append(idx)
            m ^= lsb
        
        # Build adjacency list for the subgraph
        # Vertices are 0..127 (indices 1..128 in problem)
        # Map idx to 0..n-1 for coloring?
        # We can just use the original indices.
        
        # Check if any edge exists
        # E_forbidden is a set of tuples (u, v) with u < v
        
        # Greedy Coloring
        color = {}
        next_color = 0
        # Order vertices by degree in this subgraph for better coloring?
        # Simple order is fine for bound, but degree ordering gives tighter bound.
        
        # Calculate degrees in subgraph
        deg = {v: 0 for v in vertices}
        for u, v in E_forbidden:
            if u in deg and v in deg:
                deg[u] += 1
                deg[v] += 1
        
        # Sort by degree descending
        sorted_verts = sorted(vertices, key=lambda x: deg[x], reverse=True)
        
        for v in sorted_verts:
            # Find smallest available color
            used_colors = set()
            for u, w in E_forbidden:
                if u == v and w in color:
                    used_colors.add(color[w])
                elif w == v and u in color:
                    used_colors.add(color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c > next_color:
                next_color = c
        
        return next_color + 1 # Number of colors

    # To speed up, we can pre-compute the list of edges for each vertex
    # But E_forbidden changes dynamically.
    
    # Optimization: If the number of edges is 0, bound is len(vertices).
    
    def branch_and_bound(V_rem_mask, E_forbidden, S_size, selected_list):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if V_rem_mask == 0:
            if S_size > best_size:
                best_size = S_size
                best_set = set(selected_list)
            return
        
        # Bound
        # Quick check: if S_size + popcount(V_rem_mask) <= best_size, prune
        # popcount is fast
        rem_count = bin(V_rem_mask).count('1')
        if S_size + rem_count <= best_size:
            return
            
        # Tighter bound: Greedy Coloring
        # Only compute if rem_count is large enough to be worth it?
        # Or always?
        # Greedy coloring on 128 nodes is fast.
        bound = greedy_color_bound(V_rem_mask, E_forbidden)
        if S_size + bound <= best_size:
            return
            
        # Select vertex to branch on
        # Heuristic: Vertex with highest degree in the current conflict graph (E_forbidden)
        # among V_rem
        
        # Extract vertices
        vertices = []
        m = V_rem_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertices.append(idx)
            m ^= lsb
            
        # Calculate degrees in (V_rem, E_forbidden)
        deg = {v: 0 for v in vertices}
        # Filter E_forbidden to only include edges within V_rem
        # This is O(|E|) which can be large.
        # To optimize, we can maintain a set of edges and check membership.
        # For now, just iterate E_forbidden.
        for u, v in E_forbidden:
            if u in deg and v in deg:
                deg[u] += 1
                deg[v] += 1
                
        # Pick max degree vertex
        v = max(vertices, key=lambda x: deg[x])
        
        # Branch 1: Exclude v
        new_mask = V_rem_mask & ~(1 << v)
        branch_and_bound(new_mask, E_forbidden, S_size, selected_list)
        
        # Branch 2: Include v
        # Update E_forbidden
        new_E = set(E_forbidden)
        # Add pairs that form bad triples with v
        # pairs_for_vertex[v] contains (u, w) such that {u, v, w} is bad
        # We only add pairs where u, w are in V_rem (and not v, which is removed)
        for u, w in pairs_for_vertex[v]:
            # u and w must be in new_mask
            if (new_mask & (1 << u)) and (new_mask & (1 << w)):
                # Ensure u < w for consistent tuple
                if u > w:
                    u, w = w, u
                new_E.add((u, w))
                
        # Also, we must ensure that we don't select v if it conflicts with existing E_forbidden?
        # No, E_forbidden represents constraints between *other* vertices.
        # v is being selected. The constraints *involving* v are now "satisfied" by the fact that we are not selecting the pairs (u,w) that conflict with v.
        # Wait, the logic is:
        # If we select v, we CANNOT select any pair (u,w) such that {u,v,w} is bad.
        # So we add (u,w) to the forbidden set for the *remaining* problem.
        # This is correct.
        
        # Recurse
        branch_and_bound(new_mask, new_E, S_size + 1, selected_list + [v])

    # Start B&B
    initial_mask = (1 << 128) - 1
    initial_E = set()
    # Start with the greedy result as a lower bound
    # We can also start the B&B from the greedy set to potentially improve it?
    # No, B&B explores the whole space. The lower bound just prunes.
    
    # To make it faster, we can use the greedy set to initialize best_size.
    # We already did that in greedy_init.
    
    # Run B&B
    branch_and_bound(initial_mask, initial_E, 0, [])

    # Final polish: Local search on the best found set
    # The B&B might find the global optimum, but if it timed out, we have a good candidate.
    # Let's run a simple local search (add/swap) to see if we can improve further.
    
    def local_search(selected, max_iters=50):
        selected = set(selected)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try adding any excluded element
            excluded = [x for x in range(1, 129) if x not in selected]
            for x in excluded:
                # Check if x can be added
                conflict = False
                s_list = list(selected)
                for i in range(len(s_list)):
                    a = s_list[i]
                    g_xa = gcd_table[x][a]
                    for j in range(i + 1, len(s_list)):
                        b = s_list[j]
                        if g_xa == gcd_table[x][b] == gcd_table[a][b]:
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    selected.add(x)
                    improved = True
                    break
            
            # If no add, try 1-for-2 exchange
            if not improved:
                for x in list(selected):
                    temp = selected - {x}
                    added = []
                    for y in excluded:
                        if y == x:
                            continue
                        # Check if y can be added to temp + added
                        conflict = False
                        t_list = list(temp | set(added))
                        for i in range(len(t_list)):
                            a = t_list[i]
                            g_ya = gcd_table[y][a]
                            for j in range(i + 1, len(t_list)):
                                b = t_list[j]
                                if g_ya == gcd_table[y][b] == gcd_table[a][b]:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if not conflict:
                            added.append(y)
                    if len(added) >= 2:
                        selected = temp | set(added)
                        improved = True
                        break

        return selected

    if best_set:
        s_final = local_search(best_set, max_iters=100)
        if len(s_final) > best_size:
            best_size = len(s_final)
            best_set = s_final

    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
