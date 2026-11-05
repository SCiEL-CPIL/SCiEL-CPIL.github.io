import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    # Number of candidates
    M = len(candidates)
    
    # Map candidate index (0..M-1) to original vertex (0..N-1)
    cand_to_orig = candidates
    orig_to_cand = [-1] * N
    for i, v in enumerate(candidates):
        orig_to_cand[v] = i

    # Build adjacency in candidate space (M vertices)
    # adj_cand[i] is a bitmask of neighbors of candidate i in the candidate graph
    adj_cand = [0] * M
    for i in range(M):
        v = cand_to_orig[i]
        # Get all neighbors in original graph that are also candidates
        mask = adj[v]
        # Convert to candidate bitmask
        cmask = 0
        m = mask
        while m:
            lsb = m & -m
            orig_v = lsb.bit_length() - 1
            ci = orig_to_cand[orig_v]
            if ci != -1:
                cmask |= (1 << ci)
            m ^= lsb
        adj_cand[i] = cmask

    # Complement adjacency in candidate space (for Max Clique formulation)
    # comp_adj_cand[i] is a bitmask of non-neighbors of candidate i (excluding self)
    # This is the graph in which we seek a Maximum Clique
    comp_adj_cand = [0] * M
    full_cand_mask = (1 << M) - 1
    for i in range(M):
        # Non-neighbors are those in full_cand_mask, not i, and not in adj_cand[i]
        comp_adj_cand[i] = full_cand_mask & ~adj_cand[i] & ~(1 << i)

    # Precompute degrees in comp_adj_cand for sorting heuristic
    comp_deg = [0] * M
    for i in range(M):
        comp_deg[i] = comp_adj_cand[i].bit_count()

    # Sort vertices by degree descending for better search ordering
    vertex_order = sorted(range(M), key=lambda v: -comp_deg[v])
    # Create a mapping from original index to position in sorted order
    pos_in_order = [0] * M
    for pos, v in enumerate(vertex_order):
        pos_in_order[v] = pos

    def greedy_color_bound(P):
        """
        Returns upper bound on clique size in P using greedy coloring.
        Uses vertex ordering based on degree for tighter bounds.
        """
        if P == 0:
            return 0
        
        # Get list of vertices in P, sorted by degree descending
        verts = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Sort by degree in comp_adj_cand (descending)
        verts.sort(key=lambda v: -comp_deg[v])
        
        colors = [0] * M
        max_color = 0
        
        for v in verts:
            # Find available colors
            # Neighbors of v in P
            neighbors = comp_adj_cand[v] & P
            # Used colors by neighbors
            used = set()
            m_n = neighbors
            while m_n:
                lsb = m_n & -m_n
                u = lsb.bit_length() - 1
                if colors[u] > 0:
                    used.add(colors[u])
                m_n ^= lsb
            
            # Find smallest available color
            c = 1
            while c in used:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        return max_color

    def matching_bound(P):
        """
        Compute a tight upper bound using greedy matching.
        For maximum clique, this gives a bound based on the structure.
        Returns an upper bound on the clique size in P.
        """
        if P == 0:
            return 0
        
        # Get vertices in P
        verts = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        if not verts:
            return 0
        
        # Greedy matching: repeatedly pick the vertex with max degree in remaining graph
        # and match it with its neighbor that has max degree
        remaining = set(verts)
        matched = set()
        
        while remaining:
            # Find vertex with max degree in remaining subgraph
            best_v = -1
            best_deg = -1
            for v in remaining:
                deg = 0
                m_n = comp_adj_cand[v] & P
                while m_n:
                    lsb = m_n & -m_n
                    u = lsb.bit_length() - 1
                    if u in remaining:
                        deg += 1
                    m_n ^= lsb
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            # Find neighbor of best_v in remaining with max degree
            best_u = -1
            best_u_deg = -1
            m_n = comp_adj_cand[best_v] & P
            while m_n:
                lsb = m_n & -m_n
                u = lsb.bit_length() - 1
                if u in remaining:
                    deg = 0
                    m_nn = comp_adj_cand[u] & P
                    while m_nn:
                        lsb2 = m_nn & -m_nn
                        w = lsb2.bit_length() - 1
                        if w in remaining:
                            deg += 1
                        m_nn ^= lsb2
                    if deg > best_u_deg:
                        best_u_deg = deg
                        best_u = u
                m_n ^= lsb
            
            if best_u == -1:
                # best_v is isolated, it can be in the clique alone
                matched.add(best_v)
                remaining.remove(best_v)
            else:
                # Match best_v with best_u
                matched.add(best_v)
                matched.add(best_u)
                remaining.remove(best_v)
                remaining.remove(best_u)
        
        # The bound is the size of the matched set plus the number of remaining vertices
        # Actually, for maximum clique, a better bound is:
        # Each matched edge can contribute at most 2 to the clique, and each unmatched vertex at most 1
        # But this isn't tight. Let's use a simpler approach.
        
        # Alternative: use the fact that in a clique, no two vertices can be adjacent in the complement
        # So we need an independent set in the original graph
        # The matching bound for independent set: |V| - |matching|
        # But we want clique in complement = independent set in original
        
        # Let me reconsider: we're finding maximum clique in comp_adj_cand
        # This is equivalent to maximum independent set in adj_cand
        # For maximum independent set, a bound is: |V| - maximum matching size
        
        # Compute maximum matching in the induced subgraph on P (in adj_cand)
        # But that's expensive. Let's use a greedy matching.
        
        # Actually, let's just return the greedy color bound for now and improve separately
        return greedy_color_bound(P)

    best_clique_size = 0
    best_clique_mask = 0
    timeout_flag = [False]

    def expand_clique(P, current_clique):
        nonlocal best_clique_size, best_clique_mask, timeout_flag
        
        if time.time() - start_time > DEADLINE - 1.0:
            timeout_flag[0] = True
            return
            
        if P == 0:
            return
            
        # Bound
        ub = greedy_color_bound(P)
        if current_clique.bit_count() + ub <= best_clique_size:
            return
            
        # Select vertex v from P
        # Heuristic: pick vertex with max degree in P
        m = P
        best_v = -1
        max_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (comp_adj_cand[v] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
        
        v = best_v
        lv = 1 << v
        
        # Branch 1: Include v
        new_P = P & comp_adj_cand[v]
        expand_clique(new_P, current_clique | lv)
        
        if timeout_flag[0]:
            return
            
        # Branch 2: Exclude v
        P = P & ~lv
        expand_clique(P, current_clique)

    def greedy_clique_init():
        """
        Greedy construction of a clique.
        """
        P = full_cand_mask
        clique = 0
        while P:
            # Pick vertex with max degree in P
            m = P
            best_v = -1
            max_deg = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (comp_adj_cand[v] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                break
                
            lv = 1 << best_v
            clique |= lv
            P &= comp_adj_cand[best_v]
            
        return clique

    # Get initial lower bound
    init_clique = greedy_clique_init()
    best_clique_size = init_clique.bit_count()
    best_clique_mask = init_clique
    
    # Run multiple randomized greedy inits to get better lower bound
    for _ in range(10):
        if time.time() - start_time > DEADLINE - 3.0:
            break
        # Randomized greedy
        P = full_cand_mask
        clique = 0
        while P:
            m = P
            best_v = -1
            max_score = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (comp_adj_cand[v] & P).bit_count()
                score = deg + random.uniform(0, 2.0)
                if score > max_score:
                    max_score = score
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                break
                
            lv = 1 << best_v
            clique |= lv
            P &= comp_adj_cand[best_v]
            
        if clique.bit_count() > best_clique_size:
            best_clique_size = clique.bit_count()
            best_clique_mask = clique

    # Local search improvement
    def local_improve(clique_mask):
        """
        Try to improve the clique by adding vertices or swapping.
        """
        improved = True
        while improved:
            improved = False
            
            # Try adding vertices
            m = full_cand_mask & ~clique_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Check if v can be added: it must be adjacent to all in clique
                if (clique_mask & ~comp_adj_cand[v] & ~(1 << v)) == 0:
                    clique_mask |= (1 << v)
                    improved = True
                    break
                m ^= lsb
            
            # Try swapping: remove one vertex, add another
            if not improved:
                m_c = clique_mask
                while m_c:
                    lsb_c = m_c & -m_c
                    remove_v = lsb_c.bit_length() - 1
                    new_clique_without = clique_mask & ~(1 << remove_v)
                    
                    # Try adding a vertex not in new_clique_without
                    m_add = full_cand_mask & ~new_clique_without
                    while m_add:
                        lsb_a = m_add & -m_add
                        add_v = lsb_a.bit_length() - 1
                        # Check if add_v can be added to new_clique_without
                        if (new_clique_without & ~comp_adj_cand[add_v] & ~(1 << add_v)) == 0:
                            new_clique = new_clique_without | (1 << add_v)
                            if new_clique.bit_count() > clique_mask.bit_count():
                                clique_mask = new_clique
                                improved = True
                                break
                        m_add ^= lsb_a
                    
                    if improved:
                        break
                    m_c ^= lsb_c
        
        return clique_mask

    best_clique_mask = local_improve(best_clique_mask)
    best_clique_size = best_clique_mask.bit_count()

    # Run B&B
    expand_clique(full_cand_mask, 0)
    
    # Final local improvement
    best_clique_mask = local_improve(best_clique_mask)
    best_clique_size = best_clique_mask.bit_count()
    
    # Convert best_clique_mask (candidate indices) to original values
    result = []
    m = best_clique_mask
    while m:
        lsb = m & -m
        ci = lsb.bit_length() - 1
        orig_v = cand_to_orig[ci]
        result.append(orig_v + 1)
        m ^= lsb
        
    return sorted(result)
