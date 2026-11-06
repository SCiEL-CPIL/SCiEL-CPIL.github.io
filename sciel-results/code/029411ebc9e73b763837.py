import time
import random
import numpy as np

def _build_adj_masks(colors, n):
    """Build adjacency bitmasks for both colors."""
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    return adj1, adj0

def _edge_index_table(n):
    """Build (i,j) -> edge_index table and edge_list for fast lookup."""
    table = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            table[i][j] = idx
            table[j][i] = idx
            edge_list.append((i, j))
            idx += 1
    return table, edge_list

def _count_k222_masks(adj, n):
    """Count K2,2,2 subgraphs using bitmask adjacency (fast for n<=31)."""
    count = 0
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            verts_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                verts_c.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts_c)
            for i in range(m):
                a = verts_c[i]
                for j in range(i+1, m):
                    b = verts_c[j]
                    if adj[a] & (1 << b):
                        continue
                    D = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    if D.bit_count() < 2:
                        continue
                    verts_d = []
                    mask2 = D
                    while mask2:
                        lsb = mask2 & -mask2
                        verts_d.append(lsb.bit_length() - 1)
                        mask2 ^= lsb
                    k = len(verts_d)
                    for x in range(k):
                        c = verts_d[x]
                        for y in range(x+1, k):
                            d = verts_d[y]
                            if not (adj[c] & (1 << d)):
                                count += 1
    return count

def _delta_k222_color1(adj1, i, j, n):
    """
    Compute change in K2,2,2 count for color 1 when edge (i,j) is flipped.
    Returns (delta_add, delta_remove) where:
      - If adding edge (i,j) to color 1: new_count = old_count + delta_add - delta_remove
      - If removing edge (i,j) from color 1: new_count = old_count - delta_add + delta_remove
    Actually: delta = new - old.
    Adding edge (i,j): delta = (# new K222 that use (i,j) as P1 edge) - (# old K222 that use (i,j) as P3 edge)
    Wait, let me think more carefully.
    
    K2,2,2 has 3 parts of 2 vertices each. Edge (i,j) can appear in different roles:
    - As an edge within P1 (the first non-adjacent pair): but P1 vertices are non-adjacent, so (i,j) can't be in P1
    - As an edge within P2: P2 vertices are non-adjacent, so (i,j) can't be in P2  
    - As an edge within P3: P3 vertices are non-adjacent, so (i,j) can't be in P3
    - As a cross-edge between P1 and P2: e.g., i in P1, j in P2
    - As a cross-edge between P1 and P3: e.g., i in P1, j in P3
    - As a cross-edge between P2 and P3: e.g., i in P2, j in P3
    
    So (i,j) is always a cross-edge in K2,2,2. When we add edge (i,j) to color 1,
    we create new K2,2,2 in color 1 where (i,j) is a cross-edge.
    When we remove edge (i,j) from color 1, we destroy existing K2,2,2 in color 1
    where (i,j) is a cross-edge.
    
    For adding (i,j) to color 1:
    New K2,2,2 formed where (i,j) is a cross-edge:
    Case A: i in P1={i,i'}, j in P2={j,j'}
      Need: i' adjacent to i, i' adjacent to j, i' adjacent to j' (all in color 1)
            j' adjacent to i, j' adjacent to j'... wait j' is in P2 so j' not adjacent to j
            P3={c,d}: c,d both adjacent to i,i',j,j' in color 1, c not adjacent to d
      So: i' in N1(i)∩N1(j), j' in N1(i)∩N1(j), i'≠j', i' not adj j', j' not adj i' (same)
          Then P3: c,d in N1(i)∩N1(i')∩N1(j)∩N1(j'), c not adj d
      
    This is getting complex. Let me just compute the delta by checking which K2,2,2
    subgraphs are affected.
    """
    # When adding edge (i,j) to color 1:
    # New K2,2,2 in color 1 that include edge (i,j) as a cross-edge.
    # (i,j) connects a vertex in one part to a vertex in another part.
    
    # The 6 vertices are {i, j, p, q, r, s} where:
    # Parts are 2-element sets, (i,j) is a cross-edge.
    # WLOG, assume i is in part A, j is in part B.
    # Part A = {i, a}, Part B = {j, b}, Part C = {c, d}
    # Constraints for color 1 K2,2,2:
    #   All cross-edges present: i-j, i-b, a-j, a-b, i-c, i-d, a-c, a-d, j-c, j-d, b-c, b-d
    #   No within-part edges: i-a absent, j-b absent, c-d absent
    #   (i,j) is the new edge we're adding, so it was absent before.
    
    # For the new K2,2,2 to exist after adding (i,j):
    #   a must be in N1(i) ∩ N1(j) \ {i,j} and NOT adjacent to i (i-a absent) and NOT adjacent to j (a-j... wait a IS adjacent to j)
    #   Actually: a is in Part A with i. i-a must be absent. a-j must be present. a-b must be present.
    #   a must be: adjacent to j (a-j present), NOT adjacent to i (i-a absent)
    #   Similarly b: adjacent to i (i-b present), NOT adjacent to j (j-b absent)
    #   c,d: adjacent to i, a, j, b; NOT adjacent to each other.
    
    # So:
    #   a ∈ N1(j) \ N1(i) \ {j}  (adjacent to j, not to i)
    #   b ∈ N1(i) \ N1(j) \ {i}  (adjacent to i, not to j)
    #   Additionally: a-b must be present (cross-edge between A and B)
    #   c,d ∈ N1(i) ∩ N1(a) ∩ N1(j) ∩ N1(b), c-d absent
    
    # But wait, we also need a-b present. And a is not adjacent to i, b not adjacent to j.
    # Also need: a adjacent to b (cross-edge A-B), a adjacent to c,d, b adjacent to c,d.
    
    # Let me enumerate:
    delta = 0
    
    # Case 1: i in Part A, j in Part B
    # a ∈ N1(j) \ N1(i) \ {i,j}: a adjacent to j, NOT adjacent to i
    # b ∈ N1(i) \ N1(j) \ {i,j}: b adjacent to i, NOT adjacent to j
    # Need a-b edge present
    # c,d ∈ N1(i)∩N1(a)∩N1(j)∩N1(b), c-d absent
    
    N1i = adj1[i]
    N1j = adj1[j]
    
    # a: adjacent to j, not adjacent to i, a != i, a != j
    candidates_a = N1j & ~N1i & ~(1 << i) & ~(1 << j)
    # b: adjacent to i, not adjacent to j, b != i, b != j
    candidates_b = N1i & ~N1j & ~(1 << i) & ~(1 << j)
    
    # For each (a, b) pair with a-b edge:
    mask_a = candidates_a
    while mask_a:
        lsb_a = mask_a & -mask_a
        a = lsb_a.bit_length() - 1
        mask_a ^= lsb_a
        
        # b must be in candidates_b AND adjacent to a
        mask_b = candidates_b & N1i & adj1[a] & ~(1 << a)
        while mask_b:
            lsb_b = mask_b & -mask_b
            b = lsb_b.bit_length() - 1
            mask_b ^= lsb_b
            
            # c,d in N1(i) ∩ N1(a) ∩ N1(j) ∩ N1(b), c-d absent
            C_cd = N1i & adj1[a] & N1j & adj1[b] & ~(1 << i) & ~(1 << a) & ~(1 << j) & ~(1 << b)
            if C_cd.bit_count() < 2:
                continue
            verts_cd = []
            mask_cd = C_cd
            while mask_cd:
                lsb_cd = mask_cd & -mask_cd
                verts_cd.append(lsb_cd.bit_length() - 1)
                mask_cd ^= lsb_cd
            kc = len(verts_cd)
            for x in range(kc):
                c = verts_cd[x]
                for y in range(x+1, kc):
                    d = verts_cd[y]
                    if not (adj1[c] & (1 << d)):
                        delta += 1
    
    # Case 2: i in Part A, j in Part C
    # Part A = {i, a}, Part B = {b, e}, Part C = {j, f}
    # (i,j) is cross-edge between A and C.
    # a: in A with i. i-a absent. a-b, a-e, a-j, a-f present.
    # b,e: in B. b-e absent. b-i, b-a, b-j, b-f present. e similar.
    # f: in C with j. j-f absent. f-i, f-a, f-b, f-e present.
    # 
    # a ∈ N1(j) \ N1(i) \ {i,j} (adj to j, not to i)
    # f ∈ N1(i) \ N1(j) \ {i,j} (adj to i, not to j)
    # b,e ∈ N1(i) ∩ N1(a) ∩ N1(j) ∩ N1(f), b-e absent
    # Also need a-f present (cross A-C), and a-b, a-e, f-b, f-e present (already in N1 sets)
    
    candidates_a2 = N1j & ~N1i & ~(1 << i) & ~(1 << j)
    candidates_f = N1i & ~N1j & ~(1 << i) & ~(1 << j)
    
    mask_a2 = candidates_a2
    while mask_a2:
        lsb_a2 = mask_a2 & -mask_a2
        a = lsb_a2.bit_length() - 1
        mask_a2 ^= lsb_a2
        
        # f must be in candidates_f AND adjacent to a
        mask_f = candidates_f & adj1[a] & ~(1 << a)
        while mask_f:
            lsb_f = mask_f & -mask_f
            f = lsb_f.bit_length() - 1
            mask_f ^= lsb_f
            
            # b,e in N1(i) ∩ N1(a) ∩ N1(j) ∩ N1(f), b-e absent
            C_be = N1i & adj1[a] & N1j & adj1[f] & ~(1 << i) & ~(1 << a) & ~(1 << j) & ~(1 << f)
            if C_be.bit_count() < 2:
                continue
            verts_be = []
            mask_be = C_be
            while mask_be:
                lsb_be = mask_be & -mask_be
                verts_be.append(lsb_be.bit_length() - 1)
                mask_be ^= lsb_be
            kb = len(verts_be)
            for x in range(kb):
                b = verts_be[x]
                for y in range(x+1, kb):
                    e = verts_be[y]
                    if not (adj1[b] & (1 << e)):
                        delta += 1
    
    # Case 3: i in Part B, j in Part A (symmetric to Case 1 with i,j swapped)
    # This is the same as Case 1 but with roles swapped. Since we're counting unordered
    # K2,2,2 subgraphs, and the parts are labeled, we need to be careful.
    # Actually, K2,2,2 has 3 parts. The edge (i,j) can connect:
    #   Part 1 to Part 2 (Case 1)
    #   Part 1 to Part 3 (Case 2)
    #   Part 2 to Part 3 (Case 3: i in P2, j in P3)
    
    # Case 3: i in Part B, j in Part C
    # Part A = {c, d}, Part B = {i, b}, Part C = {j, f}
    # (i,j) cross-edge between B and C.
    # b: in B with i. i-b absent. b-c, b-d, b-j, b-f present.
    # f: in C with j. j-f absent. f-c, f-d, f-i, f-b present.
    # c,d: in A. c-d absent. c-i, c-b, c-j, c-f present. d similar.
    #
    # b ∈ N1(j) \ N1(i) \ {i,j} (adj to j, not to i)
    # f ∈ N1(i) \ N1(j) \ {i,j} (adj to i, not to j)
    # c,d ∈ N1(i) ∩ N1(b) ∩ N1(j) ∩ N1(f), c-d absent
    # Also need b-f present.
    
    candidates_b3 = N1j & ~N1i & ~(1 << i) & ~(1 << j)
    candidates_f3 = N1i & ~N1j & ~(1 << i) & ~(1 << j)
    
    mask_b3 = candidates_b3
    while mask_b3:
        lsb_b3 = mask_b3 & -mask_b3
        b = lsb_b3.bit_length() - 1
        mask_b3 ^= lsb_b3
        
        mask_f3 = candidates_f3 & adj1[b] & ~(1 << b)
        while mask_f3:
            lsb_f3 = mask_f3 & -mask_f3
            f = lsb_f3.bit_length() - 1
            mask_f3 ^= lsb_f3
            
            C_cd3 = N1i & adj1[b] & N1j & adj1[f] & ~(1 << i) & ~(1 << b) & ~(1 << j) & ~(1 << f)
            if C_cd3.bit_count() < 2:
                continue
            verts_cd3 = []
            mask_cd3 = C_cd3
            while mask_cd3:
                lsb_cd3 = mask_cd3 & -mask_cd3
                verts_cd3.append(lsb_cd3.bit_length() - 1)
                mask_cd3 ^= lsb_cd3
            kc3 = len(verts_cd3)
            for x in range(kc3):
                c = verts_cd3[x]
                for y in range(x+1, kc3):
                    d = verts_cd3[y]
                    if not (adj1[c] & (1 << d)):
                        delta += 1
    
    return delta

def _has_k222_mask(adj, n):
    """
    Check if graph with bitmask adjacency 'adj' on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            
            vertices_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                vertices_in_c.append(v_idx)
                mask ^= lsb
                
            m = len(vertices_in_c)
            for i in range(m):
                a = vertices_in_c[i]
                for j in range(i + 1, m):
                    b = vertices_in_c[j]
                    if adj[a] & (1 << b):
                        continue
                    
                    neighbors_ab_in_c = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    if neighbors_ab_in_c.bit_count() < 2:
                        continue
                    
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                        
                    k = len(verts_ab)
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj[c] & (1 << d)):
                                return True
    return False

def _count_k222_masks(adj, n):
    """Count K2,2,2 subgraphs using bitmask adjacency."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            verts_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                verts_c.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts_c)
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj[a] & (1 << b):
                        continue
                    D = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    if D.bit_count() < 2:
                        continue
                    verts_d = []
                    mask2 = D
                    while mask2:
                        lsb = mask2 & -mask2
                        verts_d.append(lsb.bit_length() - 1)
                        mask2 ^= lsb
                    k = len(verts_d)
                    for x in range(k):
                        c = verts_d[x]
                        for y in range(x + 1, k):
                            d = verts_d[y]
                            if not (adj[c] & (1 << d)):
                                count += 1
    return count

def _edge_index_table(n):
    """Build (i,j) -> edge_index table and edge_list."""
    table = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            table[i][j] = idx
            table[j][i] = idx
            edge_list.append((i, j))
            idx += 1
    return table, edge_list

def _delta_violations_flip1(adj1, adj0, i, j, n):
    """
    Compute change in total violations when flipping edge (i,j) from color 0 to color 1.
    Returns delta = new_total - old_total.
    
    When adding (i,j) to color 1:
    - New K2,2,2 in color 1 where (i,j) is a cross-edge
    - Destroyed K2,2,2 in color 0 where (i,j) was a cross-edge
    """
    # Helper: count new K2,2,2 in a graph when adding edge (i,j)
    def _new_k222_on_add(adj, i, j, n):
        """Count K2,2,2 subgraphs that include edge (i,j) as a cross-edge,
        assuming (i,j) is NOT yet in adj (we're about to add it)."""
        N1i = adj[i]
        N1j = adj[j]
        count = 0
        
        # (i,j) is a cross-edge. WLOG i in part A, j in part B or C.
        # Part A={i,a}, Part B={j,b}, Part C={c,d}
        # All cross-edges present, within-part edges absent.
        # (i,j) is the new edge.
        
        # Case 1: i in P1, j in P2. P1={i,a}, P2={j,b}, P3={c,d}
        # a: adj to j (a-j), NOT adj to i (i-a absent)
        # b: adj to i (i-b), NOT adj to j (j-b absent)
        # a-b: present
        # c,d: adj to i,a,j,b; c-d absent
        cand_a = N1j & ~N1i & ~(1 << i) & ~(1 << j)
        cand_b = N1i & ~N1j & ~(1 << i) & ~(1 << j)
        
        ma = cand_a
        while ma:
            la = ma & -ma
            a = la.bit_length() - 1
            ma ^= la
            # b in cand_b, adj to a
            mb = cand_b & adj[a] & ~(1 << a)
            while mb:
                lb = mb & -mb
                b = lb.bit_length() - 1
                mb ^= lb
                # c,d in N(i)∩N(a)∩N(j)∩N(b), c-d absent
                Ccd = N1i & adj[a] & N1j & adj[b] & ~(1 << i) & ~(1 << a) & ~(1 << j) & ~(1 << b)
                if Ccd.bit_count() < 2:
                    continue
                vd = []
                mc = Ccd
                while mc:
                    lc = mc & -mc
                    vd.append(lc.bit_length() - 1)
                    mc ^= lc
                kc = len(vd)
                for x in range(kc):
                    c = vd[x]
                    for y in range(x+1, kc):
                        d = vd[y]
                        if not (adj[c] & (1 << d)):
                            count += 1
        
        # Case 2: i in P1, j in P3. P1={i,a}, P2={b,e}, P3={j,f}
        # a: adj to j, NOT adj to i
        # f: adj to i, NOT adj to j
        # a-f: present
        # b,e: adj to i,a,j,f; b-e absent
        cand_a2 = N1j & ~N1i & ~(1 << i) & ~(1 << j)
        cand_f = N1i & ~N1j & ~(1 << i) & ~(1 << j)
        
        ma2 = cand_a2
        while ma2:
            la2 = ma2 & -ma2
            a = la2.bit_length() - 1
            ma2 ^= la2
            mf = cand_f & adj[a] & ~(1 << a)
            while mf:
                lf = mf & -mf
                f = lf.bit_length() - 1
                mf ^= lf
                Cbe = N1i & adj[a] & N1j & adj[f] & ~(1 << i) & ~(1 << a) & ~(1 << j) & ~(1 << f)
                if Cbe.bit_count() < 2:
                    continue
                vb = []
                mb2 = Cbe
                while mb2:
                    lb2 = mb2 & -mb2
                    vb.append(lb2.bit_length() - 1)
                    mb2 ^= lb2
                kb = len(vb)
                for x in range(kb):
                    b = vb[x]
                    for y in range(x+1, kb):
                        e = vb[y]
                        if not (adj[b] & (1 << e)):
                            count += 1
        
        # Case 3: i in P2, j in P3. P1={c,d}, P2={i,b}, P3={j,f}
        # b: adj to j, NOT adj to i
        # f: adj to i, NOT adj to j
        # b-f: present
        # c,d: adj to i,b,j,f; c-d absent
        cand_b3 = N1j & ~N1i & ~(1 << i) & ~(1 << j)
        cand_f3 = N1i & ~N1j & ~(1 << i) & ~(1 << j)
        
        mb3 = cand_b3
        while mb3:
            lb3 = mb3 & -mb3
            b = lb3.bit_length() - 1
            mb3 ^= lb3
            mf3 = cand_f3 & adj[b] & ~(1 << b)
            while mf3:
                lf3 = mf3 & -mf3
                f = lf3.bit_length() - 1
                mf3 ^= lf3
                Ccd3 = N1i & adj[b] & N1j & adj[f] & ~(1 << i) & ~(1 << b) & ~(1 << j) & ~(1 << f)
                if Ccd3.bit_count() < 2:
                    continue
                vc = []
                mc3 = Ccd3
                while mc3:
                    lc3 = mc3 & -mc3
                    vc.append(lc3.bit_length() - 1)
                    mc3 ^= lc3
                kc3 = len(vc)
                for x in range(kc3):
                    c = vc[x]
                    for y in range(x+1, kc3):
                        d = vc[y]
                        if not (adj[c] & (1 << d)):
                            count += 1
        
        return count
    
    # New K2,2,2 in color 1 (adding (i,j) to color 1)
    new_in_c1 = _new_k222_on_add(adj1, i, j, n)
    
    # Destroyed K2,2,2 in color 0 (removing (i,j) from color 0)
    # Same computation but on adj0
    destroyed_in_c0 = _new_k222_on_add(adj0, i, j, n)
    
    return new_in_c1 - destroyed_in_c0

def _delta_violations_flip0(adj1, adj0, i, j, n):
    """
    Compute change in total violations when flipping edge (i,j) from color 1 to color 0.
    """
    N1i = adj1[i]
    N1j = adj1[j]
    N0i = adj0[i]
    N0j = adj0[j]
    
    def _new_k222_on_add(adj, i, j, n):
        N1i_l = adj[i]
        N1j_l = adj[j]
        count = 0
        cand_a = N1j_l & ~N1i_l & ~(1 << i) & ~(1 << j)
        cand_b = N1i_l & ~N1j_l & ~(1 << i) & ~(1 << j)
        ma = cand_a
        while ma:
            la = ma & -ma
            a = la.bit_length() - 1
            ma ^= la
            mb = cand_b & adj[a] & ~(1 << a)
            while mb:
                lb = mb & -mb
                b = lb.bit_length() - 1
                mb ^= lb
                Ccd = N1i_l & adj[a] & N1j_l & adj[b] & ~(1 << i) & ~(1 << a) & ~(1 << j) & ~(1 << b)
                if Ccd.bit_count() < 2:
                    continue
                vd = []
                mc = Ccd
                while mc:
                    lc = mc & -mc
                    vd.append(lc.bit_length() - 1)
                    mc ^= lc
                kc = len(vd)
                for x in range(kc):
                    c = vd[x]
                    for y in range(x+1, kc):
                        d = vd[y]
                        if not (adj[c] & (1 << d)):
                            count += 1
        cand_a2 = N1j_l & ~N1i_l & ~(1 << i) & ~(1 << j)
        cand_f = N1i_l & ~N1j_l & ~(1 << i) & ~(1 << j)
        ma2 = cand_a2
        while ma2:
            la2 = ma2 & -ma2
            a = la2.bit_length() - 1
            ma2 ^= la2
            mf = cand_f & adj[a] & ~(1 << a)
            while mf:
                lf = mf & -mf
                f = lf.bit_length() - 1
                mf ^= lf
                Cbe = N1i_l & adj[a] & N1j_l & adj[f] & ~(1 << i) & ~(1 << a) & ~(1 << j) & ~(1 << f)
                if Cbe.bit_count() < 2:
                    continue
                vb = []
                mb2 = Cbe
                while mb2:
                    lb2 = mb2 & -mb2
                    vb.append(lb2.bit_length() - 1)
                    mb2 ^= lb2
                kb = len(vb)
                for x in range(kb):
                    b = vb[x]
                    for y in range(x+1, kb):
                        e = vb[y]
                        if not (adj[b] & (1 << e)):
                            count += 1
        cand_b3 = N1j_l & ~N1i_l & ~(1 << i) & ~(1 << j)
        cand_f3 = N1i_l & ~N1j_l & ~(1 << i) & ~(1 << j)
        mb3 = cand_b3
        while mb3:
            lb3 = mb3 & -mb3
            b = lb3.bit_length() - 1
            mb3 ^= lb3
            mf3 = cand_f3 & adj[b] & ~(1 << b)
            while mf3:
                lf3 = mf3 & -mf3
                f = lf3.bit_length() - 1
                mf3 ^= lf3
                Ccd3 = N1i_l & adj[b] & N1j_l & adj[f] & ~(1 << i) & ~(1 << b) & ~(1 << j) & ~(1 << f)
                if Ccd3.bit_count() < 2:
                    continue
                vc = []
                mc3 = Ccd3
                while mc3:
                    lc3 = mc3 & -mc3
                    vc.append(lc3.bit_length() - 1)
                    mc3 ^= lc3
                kc3 = len(vc)
                for x in range(kc3):
                    c = vc[x]
                    for y in range(x+1, kc3):
                        d = vc[y]
                        if not (adj[c] & (1 << d)):
                            count += 1
        return count
    
    # Destroyed K2,2,2 in color 1 (removing (i,j) from color 1)
    destroyed_c1 = _new_k222_on_add(adj1, i, j, n)
    # New K2,2,2 in color 0 (adding (i,j) to color 0)
    new_c0 = _new_k222_on_add(adj0, i, j, n)
    
    return new_c0 - destroyed_c1

def _count_k222_violations(adj1, adj0, n):
    """Count total K2,2,2 violations in both colors. Returns (count1, count0)."""
    count1 = _count_k222_masks(adj1, n)
    count0 = _count_k222_masks(adj0, n)
    return count1, count0

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _count_k222_numpy(adj_mat, n):
    """
    Count K2,2,2 subgraphs using numpy matrix operations.
    adj_mat is an n x n boolean numpy array (symmetric, zero diagonal).
    K2,2,2: vertices {u,v} (non-adjacent), {a,b} (non-adjacent, both in N(u)∩N(v)),
    {c,d} (non-adjacent, both in N(a)∩N(b)∩N(u)∩N(v))
    """
    # Non-adjacency matrix (excluding self-loops)
    non_adj = ~adj_mat
    np.fill_diagonal(non_adj, False)
    
    # For each pair (u,v) non-adjacent: C = N(u) ∩ N(v)
    # adj_mat[u] & adj_mat[v] gives common neighbors
    common_neighbors = adj_mat @ adj_mat  # (i,j) = |N(i) ∩ N(j)|... no, this is count
    
    # Actually, let's use a different approach:
    # For each non-edge (u,v), compute C = adj[u] & adj[v]
    # Then for each non-edge (a,b) within C, compute D = (adj[a] & adj[b] & C) \ {a,b}
    # Count non-edges in D
    
    count = 0
    # Precompute: for all pairs, whether they are non-adjacent
    # Use broadcasting for efficiency
    
    # Iterate over non-adjacent pairs (u,v)
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mat[u, v]:
                continue
            
            # C = common neighbors of u and v
            C = adj_mat[u] & adj_mat[v]
            c_count = C.sum()
            if c_count < 4:
                continue
            
            # Get indices of vertices in C
            verts_c = np.where(C)[0]
            m = len(verts_c)
            
            # For each non-adjacent pair (a,b) in C
            # adj_mat[a,b] should be False
            # D = (adj[a] & adj[b] & C) excluding a,b
            # Count non-edges in D
            
            # Extract submatrix for verts_c
            sub_adj = adj_mat[np.ix_(verts_c, verts_c)]
            
            # Non-adjacent pairs within C
            non_adj_sub = ~sub_adj
            np.fill_diagonal(non_adj_sub, False)
            
            # For each non-adjacent pair (i,j) in submatrix:
            # D = sub_adj[i] & sub_adj[j] & C_mask, excluding i,j
            # Then count non-edges in D
            
            # Vectorize: for all pairs (i,j) with non_adj_sub[i,j]:
            # common = sub_adj[i] & sub_adj[j]
            # But we need to exclude i and j from common
            # And then count non-edges among those common neighbors
            
            # This is getting complex. Let's do a hybrid approach.
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj_mat[a, b]:
                        continue
                    
                    # D = (adj[a] & adj[b] & C) \ {a,b}
                    D = adj_mat[a] & adj_mat[b] & C
                    D[a] = False
                    D[b] = False
                    d_count = D.sum()
                    if d_count < 2:
                        continue
                    
                    # Count non-edges in D
                    verts_d = np.where(D)[0]
                    k = len(verts_d)
                    if k < 2:
                        continue
                    sub_d = adj_mat[np.ix_(verts_d, verts_d)]
                    non_edges_d = (~sub_d).sum() // 2  # symmetric
                    # Actually need to count pairs that are non-adjacent
                    # non_adj_d = ~sub_d, fill diagonal False, count True
                    non_adj_d = ~sub_d
                    np.fill_diagonal(non_adj_d, False)
                    count += non_adj_d.sum() // 2
    return count

def _local_search_vec(colors, n, max_iters=50000, time_limit=50, temp_start=1.0, temp_end=0.01):
    """
    Vectorized local search with simulated annealing.
    Uses numpy adjacency matrix for fast updates.
    """
    start_time = time.time()
    
    # Build numpy adjacency matrix for color 1
    adj1 = np.zeros((n, n), dtype=np.bool_)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i, j] = True
                adj1[j, i] = True
            idx += 1
    
    # Precompute edge index: (i,j) -> index in colors list
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    # Initial violation count
    def get_violation_count():
        """Count total K2,2,2 violations in both colors."""
        # Color 1 violations
        v1 = _count_k222_numpy(adj1, n)
        # Color 0 is complement
        adj0 = ~adj1
        np.fill_diagonal(adj0, False)
        v0 = _count_k222_numpy(adj0, n)
        return v1 + v0
    
    cur_violations = get_violation_count()
    if cur_violations == 0:
        return True
    
    best_violations = cur_violations
    best_colors = colors[:]
    
    # Simulated annealing parameters
    it = 0
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        progress = it / max_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Pick an edge to flip
        # Strategy: with some probability, pick an edge involved in a violation
        # Otherwise, random edge
        if random.random() < 0.7:
            # Try to find an edge in a violation
            # Pick random non-edge in color 1
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj1[u, v]:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    idx_flip = random.randint(0, len(colors) - 1)
                    it += 1
                    continue
            
            C = adj1[u] & adj1[v]
            if C.sum() < 4:
                idx_flip = random.randint(0, len(colors) - 1)
            else:
                verts_c = np.where(C)[0]
                m = len(verts_c)
                found = False
                for _ in range(min(10, m * (m-1) // 2)):
                    i = random.randint(0, m - 1)
                    j = random.randint(0, m - 1)
                    if i == j:
                        continue
                    a = verts_c[i]
                    b = verts_c[j]
                    if adj1[a, b]:
                        continue
                    D = adj1[a] & adj1[b] & C
                    D[a] = False
                    D[b] = False
                    if D.sum() < 2:
                        continue
                    verts_d = np.where(D)[0]
                    if len(verts_d) < 2:
                        continue
                    x = random.randint(0, len(verts_d) - 1)
                    y = random.randint(0, len(verts_d) - 1)
                    if x == y:
                        continue
                    c = verts_d[x]
                    d = verts_d[y]
                    if adj1[c, d]:
                        continue
                    # Found a violation! Flip an edge in it
                    # Candidates: edges (a,c), (a,d), (b,c), (b,d) that are in color 1
                    candidates = []
                    for pair in [(a,c), (a,d), (b,c), (b,d)]:
                        p1, p2 = sorted(pair)
                        if adj1[p1, p2]:
                            candidates.append(edge_idx[(p1, p2)])
                    if candidates:
                        idx_flip = random.choice(candidates)
                        found = True
                        break
                if not found:
                    idx_flip = random.randint(0, len(colors) - 1)
        else:
            idx_flip = random.randint(0, len(colors) - 1)
        
        # Flip the edge
        i, j = None, None
        # Find which edge idx_flip corresponds to
        # We need to track this. Let's precompute a reverse mapping.
        # Actually, let's just iterate to find it, or precompute.
        
        # Precompute: idx -> (i, j)
        # We'll do this once outside the loop
        # For now, let's use a different approach: store the (i,j) for each idx
        # This is O(n^2) memory but n <= 31 so it's fine
        
        # Actually, let's restructure: precompute edge_list
        # edge_list[idx] = (i, j)
        
        # Let me refactor to precompute this
        break  # Placeholder, will fix below
    
    # Final check
    adj1_final = np.zeros((n, n), dtype=np.bool_)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1_final[i, j] = True
                adj1_final[j, i] = True
            idx += 1
    adj0_final = ~adj1_final
    np.fill_diagonal(adj0_final, False)
    v1 = _count_k222_numpy(adj1_final, n)
    v0 = _count_k222_numpy(adj0_final, n)
    return v1 + v0 == 0

def _local_search(colors, n, max_iters=5000, time_limit=50):
    """Local search to find valid coloring using violation-guided flips."""
    start = time.time()
    
    # Check initial
    adj1, adj0 = _build_adj_masks(colors, n)
    if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
        return True
    
    # Precompute edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Rebuild adjacency
        adj1, adj0 = _build_adj_masks(colors, n)
        
        # Check if valid
        if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
            return True
        
        # Find a violating edge and flip it
        # Strategy: try to find an edge involved in a K2,2,2 violation
        flipped = False
        attempts = 0
        max_attempts = min(50, max_iters - it)
        
        while not flipped and attempts < max_attempts:
            attempts += 1
            
            # Randomly pick a non-edge in color 1 (potential P1 pair)
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj1[u] & (1 << v)):
                continue
            
            C = adj1[u] & adj1[v]
            if C.bit_count() < 4:
                continue
            
            vertices_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                vertices_in_c.append(v_idx)
                mask ^= lsb
            
            m = len(vertices_in_c)
            if m < 4:
                continue
            
            # Try to find a violation
            for i in range(m):
                a = vertices_in_c[i]
                for j in range(i + 1, m):
                    b = vertices_in_c[j]
                    if adj1[a] & (1 << b):
                        continue
                    
                    neighbors_ab_in_c = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
                    if neighbors_ab_in_c.bit_count() < 2:
                        continue
                    
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                        
                    k = len(verts_ab)
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj1[c] & (1 << d)):
                                # Found a K2,2,2 violation in color 1!
                                # Flip one of the edges in this violation
                                # Options: flip edge (u,v) to break P1, or flip an edge within P2/P3
                                # Flip a random edge from the violation
                                candidates = []
                                # Edge (u,v) - but it's a non-edge in color 1, so flipping makes it color 1
                                # Actually we want to break the violation
                                # Flip edge (a,c) - if it exists in color 1
                                if adj1[a] & (1 << c):
                                    key = (a, c) if a < c else (c, a)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[a] & (1 << d):
                                    key = (a, d) if a < d else (d, a)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[b] & (1 << c):
                                    key = (b, c) if b < c else (c, b)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[b] & (1 << d):
                                    key = (b, d) if b < d else (d, b)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                
                                if candidates:
                                    idx_to_flip = random.choice(candidates)
                                    colors[idx_to_flip] = 1 - colors[idx_to_flip]
                                    flipped = True
                                    break
                    if flipped:
                        break
                if flipped:
                    break
            if flipped:
                break
        
        # If no violation found in color 1, check color 0
        if not flipped:
            for u in range(n):
                for v in range(u + 1, n):
                    if adj0[u] & (1 << v):
                        continue
                    
                    C = adj0[u] & adj0[v]
                    if C.bit_count() < 4:
                        continue
                    
                    vertices_in_c = []
                    mask = C
                    while mask:
                        lsb = mask & -mask
                        v_idx = lsb.bit_length() - 1
                        vertices_in_c.append(v_idx)
                        mask ^= lsb
            
                    m = len(vertices_in_c)
                    if m < 4:
                        continue
                    
                    for i in range(m):
                        a = vertices_in_c[i]
                        for j in range(i + 1, m):
                            b = vertices_in_c[j]
                            if adj0[a] & (1 << b):
                                continue
                            
                            neighbors_ab_in_c = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
                            if neighbors_ab_in_c.bit_count() < 2:
                                continue
                            
                            verts_ab = []
                            mask2 = neighbors_ab_in_c
                            while mask2:
                                lsb = mask2 & -mask2
                                v_idx = lsb.bit_length() - 1
                                verts_ab.append(v_idx)
                                mask2 ^= lsb
                                
                            k = len(verts_ab)
                            for x in range(k):
                                c = verts_ab[x]
                                for y in range(x + 1, k):
                                    d = verts_ab[y]
                                    if not (adj0[c] & (1 << d)):
                                        # Found violation in color 0
                                        # Flip an edge that's color 0 to break it
                                        candidates = []
                                        if not (adj1[a] & (1 << c)):
                                            key = (a, c) if a < c else (c, a)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[a] & (1 << d)):
                                            key = (a, d) if a < d else (d, a)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[b] & (1 << c)):
                                            key = (b, c) if b < c else (c, b)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[b] & (1 << d)):
                                            key = (b, d) if b < d else (d, b)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        
                                        if candidates:
                                            idx_to_flip = random.choice(candidates)
                                            colors[idx_to_flip] = 1 - colors[idx_to_flip]
                                            flipped = True
                                            break
                            if flipped:
                                break
                        if flipped:
                            break
                    if flipped:
                        break
                if flipped:
                    break
        
        # If still no flip, do a random flip
        if not flipped:
            idx_to_flip = random.randint(0, len(colors) - 1)
            colors[idx_to_flip] = 1 - colors[idx_to_flip]
    
    # Final check
    adj1, adj0 = _build_adj_masks(colors, n)
    return not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n)

def _fast_local_search(colors, n, time_limit=50):
    """
    Fast local search with incremental delta evaluation.
    Uses bitmask adjacency and computes violation delta per flip in O(n^4).
    """
    start_time = time.time()
    n_edges = n * (n - 1) // 2
    edge_table, edge_list = _edge_index_table(n)
    
    # Build bitmask adjacency for color 1
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    # Initial violation count
    v1 = _count_k222_masks(adj1, n)
    v0 = _count_k222_masks(adj0, n)
    cur_total = v1 + v0
    
    if cur_total == 0:
        return True
    
    best_total = cur_total
    best_colors = colors[:]
    best_v1 = v1
    best_v0 = v0
    
    # Simulated annealing with incremental updates
    temp_start = 2.0
    temp_end = 0.01
    max_iters = 200000
    it = 0
    
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        progress = it / max_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Choose edge to flip
        idx_flip = random.randint(0, n_edges - 1)
        i, j = edge_list[idx_flip]
        old_val = colors[idx_flip]
        
        # Compute delta
        if old_val == 0:
            # Flipping 0->1: add to color 1, remove from color 0
            delta = _delta_violations_flip1(adj1, adj0, i, j, n)
        else:
            # Flipping 1->0: remove from color 1, add to color 0
            delta = _delta_violations_flip0(adj1, adj0, i, j, n)
        
        new_total = cur_total + delta
        
        # Accept/reject
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            # Apply flip
            colors[idx_flip] = 1 - old_val
            if old_val == 0:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
                adj0[i] &= ~(1 << j)
                adj0[j] &= ~(1 << i)
            else:
                adj1[i] &= ~(1 << j)
                adj1[j] &= ~(1 << i)
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            cur_total = new_total
            
            if cur_total < best_total:
                best_total = cur_total
                best_colors = colors[:]
                # Recompute v1, v0 for tracking (expensive but rare)
                best_v1 = _count_k222_masks(adj1, n)
                best_v0 = _count_k222_masks(adj0, n)
                
                if best_total == 0:
                    colors[:] = best_colors
                    return True
        it += 1
    
    # Use best found
    colors[:] = best_colors
    return best_total == 0

def construct():
    deadline = time.time() + 135  # Safety margin (150s limit, leave 15s buffer)
    
    # Try Paley constructions for n=29
    n_target = 29
    p_target = 29
    colors = _paley(n_target, p_target)
    if colors:
        adj1, adj0 = _build_adj_masks(colors, n_target)
        if not _has_k222_mask(adj1, n_target) and not _has_k222_mask(adj0, n_target):
            return {'n': n_target, 'colors': colors}
        else:
            remaining = deadline - time.time()
            if remaining > 15:
                if _fast_local_search(colors, n_target, time_limit=int(remaining - 10)):
                    return {'n': n_target, 'colors': colors}

    # Try other primes for n=29
    primes = [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    for p in primes:
        if time.time() > deadline - 20:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        adj1, adj0 = _build_adj_masks(colors, 29)
        if not _has_k222_mask(adj1, 29) and not _has_k222_mask(adj0, 29):
            return {'n': 29, 'colors': colors}
        remaining = deadline - time.time()
        if remaining > 15:
            if _fast_local_search(colors, 29, time_limit=int(remaining - 10)):
                return {'n': 29, 'colors': colors}

    # Try random initializations for n=29 with multiple restarts
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 12:
            # Use shorter time per restart to allow more restarts
            t_per = min(int(remaining - 10), 8)
            if t_per >= 3:
                if _fast_local_search(colors, 29, time_limit=t_per):
                    return {'n': 29, 'colors': colors}

    # Fallback: Try to find valid coloring for smaller n
    for n in range(28, 16, -1):
        if time.time() > deadline - 10:
            break
        p = n
        while p < 100:
            is_prime = True
            for ii in range(2, int(p**0.5) + 1):
                if p % ii == 0:
                    is_prime = False
                    break
            if is_prime and p % 4 == 1:
                break
            p += 1
            
        colors = _paley(n, p)
        if colors:
            adj1, adj0 = _build_adj_masks(colors, n)
            if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
                return {'n': n, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 10:
                if _fast_local_search(colors, n, time_limit=int(remaining - 5)):
                    return {'n': n, 'colors': colors}
                
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        remaining = deadline - time.time()
        if remaining > 10:
            if _fast_local_search(colors, n, time_limit=int(remaining - 5)):
                return {'n': n, 'colors': colors}

    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
