"""Rewritten: fast A2-based C6 check + 2-opt local search."""
import time
import random

def _init_bits(n, edges):
    bits = [0] * n
    for u, v in edges:
        bits[u] |= (1 << v)
        bits[v] |= (1 << u)
    return bits

def _compute_A2(bits, n):
    A2 = [0] * n
    for u in range(n):
        nu = bits[u]
        while nu:
            lsb = nu & (-nu)
            v = lsb.bit_length() - 1
            nu ^= lsb
            A2[u] |= bits[v]
        A2[u] &= ~bits[u]
        A2[u] &= ~(1 << u)
    return A2

def _update_A2_add(bits, A2, u, v):
    """After adding edge (u,v), update A2 for all neighbors of u and v."""
    for w in (u, v):
        # A2[w] gains: for each x in N(w), x's neighbors that are now reachable in 2 steps
        # New 2-step paths through u-v edge: N(u) reaches v's neighbors, N(v) reaches u's neighbors
        # A2[x] for x in N(u): gains N(v) (since x-u-v-...)
        # A2[x] for x in N(v): gains N(u)
        # Also A2[u] gains N(v), A2[v] gains N(u)
        pass
    # Simpler: just recompute A2 for affected vertices
    # Vertices affected: u, v, and all neighbors of u and v
    affected = set()
    nu = bits[u]
    while nu:
        lsb = nu & (-nu)
        affected.add(lsb.bit_length() - 1)
        nu ^= lsb
    nv = bits[v]
    while nv:
        lsb = nv & (-nv)
        affected.add(lsb.bit_length() - 1)
        nv ^= lsb
    affected.add(u)
    affected.add(v)
    for w in affected:
        nw = bits[w]
        a2w = 0
        while nw:
            lsb = nw & (-nw)
            x = lsb.bit_length() - 1
            nw ^= lsb
            a2w |= bits[x]
        A2[w] = a2w & ~bits[w] & ~(1 << w)

def _update_A2_remove(bits, A2, u, v):
    """After removing edge (u,v), update A2."""
    affected = set()
    nu = bits[u]
    while nu:
        lsb = nu & (-nu)
        affected.add(lsb.bit_length() - 1)
        nu ^= lsb
    nv = bits[v]
    while nv:
        lsb = nv & (-nv)
        affected.add(lsb.bit_length() - 1)
        nv ^= lsb
    affected.add(u)
    affected.add(v)
    for w in affected:
        nw = bits[w]
        a2w = 0
        while nw:
            lsb = nw & (-nw)
            x = lsb.bit_length() - 1
            nw ^= lsb
            a2w |= bits[x]
        A2[w] = a2w & ~bits[w] & ~(1 << w)

def _creates_c6(bits, A2, u, v):
    """Fast check if adding edge (u,v) creates a C6.
    C6: u-a-b-c-d-v where all distinct.
    Equivalent: exists a in N(u), c in N(v), a != c, with length-3 path a-b-c-d-... 
    Actually: a-b-c-d where a in N(u), d in N(v), a!=c, b,c distinct, all avoiding u,v.
    Simpler: a in N(u), c in N(v), a != c, and there exists a path a-x-c of length 2 
    (i.e., common neighbor of a and c) avoiding u and v.
    Wait, that's only length 3 from a to c: a-x-y-c? No.
    
    Let me reconsider. C6: u-a-b-c-d-v. 
    a in N(u), d in N(v). Path a-b-c-d of length 3, all avoiding u,v.
    So we need: a in N(u) \ {v}, d in N(v) \ {u}, a != d, and a length-3 path from a to d 
    avoiding {u, v, a, d}.
    
    Length-3 path a-b-c-d means: b in N(a), c in N(b) ∩ N(d), b != c, b,c avoiding {u,v,a,d}.
    
    Alternative: use A2. A2[a] = vertices reachable from a in exactly 2 steps.
    A length-3 path a-b-c-d exists iff there's c in N(d) such that c in A2[a] and 
    the intermediate b can be chosen to avoid constraints.
    
    Actually, let me use a different decomposition:
    C6: u-a-b-c-d-v. Fix a in N(u), c in N(v). Then we need a path a-b-c of length 2 
    (a-b, b-c) where b avoids {u,v,a,c}, AND then c-d-v where d avoids {u,v,a,b,c}.
    Wait no, that's only 5 vertices: u,a,b,c,d,v. The path is u-a-b-c-d-v.
    So: a in N(u), d in N(v), and a path a-b-c-d of length 3 avoiding {u,v}.
    
    Let me use: a in N(u)\{v}, d in N(v)\{u}, a!=d, and exists b,c with:
      b in N(a)\{u,v,a}, c in N(b)\{u,v,a,b}, c in N(d)\{u,v,d}, c != a, b != d.
    
    This is getting complex. Let me use the A2 approach differently.
    
    A C6 u-a-b-c-d-v exists iff:
    There exist a in N(u), c in N(v), a != c, such that there's a path of length 3 from a to c
    avoiding {u,v}. A path of length 3 from a to c: a-x-y-c.
    This exists iff there's y in N(c) such that y is in A2[a] (reachable in 2 steps from a) 
    and the path can avoid u,v.
    
    Actually, A2[a] includes y if there's a 2-step path a-x-y. For the C6, we need 
    a-x-y-c where x,y avoid {u,v,a,c} and x!=y.
    
    Let me just use: for each a in N(u)\{v}, for each c in N(v)\{u}, a!=c,
    check if N(a) & N(c) has a vertex b that is not in {u,v,a,c}.
    If so, we have a-b-c, and then we need c-d-v... wait no.
    
    OK let me be very careful. C6 = u-a-b-c-d-v (6 edges, 6 vertices).
    Edges: (u,a), (a,b), (b,c), (c,d), (d,v), (v,u).
    So: a in N(u), b in N(a), c in N(b), d in N(c), v in N(d).
    All 6 vertices distinct: u,a,b,c,d,v all different.
    
    Given u,v: we need a in N(u)\{v}, d in N(v)\{u}, a != d,
    and a path a-b-c-d of length 3 with b,c distinct from {u,v,a,d} and b != c.
    
    Path a-b-c-d: b in N(a), c in N(b) ∩ N(d), b != c, b,c not in {u,v,a,d}.
    
    Using bitsets: for each a in N(u)\{v}, for each d in N(v)\{u}, a!=d:
      candidates_b = N(a) & ~(1<<u) & ~(1<<v) & ~(1<<a) & ~(1<<d)
      For each b in candidates_b:
        if N(b) & N(d) & ~(1<<u) & ~(1<<v) & ~(1<<a) & ~(1<<b) & ~(1<<d): return True
    
    This is O(|N(u)| * |N(v)| * avg_degree) which for 40 vertices with ~5-6 degree is fine.
    """
    nu = bits[u] & ~(1 << v)
    nv = bits[v] & ~(1 << u)
    
    mask_uv = ~(1 << u) & ~(1 << v)
    
    a_set = nu
    while a_set:
        lsb = a_set & (-a_set)
        a = lsb.bit_length() - 1
        a_set ^= lsb
        
        na = bits[a] & mask_uv & ~(1 << a)
        
        d_set = nv & ~(1 << a)
        while d_set:
            lsb2 = d_set & (-d_set)
            d = lsb2.bit_length() - 1
            d_set ^= lsb2
            
            nd = bits[d] & mask_uv & ~(1 << d)
            
            # b in N(a), c in N(b) ∩ N(d), all avoiding u,v,a,d, b!=c
            b_candidates = na & ~(1 << d)
            while b_candidates:
                lsb3 = b_candidates & (-b_candidates)
                b = lsb3.bit_length() - 1
                b_candidates ^= lsb3
                
                # c in N(b) ∩ N(d) avoiding u,v,a,b,d
                c_set = bits[b] & nd & mask_uv & ~(1 << a) & ~(1 << b) & ~(1 << d)
                if c_set:
                    return True
    
    return False

def _has_c6(bits, A2, n):
    """Check if the current graph (with bits) has any C6.
    For each edge (u,v), check if removing it would leave a C6, i.e., 
    if there's a path of length 5 between u and v avoiding u,v.
    Actually simpler: for each edge (u,v), check _creates_c6 on the graph WITHOUT (u,v).
    But that's expensive. Instead, directly enumerate C6s.
    
    For each pair (u,v) that is an edge, check if there's a path u-a-b-c-d-v 
    where a,b,c,d are distinct and not in {u,v}.
    """
    for u in range(n):
        nu = bits[u]
        while nu:
            lsb = nu & (-nu)
            v = lsb.bit_length() - 1
            nu ^= lsb
            if v <= u:
                continue
            # Check if there's a C6 using edge (u,v)
            # i.e., path of length 5 from u to v: u-a-b-c-d-v
            # a in N(u)\{v}, d in N(v)\{u}, a!=d, path a-b-c-d avoiding {u,v}
            mask_uv = ~(1 << u) & ~(1 << v)
            a_set = bits[u] & mask_uv
            while a_set:
                lsa = a_set & (-a_set)
                a = lsa.bit_length() - 1
                a_set ^= lsa
                na = bits[a] & mask_uv & ~(1 << a)
                d_set = bits[v] & mask_uv & ~(1 << a)
                while d_set:
                    lsd = d_set & (-d_set)
                    d = lsd.bit_length() - 1
                    d_set ^= lsd
                    nd = bits[d] & mask_uv & ~(1 << d)
                    b_cand = na & ~(1 << d)
                    while b_cand:
                        lsb3 = b_cand & (-b_cand)
                        b = lsb3.bit_length() - 1
                        b_cand ^= lsb3
                        c_set = bits[b] & nd & mask_uv & ~(1 << a) & ~(1 << b) & ~(1 << d)
                        if c_set:
                            return True
    return False

def _greedy_build(n, deadline, rng, order='random'):
    """Build C6-free graph greedily."""
    bits = [0] * n
    A2 = [0] * n
    edge_set = set()
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edge_set.add((u, v))
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
    A2 = _compute_A2(bits, n)
    
    # Generate candidates
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    
    if order == 'random':
        rng.shuffle(candidates)
    elif order == 'degree':
        # Sort by sum of current degrees (low first)
        candidates.sort(key=lambda uv: bits[uv[0]].bit_count() + bits[uv[1]].bit_count())
    elif order == 'cluster':
        # Prefer edges within same K5 block first, then cross-block
        def block(u):
            return u // 5
        candidates.sort(key=lambda uv: (0 if block(uv[0]) == block(uv[1]) else 1,
                                         bits[uv[0]].bit_count() + bits[uv[1]].bit_count()))
    
    # Multiple passes
    progress = True
    while progress and time.time() < deadline:
        progress = False
        if order == 'degree' or order == 'cluster':
            candidates.sort(key=lambda uv: bits[uv[0]].bit_count() + bits[uv[1]].bit_count())
        
        for u, v in candidates:
            if time.time() > deadline:
                return bits, A2, edge_set
            if (u, v) in edge_set:
                continue
            if not _creates_c6(bits, A2, u, v):
                edge_set.add((u, v))
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
                _update_A2_add(bits, A2, u, v)
                progress = True
    
    return bits, A2, edge_set

def _local_search(bits, A2, edge_set, n, deadline, rng):
    """Local search with 1-swap and 2-opt moves."""
    best_count = len(edge_set)
    best_bits = bits[:]
    best_A2 = A2[:]
    best_edges = edge_set.copy()
    
    no_improve = 0
    max_no_improve = 300
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline and no_improve < max_no_improve:
        move_type = rng.random()
        
        if move_type < 0.6:
            # 1-swap: remove one edge, try to add another
            if not edge_set:
                break
            removed = rng.choice(tuple(edge_set))
            ru, rv = removed
            
            # Temporarily remove
            bits[ru] &= ~(1 << rv)
            bits[rv] &= ~(1 << ru)
            _update_A2_remove(bits, A2, ru, rv)
            edge_set.discard(removed)
            
            added = None
            # Try random non-edges
            for _ in range(300):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                if not _creates_c6(bits, A2, u, v):
                    added = (u, v)
                    break
            
            if added is None:
                # Revert
                edge_set.add(removed)
                bits[ru] |= (1 << rv)
                bits[rv] |= (1 << ru)
                _update_A2_add(bits, A2, ru, rv)
                no_improve += 1
                continue
            
            au, av = added
            edge_set.add(added)
            bits[au] |= (1 << av)
            bits[av] |= (1 << au)
            _update_A2_add(bits, A2, au, av)
            
            new_count = len(edge_set)
            if new_count > best_count:
                best_count = new_count
                best_bits = bits[:]
                best_A2 = A2[:]
                best_edges = edge_set.copy()
                no_improve = 0
            else:
                # Revert
                edge_set.discard(added)
                bits[au] &= ~(1 << av)
                bits[av] &= ~(1 << au)
                _update_A2_remove(bits, A2, au, av)
                edge_set.add(removed)
                bits[ru] |= (1 << rv)
                bits[rv] |= (1 << ru)
                _update_A2_add(bits, A2, ru, rv)
                no_improve += 1
        
        else:
            # 2-opt: remove one edge, try to add two edges
            if not edge_set:
                break
            removed = rng.choice(tuple(edge_set))
            ru, rv = removed
            
            # Temporarily remove
            bits[ru] &= ~(1 << rv)
            bits[rv] &= ~(1 << ru)
            _update_A2_remove(bits, A2, ru, rv)
            edge_set.discard(removed)
            
            added1 = None
            added2 = None
            # Try to find two edges to add
            for _ in range(200):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                if not _creates_c6(bits, A2, u, v):
                    added1 = (u, v)
                    break
            
            if added1 is not None:
                au, av = added1
                edge_set.add(added1)
                bits[au] |= (1 << av)
                bits[av] |= (1 << au)
                _update_A2_add(bits, A2, au, av)
                
                for _ in range(200):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_set:
                        continue
                    if not _creates_c6(bits, A2, u, v):
                        added2 = (u, v)
                        break
            
            if added2 is not None:
                bw, bv = added2
                edge_set.add(added2)
                bits[bw] |= (1 << bv)
                bits[bv] |= (1 << bw)
                _update_A2_add(bits, A2, bw, bv)
                
                new_count = len(edge_set)
                if new_count > best_count:
                    best_count = new_count
                    best_bits = bits[:]
                    best_A2 = A2[:]
                    best_edges = edge_set.copy()
                    no_improve = 0
                else:
                    # Revert both adds and the remove
                    edge_set.discard(added2)
                    bits[bw] &= ~(1 << bv)
                    bits[bv] &= ~(1 << bw)
                    _update_A2_remove(bits, A2, bw, bv)
                    edge_set.discard(added1)
                    bits[au] &= ~(1 << av)
                    bits[av] &= ~(1 << au)
                    _update_A2_remove(bits, A2, au, av)
                    edge_set.add(removed)
                    bits[ru] |= (1 << rv)
                    bits[rv] |= (1 << ru)
                    _update_A2_add(bits, A2, ru, rv)
                    no_improve += 1
            else:
                # Revert added1 and the remove
                edge_set.discard(added1)
                bits[au] &= ~(1 << av)
                bits[av] &= ~(1 << au)
                _update_A2_remove(bits, A2, au, av)
                edge_set.add(removed)
                bits[ru] |= (1 << rv)
                bits[rv] |= (1 << ru)
                _update_A2_add(bits, A2, ru, rv)
                no_improve += 1
    
    return best_bits, best_A2, best_edges, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start with different orderings
    starts = [
        ('random', 42),
        ('degree', 43),
        ('cluster', 44),
        ('random', 100),
        ('degree', 200),
        ('random', 300),
        ('cluster', 400),
        ('random', 500),
        ('degree', 600),
        ('random', 700),
    ]
    
    for order, seed in starts:
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Allocate time: 30% for greedy, 70% for local search
        greedy_deadline = min(deadline - 5, start_time + 20)
        bits, A2, edge_set = _greedy_build(n, greedy_deadline, rng, order)
        
        if len(edge_set) > best_overall:
            best_overall = len(edge_set)
            best_edges_overall = edge_set.copy()
        
        # Local search
        ls_deadline = min(deadline - 3, start_time + 50)
        bits, A2, edge_set, count = _local_search(bits, A2, edge_set, n, ls_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edge_set.copy()
    
    # Final verification
    if best_edges_overall:
        final_bits = _init_bits(n, best_edges_overall)
        final_A2 = _compute_A2(final_bits, n)
        
        if _has_c6(final_bits, final_A2, n):
            # Fallback: remove problematic edges
            # Just return the K5 base
            fallback = set()
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u, v = 5 * k + a, 5 * k + b
                        fallback.add((u, v))
            return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]
