"""K4-free, no independent 6-set graph maximizer.
Strategy: Modular composition (blow-up / lexicographic product).
1) Build small valid "block" graphs (n=16..20) via targeted local search.
2) Build small valid "meta" graphs (n=3..5) that are K4-free and alpha_6-free.
3) Combine via lexicographic product to get large valid graphs.
4) Also try direct construction at larger n as fallback.
"""
import time
import random

DEADLINE = 138  # safety margin before 150s kill

def _build_adj(n, edge_set):
    adj = [0] * n
    for (u, v) in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                if mask:
                    return True
    return False

def _has_independent_6(adj, n):
    if n < 6:
        return False
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
    # Look for K6 in complement
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                    continue
                for l in range(k + 1, n):
                    if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                        continue
                    common_c = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_c & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                    if not mask:
                        continue
                    # Need 2 more vertices from mask that are adjacent in complement
                    bits = []
                    m = mask
                    while m:
                        b = m & (-m)
                        bits.append(b.bit_length() - 1)
                        m ^= b
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            va, vb = bits[a], bits[b]
                            if (comp[va] >> vb) & 1:
                                return True
    return False

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_independent_6(adj, n)

def _validate_graph(graph):
    n = graph['n']
    if n < 1 or n > 40:
        return False
    adj = _build_adj(n, graph['edges'])
    return _is_valid(adj, n)

def _paley_graph(p):
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    edge_set = set()
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edge_set.add((i, j))
    return {'n': p, 'edges': sorted(edge_set)}

def _clebsch_graph():
    n = 16
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            xor = i ^ j
            w = bin(xor).count('1')
            if w == 1 or w == 4:
                edge_set.add((i, j))
    return {'n': n, 'edges': sorted(edge_set)}

def _local_search_block(n, time_limit, seed_offset=0):
    """Find a valid K4-free, no-alpha-6 graph on n vertices via local search."""
    t0 = time.time()
    rng = random.Random(seed_offset)
    
    # Try multiple random starts
    for attempt in range(50):
        if time.time() - t0 > time_limit:
            break
        
        # Initialize with random graph at moderate density
        edge_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.35:
                    edge_set.add((i, j))
        
        adj = _build_adj(n, edge_set)
        
        # Iteratively fix violations
        for step in range(500):
            if time.time() - t0 > time_limit:
                break
            
            if _is_valid(adj, n):
                return {'n': n, 'edges': sorted(edge_set)}
            
            # Find a K4 and break it
            fixed = False
            for i in range(n):
                if time.time() - t0 > time_limit:
                    break
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        continue
                    for k in range(j + 1, n):
                        if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                            continue
                        common = adj[i] & adj[j] & adj[k]
                        mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                        if mask:
                            # Remove a random edge from this K4
                            m = mask
                            verts = []
                            while m:
                                b = m & (-m)
                                verts.append(b.bit_length() - 1)
                                m ^= b
                            v4 = rng.choice(verts)
                            # Pick which edge to remove
                            choices = [(i,j), (i,k), (j,k), (i,v4), (j,v4), (k,v4)]
                            u, v = rng.choice(choices)
                            if u > v:
                                u, v = v, u
                            edge_set.discard((u, v))
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            fixed = True
                            break
                    if fixed:
                        break
                if fixed:
                    break
            
            if fixed:
                continue
            
            # No K4 found; check for independent 6 and fix
            comp = [((1 << n) - 1) ^ adj[x] ^ (1 << x) for x in range(n)]
            fixed = False
            for i in range(n):
                if time.time() - t0 > time_limit:
                    break
                for j in range(i + 1, n):
                    if not (comp[i] >> j) & 1:
                        continue
                    for k in range(j + 1, n):
                        if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                            continue
                        for l in range(k + 1, n):
                            if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                                continue
                            common_c = comp[i] & comp[j] & comp[k] & comp[l]
                            mask = common_c & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                            if not mask:
                                continue
                            bits = []
                            m = mask
                            while m:
                                b = m & (-m)
                                bits.append(b.bit_length() - 1)
                                m ^= b
                            if len(bits) >= 2:
                                a = rng.choice(bits)
                                b = rng.choice(bits)
                                if a == b:
                                    a = bits[(bits.index(a) + 1) % len(bits)]
                                if a > b:
                                    a, b = b, a
                                edge_set.add((a, b))
                                adj[a] |= (1 << b)
                                adj[b] |= (1 << a)
                                fixed = True
                            break
                        if fixed:
                            break
                    if fixed:
                        break
                if fixed:
                    break
            
            if not fixed:
                # Random perturbation
                for _ in range(3):
                    a = rng.randint(0, n - 1)
                    b = rng.randint(0, n - 1)
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    if (a, b) in edge_set:
                        edge_set.discard((a, b))
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    else:
                        edge_set.add((a, b))
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
    
    return None

def _build_meta_graphs():
    """Return list of small valid meta-graphs (K4-free, no alpha-6)."""
    metas = []
    
    # n=3: path P3 (2 edges) - K4-free, alpha=2
    metas.append({'n': 3, 'edges': [(0,1), (1,2)]})
    # n=3: triangle K3 - K4-free, alpha=1
    metas.append({'n': 3, 'edges': [(0,1), (1,2), (0,2)]})
    # n=4: C4 - K4-free, alpha=2
    metas.append({'n': 4, 'edges': [(0,1), (1,2), (2,3), (3,0)]})
    # n=4: P4 - K4-free, alpha=2
    metas.append({'n': 4, 'edges': [(0,1), (1,2), (2,3)]})
    # n=4: K4 minus one edge - K4-free, alpha=1
    metas.append({'n': 4, 'edges': [(0,1), (0,2), (0,3), (1,2), (1,3)]})
    # n=5: C5 - K4-free, alpha=2
    metas.append({'n': 5, 'edges': [(0,1), (1,2), (2,3), (3,4), (4,0)]})
    # n=5: house graph - K4-free, alpha=2
    metas.append({'n': 5, 'edges': [(0,1), (1,2), (2,3), (3,4), (4,0), (1,3)]})
    # n=5: K5 minus matching - need to check
    # n=5: wheel W5 (C4 + center) - K4-free? Center connects to all 4, C4 has no K3, so max clique is 3. Alpha: independent set in C4 is 2, plus center? No, center adjacent to all. So alpha=2.
    metas.append({'n': 5, 'edges': [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]})
    
    # Validate all
    valid_metas = []
    for m in metas:
        if _validate_graph(m):
            valid_metas.append(m)
    
    return valid_metas

def _blowup(block, meta):
    """Lexicographic product: G[block, meta].
    Vertices: (b, m) where b in block, m in meta.
    Edge: (b1,m1)-(b2,m2) iff b1=b2 and m1-m2 in meta, OR b1-b2 in block.
    """
    nb = block['n']
    nm = meta['n']
    n = nb * nm
    block_edges = set(tuple(e) for e in block['edges'])
    meta_edges = set(tuple(e) for e in meta['edges'])
    
    edge_set = set()
    for b1 in range(nb):
        for b2 in range(nb):
            if b1 == b2:
                continue
            if (b1, b2) in block_edges:
                # All cross edges between fibers
                for m1 in range(nm):
                    for m2 in range(nm):
                        u = b1 * nm + m1
                        v = b2 * nm + m2
                        if u < v:
                            edge_set.add((u, v))
            # If b1-b2 NOT in block, no cross edges
        # Within fiber b1: use meta graph
        for m1 in range(nm):
            for m2 in range(m1 + 1, nm):
                if (m1, m2) in meta_edges:
                    u = b1 * nm + m1
                    v = b1 * nm + m2
                    edge_set.add((u, v))
    
    return {'n': n, 'edges': sorted(edge_set)}

def _direct_local_search(n, time_limit, seed_offset=0):
    """Direct local search for a valid graph on exactly n vertices."""
    t0 = time.time()
    rng = random.Random(seed_offset + 1000)
    
    for attempt in range(30):
        if time.time() - t0 > time_limit:
            break
        
        # Start from a structured initial graph
        edge_set = set()
        # Use a grid-like or random structure
        density = 0.3
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    edge_set.add((i, j))
        
        adj = _build_adj(n, edge_set)
        
        for step in range(300):
            if time.time() - t0 > time_limit:
                break
            
            if _is_valid(adj, n):
                return {'n': n, 'edges': sorted(edge_set)}
            
            # Fix K4
            fixed = False
            for i in range(n):
                if time.time() - t0 > time_limit:
                    break
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        continue
                    for k in range(j + 1, n):
                        if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                            continue
                        common = adj[i] & adj[j] & adj[k]
                        mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                        if mask:
                            m = mask
                            verts = []
                            while m:
                                b = m & (-m)
                                verts.append(b.bit_length() - 1)
                                m ^= b
                            v4 = rng.choice(verts)
                            choices = [(i,j), (i,k), (j,k), (i,v4), (j,v4), (k,v4)]
                            u, v = rng.choice(choices)
                            if u > v:
                                u, v = v, u
                            edge_set.discard((u, v))
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            fixed = True
                            break
                    if fixed:
                        break
                if fixed:
                    break
            
            if fixed:
                continue
            
            # Fix independent 6
            if n >= 6:
                comp = [((1 << n) - 1) ^ adj[x] ^ (1 << x) for x in range(n)]
                fixed = False
                for i in range(n):
                    if time.time() - t0 > time_limit:
                        break
                    for j in range(i + 1, n):
                        if not (comp[i] >> j) & 1:
                            continue
                        for k in range(j + 1, n):
                            if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                                continue
                            for l in range(k + 1, n):
                                if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                                    continue
                                common_c = comp[i] & comp[j] & comp[k] & comp[l]
                                mask = common_c & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                                if not mask:
                                    continue
                                bits = []
                                m = mask
                                while m:
                                    b = m & (-m)
                                    bits.append(b.bit_length() - 1)
                                    m ^= b
                                if len(bits) >= 2:
                                    a = rng.choice(bits)
                                    b = rng.choice(bits)
                                    if a == b:
                                        a = bits[(bits.index(a) + 1) % len(bits)]
                                    if a > b:
                                        a, b = b, a
                                    edge_set.add((a, b))
                                    adj[a] |= (1 << b)
                                    adj[b] |= (1 << a)
                                    fixed = True
                                break
                            if fixed:
                                break
                        if fixed:
                            break
                    if fixed:
                        break
                
                if not fixed:
                    for _ in range(3):
                        a = rng.randint(0, n - 1)
                        b = rng.randint(0, n - 1)
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        if (a, b) in edge_set:
                            edge_set.discard((a, b))
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                        else:
                            edge_set.add((a, b))
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
    
    return None

def construct():
    t0 = time.time()
    best_graph = None
    best_n = 0
    
    def _try_update(g):
        nonlocal best_graph, best_n
        if g is not None and _validate_graph(g):
            if g['n'] > best_n:
                best_n = g['n']
                best_graph = g
    
    # Phase 1: Try known small constructions as blocks
    blocks = []
    for p in [17, 29]:
        g = _paley_graph(p)
        if g is not None and _validate_graph(g):
            blocks.append(g)
    
    clebsch = _clebsch_graph()
    if _validate_graph(clebsch):
        blocks.append(clebsch)
    
    # Also try to build blocks via local search at n=16,18,20
    for bn in [16, 18, 20]:
        if time.time() - t0 > DEADLINE - 60:
            break
        remaining = DEADLINE - 60 - (time.time() - t0)
        if remaining < 5:
            break
        g = _local_search_block(bn, remaining / 3, seed_offset=bn)
        if g is not None:
            blocks.append(g)
    
    # Phase 2: Build meta graphs and combine via blow-up
    meta_graphs = _build_meta_graphs()
    
    for block in blocks:
        if time.time() - t0 > DEADLINE - 30:
            break
        for meta in meta_graphs:
            if time.time() - t0 > DEADLINE - 30:
                break
            combined_n = block['n'] * meta['n']
            if combined_n > 40:
                continue
            if combined_n <= best_n:
                continue
            combined = _blowup(block, meta)
            _try_update(combined)
    
    # Phase 3: Direct local search at larger n
    for target_n in [30, 32, 34, 35, 36, 37, 38]:
        if time.time() - t0 > DEADLINE - 15:
            break
        if target_n <= best_n:
            continue
        remaining = DEADLINE - 15 - (time.time() - t0)
        if remaining < 10:
            break
        g = _direct_local_search(target_n, remaining, seed_offset=target_n)
        _try_update(g)
    
    # Phase 4: If we have a good graph, try to grow by adding vertices
    if best_graph is not None and time.time() - t0 < DEADLINE - 10:
        n = best_n
        edge_set = set(tuple(e) for e in best_graph['edges'])
        remaining = DEADLINE - 10 - (time.time() - t0)
        
        for new_n in range(n + 1, min(n + 8, 41)):
            if time.time() - t0 > DEADLINE - 5:
                break
            adj = _build_adj(n, edge_set)
            success = False
            for trial in range(200):
                if time.time() - t0 > DEADLINE - 5:
                    break
                # Try random connection pattern for new vertex
                new_adj = list(adj) + [0]
                for j in range(n):
                    if random.random() < 0.4:
                        new_adj[new_n - 1] |= (1 << j)
                        new_adj[j] |= (1 << (new_n - 1))
                
                if _is_valid(new_adj, new_n):
                    for j in range(n):
                        if (new_adj[new_n - 1] >> j) & 1:
                            edge_set.add((j, new_n - 1))
                    n = new_n
                    best_n = new_n
                    success = True
                    break
            
            if not success:
                break
            
            best_graph = {'n': n, 'edges': sorted(edge_set)}
    
    if best_graph is None:
        # Fallback: return a small valid graph
        # C5 is K4-free and alpha=2
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}
    
    return best_graph
