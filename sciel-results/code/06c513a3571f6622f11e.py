"""K4-free, no independent 6-set graph maximizer.
Strategy: Constraint-driven greedy construction + monotone K4-repair.
1) Start from a known valid graph (try to construct n=35 via greedy).
2) Greedy vertex addition: for each new vertex, choose connections to maximize slack.
3) Monotone repair: only remove edges from K4s (preserves no-independent-6).
4) Try to extend to n=36 with the same process.
"""
import time
import random

DEADLINE = 140  # safety margin before 150s kill


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


def _find_k4s(adj, n, max_count=100):
    """Return list of K4 vertex tuples."""
    k4s = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                m = mask
                while m:
                    b = m & (-m)
                    v = b.bit_length() - 1
                    k4s.append((i, j, k, v))
                    if len(k4s) >= max_count:
                        return k4s
                    m ^= b
    return k4s


def _has_independent_6(adj, n):
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
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
                    common_ijk = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_ijk & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
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


def _validate(adj, n):
    if n < 1 or n > 40:
        return False
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True


def _slack_for_vertex(adj, n, new_v):
    """Compute slack: how 'safe' is adding connections from new_v to existing vertices.
    Returns a score where higher = safer."""
    # For each existing vertex, check if connecting would create a K4
    # A K4 is created if new_v connects to i,j,k where i,j,k form a triangle
    # and all three are connected to new_v
    # Count potential K4 completions
    k4_risk = 0
    ind_risk = 0
    for i in range(n):
        if not (adj[new_v] >> i) & 1:
            continue
        for j in range(i + 1, n):
            if not (adj[new_v] >> j) & 1:
                continue
            if not (adj[i] >> j) & 1:
                continue
            # i,j are connected to new_v and to each other
            # Check if any k connected to all three forms K4
            common = adj[i] & adj[j] & adj[new_v]
            mask = common & ~(1 << i) & ~(1 << j) & ~(1 << new_v)
            k4_risk += mask.bit_count()
    # Independent set risk: if new_v is NOT connected to i,j,k,l,m, and those 5 form an independent set
    # We just count non-neighbors that are mutually non-adjacent
    non_neigh = [i for i in range(n) if not (adj[new_v] >> i) & 1]
    # Check if any 5 of non_neigh form an independent set (in original graph)
    # This is expensive; approximate by checking if non_neigh has a clique of size 5 in complement
    # For simplicity, just count non-neighbors
    ind_risk = len(non_neigh)
    # Slack = want few K4 risks and few independent risks
    # But we need to balance: connecting reduces ind_risk, disconnecting reduces k4_risk
    return -k4_risk * 10 + (n - len(non_neigh))  # Prefer more connections (reduces ind set)


def _greedy_construct(target_n, time_limit, seed=42):
    """Build a graph on target_n vertices using greedy vertex addition."""
    t0 = time.time()
    random.seed(seed)
    n = target_n
    adj = [0] * n
    
    # Start with a small valid seed graph (e.g., 5-clique complement = independent set of 5, which is fine)
    # Actually, start with empty graph and add vertices one by one
    # Better: start with a known small valid graph and grow
    
    # Initialize: vertex 0 has no edges
    # For each new vertex v (1 to n-1), decide its connections
    for v in range(1, n):
        if time.time() - t0 > time_limit:
            break
        # Try multiple random connection patterns, pick the one with best slack
        best_score = -10**9
        best_mask = 0
        num_trials = 50
        for trial in range(num_trials):
            # Random connection pattern with bias toward ~50% edges
            mask = 0
            for i in range(v):
                if random.random() < 0.45:  # Slightly less than half to avoid K4s
                    mask |= (1 << i)
            # Compute slack
            adj[v] = mask
            # Compute how many K4s this creates
            k4_count = 0
            for i in range(v):
                if not (mask >> i) & 1:
                    continue
                for j in range(i + 1, v):
                    if not (mask >> j) & 1:
                        continue
                    if not (adj[i] >> j) & 1:
                        continue
                    common = adj[i] & adj[j] & mask
                    cmask = common & ~(1 << i) & ~(1 << j) & ~(1 << v)
                    k4_count += cmask.bit_count()
            # Compute independent set risk: count non-neighbors
            non_neigh_count = 0
            for i in range(v):
                if not (mask >> i) & 1:
                    non_neigh_count += 1
            # Score: minimize K4s, but also ensure we don't have too many non-neighbors
            score = -k4_count * 100 - non_neigh_count * 2
            if score > best_score:
                best_score = score
                best_mask = mask
        adj[v] = best_mask
        # Also set reverse
        for i in range(v):
            if (best_mask >> i) & 1:
                adj[i] |= (1 << v)
    
    # Now repair: remove edges from K4s
    for iteration in range(200):
        if time.time() - t0 > time_limit:
            break
        k4s = _find_k4s(adj, n, max_count=50)
        if not k4s:
            break
        # Remove an edge from a random K4
        i, j, k, l = random.choice(k4s)
        # Pick a random edge from the K4 to remove
        edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
        u, w = random.choice(edges_in_k4)
        if u > w:
            u, w = w, u
        adj[u] &= ~(1 << w)
        adj[w] &= ~(1 << u)
    
    # Check validity
    if _has_k4(adj, n):
        # Still has K4s, try more aggressive repair
        for iteration in range(500):
            if time.time() - t0 > time_limit:
                break
            k4s = _find_k4s(adj, n, max_count=20)
            if not k4s:
                break
            i, j, k, l = k4s[0]
            edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            u, w = random.choice(edges_in_k4)
            if u > w:
                u, w = w, u
            adj[u] &= ~(1 << w)
            adj[w] &= ~(1 << u)
    
    if _has_independent_6(adj, n):
        return None  # Invalid
    
    if _has_k4(adj, n):
        return None  # Invalid
    
    # Convert to edge list
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edge_set.add((i, j))
    return {'n': n, 'edges': sorted(edge_set)}


def _try_extend(graph, new_n, time_limit, seed=0):
    """Try to extend a valid graph to new_n vertices."""
    t0 = time.time()
    random.seed(seed)
    old_n = graph['n']
    old_edge_set = set(tuple(e) for e in graph['edges'])
    adj = _build_adj(old_n, old_edge_set)
    
    for trial in range(100):
        if time.time() - t0 > time_limit:
            break
        # Create new adj with extra vertices
        new_adj = [0] * new_n
        for i in range(old_n):
            new_adj[i] = adj[i]
        
        # For each new vertex, randomly connect to old vertices
        for v in range(old_n, new_n):
            mask = 0
            for i in range(old_n):
                if random.random() < 0.4:
                    mask |= (1 << i)
            new_adj[v] = mask
            for i in range(old_n):
                if (mask >> i) & 1:
                    new_adj[i] |= (1 << v)
            # Also connect new vertices to each other
            for w in range(v + 1, new_n):
                if random.random() < 0.3:
                    new_adj[v] |= (1 << w)
                    new_adj[w] |= (1 << v)
        
        # Repair K4s (monotone: only remove edges)
        for iteration in range(300):
            if time.time() - t0 > time_limit:
                break
            k4s = _find_k4s(new_adj, new_n, max_count=30)
            if not k4s:
                break
            i, j, k, l = random.choice(k4s)
            edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            u, w = random.choice(edges_in_k4)
            if u > w:
                u, w = w, u
            new_adj[u] &= ~(1 << w)
            new_adj[w] &= ~(1 << u)
        
        if _has_k4(new_adj, new_n) or _has_independent_6(new_adj, new_n):
            continue
        
        # Valid! Build edge set
        edge_set = set()
        for i in range(new_n):
            for j in range(i + 1, new_n):
                if (new_adj[i] >> j) & 1:
                    edge_set.add((i, j))
        return {'n': new_n, 'edges': sorted(edge_set)}
    
    return None


def _paley_graph(p):
    """Paley graph on p vertices where p ≡ 1 mod 4."""
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


def _complement_of(graph):
    n = graph['n']
    existing = set(tuple(e) for e in graph['edges'])
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in existing:
                edge_set.add((i, j))
    return {'n': n, 'edges': sorted(edge_set)}


def _validate_graph(graph):
    n = graph['n']
    adj = _build_adj(n, graph['edges'])
    return _validate(adj, n)


def construct():
    t0 = time.time()
    best_graph = None
    best_n = 0

    # Phase 1: Try deterministic constructions
    candidates = []
    for p in [17, 29, 37]:
        g = _paley_graph(p)
        if g is not None:
            candidates.append(g)
            candidates.append(_complement_of(g))

    for g in candidates:
        if time.time() - t0 > DEADLINE - 60:
            break
        if _validate_graph(g):
            if g['n'] > best_n:
                best_n = g['n']
                best_graph = g

    # Phase 2: Greedy construction for various target sizes
    # Try to build graphs of size 30-36 via greedy method
    for target in [36, 35, 34, 33, 32, 31, 30]:
        if time.time() - t0 > DEADLINE - 30:
            break
        if target <= best_n:
            continue
        # Try multiple seeds
        for seed in range(20):
            if time.time() - t0 > DEADLINE - 20:
                break
            g = _greedy_construct(target, time_limit=3.0, seed=seed)
            if g is not None:
                if g['n'] > best_n:
                    best_n = g['n']
                    best_graph = g
                break  # Got a valid one for this target, move to next

    # Phase 3: Try to extend the best graph
    if best_graph is not None:
        for target in range(best_n + 1, min(best_n + 4, 41)):
            if time.time() - t0 > DEADLINE - 15:
                break
            for seed in range(30):
                if time.time() - t0 > DEADLINE - 10:
                    break
                g = _try_extend(best_graph, target, time_limit=2.0, seed=seed)
                if g is not None:
                    if g['n'] > best_n:
                        best_n = g['n']
                        best_graph = g
                    break

    # Phase 4: Final attempt - greedy from scratch for n=36 with more time
    if best_n < 36 and time.time() - t0 < DEADLINE - 10:
        remaining = DEADLINE - 10 - (time.time() - t0)
        if remaining > 5:
            g = _greedy_construct(36, time_limit=remaining, seed=12345)
            if g is not None:
                if g['n'] > best_n:
                    best_n = g['n']
                    best_graph = g

    if best_graph is None:
        # Fallback: small valid graph
        # 5-vertex cycle: no K4, no independent 6 (only 5 vertices)
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}

    return best_graph
