import time
import random
import numpy as np

def _bitcount(x):
    return bin(x).count('1')

def _is_c4_free(n, adj):
    """Check if graph is C4-free using bitsets."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            if common & (common - 1):  # More than one bit set
                return False
    return True

def _creates_c4(u, v, adj, n):
    """Check if adding edge (u,v) creates a C4.
    C4: u-a-b-v where a in N(u), b in N(v), and (a,b) is an edge.
    Equivalent to: exists a in N(u)\{v} such that a is adjacent to some b in N(v)\{u}.
    """
    nu = adj[u] & ~(1 << v)
    nv = adj[v] & ~(1 << u)
    # For each a in N(u)\{v}, check if a has any neighbor in N(v)\{u}
    # This is: (N(a) & nv) != 0 for some a in nu
    a_mask = nu
    while a_mask:
        a_bit = a_mask & (-a_mask)
        a = a_bit.bit_length() - 1
        a_mask ^= a_bit
        if adj[a] & nv:
            return True
    return False

def _min_degree_from_bitsets(n, adj):
    min_deg = n
    for u in range(n):
        d = _bitcount(adj[u])
        if d < min_deg:
            min_deg = d
    return min_deg

def _try_add_edge(u, v, adj, n):
    """Add edge (u,v) if it doesn't create C4. Returns True if added."""
    if u == v:
        return False
    if (adj[u] >> v) & 1:
        return False
    if _creates_c4(u, v, adj, n):
        return False
    adj[u] |= 1 << v
    adj[v] |= 1 << u
    return True

def _try_remove_edge(u, v, adj, n):
    """Remove edge (u,v) if it exists. Returns True if removed."""
    if not ((adj[u] >> v) & 1):
        return False
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    return True

def _construct_paley(q):
    """Construct Paley graph over F_q (q = p^k, p prime).
    Edge (a,b) if (b-a) is a quadratic residue mod q.
    """
    # For prime p, compute quadratic residues
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    adj = [0] * q
    for a in range(q):
        for b in range(a + 1, q):
            if (b - a) % q in residues:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
    return adj

def _construct_polar_graph(q):
    """Construct a C4-free graph using a polar graph construction.
    Use the parabolic quadric Q^+(3,q) or similar.
    For simplicity, use a known construction: the friendship graph variant
    or use Paley graph as seed.
    """
    # Fall back to Paley
    return _construct_paley(q)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        found = False
        
        # Strategy 1: Use Paley graph as seed if target_n is prime
        if target_n in [41, 43, 47]:
            if time.time() > deadline:
                break
            adj_seed = _construct_paley(target_n)
            md = _min_degree_from_bitsets(target_n, adj_seed)
            if md >= min_deg_needed and _is_c4_free(target_n, adj_seed):
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (adj_seed[u] >> v) & 1:
                            edges.append([u, v])
                best_n = target_n
                best_edges = edges
                found = True
                break
        
        # Strategy 2: Random greedy + local search
        for seed in range(3000):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            
            n = target_n
            adj = [0] * n
            deg = [0] * n
            
            # Start with a random graph with moderate density
            p = 0.25
            
            all_edges = list(range(n) * (n - 1) // 2)  # placeholder
            import itertools
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if random.random() < p:
                    if _try_add_edge(u, v, adj, n):
                        deg[u] += 1
                        deg[v] += 1
            
            # Local search with edge swaps to balance degrees
            for iteration in range(1000):
                if time.time() > deadline:
                    break
                
                # Check if we've reached target
                min_deg = _min_degree_from_bitsets(n, adj)
                if min_deg >= min_deg_needed:
                    # Verify C4-free
                    if _is_c4_free(n, adj):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if (adj[u] >> v) & 1:
                                    edges.append([u, v])
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        found = True
                        break
                
                # Find vertex with minimum degree
                min_deg_vertex = 0
                min_deg_val = n
                max_deg_vertex = 0
                max_deg_val = 0
                for u in range(n):
                    d = deg[u]
                    if d < min_deg_val:
                        min_deg_val = d
                        min_deg_vertex = u
                    if d > max_deg_val:
                        max_deg_val = d
                        max_deg_vertex = u
                
                u = min_deg_vertex
                v = max_deg_vertex
                
                if min_deg_val >= min_deg_needed:
                    pass
                elif max_deg_val - min_deg_val < 2:
                    # Degrees are balanced but too low. Try to add edges to u.
                    added = False
                    candidates = [w for w in range(n) if w != u and not ((adj[u] >> w) & 1)]
                    random.shuffle(candidates)
                    for w in candidates:
                        if _try_add_edge(u, w, adj, n):
                            deg[u] += 1
                            deg[w] += 1
                            added = True
                            break
                    if not added:
                        break
                else:
                    # Try to swap edge (v, w) to (u, w)
                    swapped = False
                    w_candidates = []
                    mask = adj[v]
                    while mask:
                        w_bit = mask & (-mask)
                        w = w_bit.bit_length() - 1
                        mask ^= w_bit
                        if w == u:
                            continue
                        if not ((adj[u] >> w) & 1):
                            w_candidates.append(w)
                    
                    random.shuffle(w_candidates)
                    
                    for w in w_candidates:
                        # Check if v can afford to lose this edge
                        if deg[v] - 1 < min_deg_needed:
                            continue
                        
                        # Check if adding (u, w) is safe
                        if not _creates_c4(u, w, adj, n):
                            # Perform swap
                            _try_remove_edge(v, w, adj, n)
                            deg[v] -= 1
                            deg[w] -= 1
                            
                            _try_add_edge(u, w, adj, n)
                            deg[u] += 1
                            deg[w] += 1
                            
                            swapped = True
                            break
                    
                    if not swapped:
                        # Try adding edges to u directly if swap failed
                        added = False
                        candidates = [w for w in range(n) if w != u and not ((adj[u] >> w) & 1)]
                        random.shuffle(candidates)
                        for w in candidates:
                            if _try_add_edge(u, w, adj, n):
                                deg[u] += 1
                                deg[w] += 1
                                added = True
                                break
                        if not added:
                            break
            
            if found:
                break
        
        if found:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
