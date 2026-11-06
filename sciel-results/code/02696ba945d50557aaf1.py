import time
import random

_N = 20
_ALL_PAIRS = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_ALL_TRIPLES = []
for _a in range(_N):
    for _b in range(_a + 1, _N):
        for _c in range(_b + 1, _N):
            _ALL_TRIPLES.append((_a, _b, _c))

def _calc_violations(adj):
    """Returns list of triples that have >= 3 common neighbors."""
    viol = []
    for a, b, c in _ALL_TRIPLES:
        common = adj[a] & adj[b] & adj[c]
        if common.bit_count() >= 3:
            viol.append((a, b, c))
    return viol

def _count_edges(adj):
    return sum(a.bit_count() for a in adj) // 2

def construct():
    n = _N
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = None
    best_edges = 0
    
    # Initial best: K_{2,18}
    seed_adj = [0] * n
    for a in (0, 1):
        for v in range(2, n):
            seed_adj[a] |= (1 << v)
            seed_adj[v] |= (1 << a)
    if not _calc_violations(seed_adj):
        best_adj = seed_adj[:]
        best_edges = _count_edges(best_adj)
        
    rng = random.Random(42)
    
    num_restarts = 100
    # We don't know time per restart, so we just loop until deadline
    
    while time.time() < deadline:
        r = int((time.time() - start_time) * 10) # pseudo restart id for seed variation?
        # Better: just use rng
        
        # Strategy for initial state
        mode = rng.randint(0, 2)
        if mode == 0:
            # Dense random
            adj = [0] * n
            p = 0.3 + 0.2 * (rng.random())
            for u, v in _ALL_PAIRS:
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        elif mode == 1:
            # K_{2,18} base + noise
            adj = [0] * n
            for a in (0, 1):
                for v in range(2, n):
                    adj[a] |= (1 << v)
                    adj[v] |= (1 << a)
            for u, v in _ALL_PAIRS:
                if u > 1 and v > 1 and rng.random() < 0.15:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            # Random sparse
            adj = [0] * n
            p = 0.2 + 0.1 * (rng.random())
            for u, v in _ALL_PAIRS:
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        # Phase 1: Greedy Repair
        # Limit repair steps to avoid infinite loops on very dense invalid graphs
        max_repair_steps = 1000
        for step in range(max_repair_steps):
            if time.time() > deadline:
                break
            viol = _calc_violations(adj)
            if not viol:
                break
            
            a, b, c = rng.choice(viol)
            common = adj[a] & adj[b] & adj[c]
            
            # Get common neighbors
            cns = []
            mm = common
            while mm:
                lsb = mm & (-mm)
                k = lsb.bit_length() - 1
                cns.append(k)
                mm ^= lsb
                
            if not cns:
                break # Should not happen if viol is not empty
                
            tri_vertex = rng.choice([a, b, c])
            cnb = rng.choice(cns)
            
            adj[tri_vertex] &= ~(1 << cnb)
            adj[cnb] &= ~(1 << tri_vertex)
            
        # Check if valid
        if not _calc_violations(adj):
            # Phase 2: Local Search (Add Edges)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                pairs = _ALL_PAIRS[:]
                rng.shuffle(pairs)
                for u, v in pairs:
                    if time.time() > deadline:
                        break
                    if not ((adj[u] >> v) & 1):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if not _calc_violations(adj):
                            improved = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            
            cur_edges = _count_edges(adj)
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = adj[:]
                
        # Phase 3: Perturbation
        # Remove 2-5 random edges to escape local optimum
        existing = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
        if existing:
            k = rng.randint(2, 5)
            for _ in range(k):
                if not existing: break
                u, v = rng.choice(existing)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                existing.remove((u, v))

    if best_adj is None:
        best_adj = [0] * n
        for i in range(1, n):
            best_adj[0] |= (1 << i)
            best_adj[i] |= (1 << 0)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
