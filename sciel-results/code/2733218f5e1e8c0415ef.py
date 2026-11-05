import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin

def _deadline():
    return time.time() >= _DEADLINE

def _is_regular_mask(adj, mask):
    """True if induced subgraph on mask is regular."""
    first = -1
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        deg = (adj[v] & mask).bit_count()
        if first == -1:
            first = deg
        elif deg != first:
            return False
    return True

def _find_witness_full(adj, n):
    """Return mask of a regular induced subgraph of size >= 6, or None."""
    # Check k from 6 to n
    # Smaller k are more likely to be regular in random graphs?
    # Actually, larger k are harder to find, but there are more of them.
    # Let's just iterate all.
    for k in range(6, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(adj, mask):
                return mask
    return None

def _find_witness_sample(adj, n, rng, num_samples=50):
    """Return mask of a regular induced subgraph found by sampling, or None."""
    for _ in range(num_samples):
        k = rng.randint(6, min(10, n))
        comb = rng.sample(range(n), k)
        mask = 0
        for v in comb:
            mask |= (1 << v)
        if _is_regular_mask(adj, mask):
            return mask
    return None

def _valid(adj, n):
    return _find_witness_full(adj, n) is None

def _random_graph(n, p=0.5, rng=None):
    rng = rng or random
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _flip_edge(adj, u, v):
    adj[u] ^= (1 << v)
    adj[v] ^= (1 << u)

def _local_search(adj, n, rng, max_iters=2000):
    """Local search to eliminate regular subgraphs."""
    cur = adj[:]
    for it in range(max_iters):
        if _deadline():
            break
        
        # Try to find a witness using sampling
        w = _find_witness_sample(cur, n, rng, 50)
        
        if w is not None:
            # Found a witness, flip an edge inside it
            verts = []
            m = w
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            if len(verts) >= 2:
                u, v = rng.sample(verts, 2)
                _flip_edge(cur, u, v)
        else:
            # No witness found in sample.
            # Check if it's actually valid.
            # To save time, we only do the full check every 10 iterations
            # or if we've done a lot of steps.
            if it % 10 == 0:
                if _valid(cur, n):
                    return cur
                else:
                    # It's not valid, but sample missed it.
                    # Do a full search to find it and repair.
                    w_full = _find_witness_full(cur, n)
                    if w_full is not None:
                        verts = []
                        m = w_full
                        while m:
                            lsb = m & (-m)
                            v = lsb.bit_length() - 1
                            verts.append(v)
                            m ^= lsb
                        if len(verts) >= 2:
                            u, v = rng.sample(verts, 2)
                            _flip_edge(cur, u, v)
    return cur

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    rng = random.Random(12345)
    best_n = 5
    best_adj = [0] * 5
    best_edges = []

    # Try to find valid graphs for n=20, 21, 22
    # Start with 20 as it's known to exist
    for target_n in [20, 21, 22]:
        if _deadline():
            break
        
        found = False
        # Multiple random restarts
        for attempt in range(20):
            if _deadline():
                break
            
            # Random initialization
            # Vary p slightly
            p = rng.uniform(0.45, 0.55)
            adj = _random_graph(target_n, p, rng)
            
            # Local search
            improved = _local_search(adj, target_n, rng, max_iters=1000)
            
            if _valid(improved, target_n):
                best_n = target_n
                best_adj = improved
                best_edges = _edges_from_adj(improved, target_n)
                found = True
                break
        
        if found:
            # If we found a graph for this n, we can try the next larger n
            # The loop continues to the next target_n
            pass
        else:
            # If we didn't find one for this n, we might still try the next?
            # If n=20 fails, n=21 is unlikely to succeed easily.
            # But let's just continue.
            pass

    # If we didn't find anything for 20-22, fallback to smaller n?
    # The problem says "Maximize n".
    # If we only found n=5, return that.
    # But we should try to find n=6,7... if 20 fails?
    # Given the deadline, if 20 fails, we probably don't have time for 6.
    # But n=5 is a valid fallback.
    
    return {'n': best_n, 'edges': best_edges}
