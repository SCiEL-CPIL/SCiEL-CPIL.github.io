import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _check_c4(adj):
    """adj: list of sets. Return True if no C4."""
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _alpha_greedy_lb(adj):
    """Greedy lower bound on independence number."""
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        best_v = None
        best_deg = 10**9
        for v in remaining:
            d = len(adj[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def _has_indep_11(adj):
    """Check if there is an independent set of size 11."""
    n = len(adj)
    if n < 11:
        return False
    # Simple greedy check first
    if _alpha_greedy_lb(adj) < 11:
        # Greedy says < 11, but might be wrong (greedy is lower bound? No, greedy is a valid independent set, so it's a lower bound on alpha).
        # If greedy finds < 11, alpha could still be >= 11.
        # Wait, greedy finds a maximal independent set. Its size is a lower bound on alpha.
        # If the lower bound is < 11, alpha might be >= 11.
        # If the lower bound is >= 11, alpha is definitely >= 11.
        # So this check is not conclusive for "no independent set of 11".
        pass
    
    # Exact check using branch and bound
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)
    
    sys.setrecursionlimit(10000)
    
    def branch(remaining_mask, current_size):
        if current_size >= 11:
            return True
        if remaining_mask == 0:
            return False
        # Upper bound
        if current_size + bin(remaining_mask).count("1") < 11:
            return False
        
        # Pick vertex with highest degree in remaining
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = bin(remaining_mask & masks[v]).count("1")
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        # Branch include
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        
        # Branch exclude
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        
        return False
    
    full = (1 << n) - 1
    return branch(full, 0)

def _build_polarity_graph_q5():
    """Polarity graph of PG(2,5): 31 vertices, C4-free."""
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, 3, 5)
                    cx, cy, cz = 1, (y*inv) % 5, (z*inv) % 5
                elif y != 0:
                    inv = pow(y, 3, 5)
                    cx, cy, cz = 0, 1, (z*inv) % 5
                else:
                    cx, cy, cz = 0, 0, 1
                key = (cx, cy, cz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    npts = len(points)
    adj = [set() for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            a = points[i]
            b = points[j]
            s = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % 5
            if s == 0:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _local_search(adj, target_n, max_iters=20000):
    n = len(adj)
    adj = [set(s) for s in adj]
    best_adj = [set(s) for s in adj]
    best_n = n
    
    # Try to grow
    for _ in range(target_n - n):
        if _deadline_hit():
            break
        added = False
        for attempt in range(200):
            new_idx = n
            new_neigh = set()
            # Random subset
            for v in range(n):
                if random.random() < 0.3:
                    new_neigh.add(v)
            
            # Check C4 condition
            valid = True
            neigh_list = list(new_neigh)
            for i in range(len(neigh_list)):
                for j in range(i+1, len(neigh_list)):
                    a, b = neigh_list[i], neigh_list[j]
                    if adj[a] & adj[b]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for v in new_neigh:
                    adj[v].add(new_idx)
                adj.append(set(new_neigh))
                n = new_idx + 1
                added = True
                # Check if this improves best
                if n > best_n and not _has_indep_11(adj):
                    best_n = n
                    best_adj = [set(s) for s in adj]
                break
        
    # Try to add edges to reduce alpha
    for it in range(max_iters):
        if _deadline_hit():
            break
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v or v in adj[u]:
            continue
        if len(adj[u] & adj[v]) >= 2:
            continue
        
        adj[u].add(v)
        adj[v].add(u)
        
        if not _has_indep_11(adj):
            if n > best_n:
                best_n = n
                best_adj = [set(s) for s in adj]
        else:
            adj[u].discard(v)
            adj[v].discard(u)
            
    return best_adj

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 118.0
    
    adj = _build_polarity_graph_q5()
    best_adj = _local_search(adj, 39, max_iters=10000)
    
    n = len(best_adj)
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
                
    if not _check_c4(best_adj) or _has_indep_11(best_adj):
        # Fallback to seed
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
