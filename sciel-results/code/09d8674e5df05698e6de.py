import time
import random
import itertools

N = 16
VERTS = list(range(N))
SUBSETS = list(itertools.combinations(VERTS, 8))
SUBSET_MASKS = []
for s in SUBSETS:
    m = 0
    for v in s:
        m |= (1 << v)
    SUBSET_MASKS.append(m)

PARTITIONS = []
for s in SUBSETS:
    parts = []
    s_set = set(s)
    seen = set()
    for a in itertools.combinations(s, 4):
        b = tuple(sorted(s_set - set(a)))
        key = (tuple(sorted(a)), tuple(sorted(b)))
        if key not in seen:
            seen.add(key)
            parts.append((a, b))
    PARTITIONS.append(parts)

MATCHINGS = []
for perm in itertools.permutations(range(4)):
    mask = 0
    for i, j in enumerate(perm):
        mask |= (1 << (i * 4 + j))
    MATCHINGS.append(mask)

def has_q3(adj):
    for i in range(len(SUBSETS)):
        mask_s = SUBSET_MASKS[i]
        s = SUBSETS[i]
        edges = 0
        for v in s:
            edges += (adj[v] & mask_s).bit_count()
        edges //= 2
        if edges < 12:
            continue
        
        for a, b in PARTITIONS[i]:
            miss = 0
            for i_a, u in enumerate(a):
                for i_b, v in enumerate(b):
                    if not (adj[u] >> v) & 1:
                        miss |= (1 << (i_a * 4 + i_b))
            
            for m in MATCHINGS:
                if (miss & m) == m:
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_count = len(best_edges)
    
    # Try random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random init
        adj = [0] * N
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < 0.6:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Greedy cleanup
        # If has_q3 is too slow, this might hang.
        # We assume it's fast enough for a few iterations.
        
        # SA
        T = 1.0
        for step in range(5000):
            if time.time() > deadline:
                break
            
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v: continue
            
            is_present = (adj[u] >> v) & 1
            if is_present:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                if [u, v] in edges: edges.remove([u, v])
                elif [v, u] in edges: edges.remove([v, u])
                delta = -1
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                delta = 1
            
            if has_q3(adj):
                # Revert
                if is_present:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if [u, v] not in edges: edges.append([u, v])
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    if [u, v] in edges: edges.remove([u, v])
            else:
                if delta > 0 or random.random() < 0.1:
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = [e[:] for e in edges]
                else:
                    # Revert if not accepted
                    if is_present:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if [u, v] not in edges: edges.append([u, v])
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        if [u, v] in edges: edges.remove([u, v])
            
            T *= 0.999

    return best_edges
