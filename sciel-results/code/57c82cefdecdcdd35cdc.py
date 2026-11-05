import time
import random

# EVOLVE-BLOCK-START

_DEADLINE = 0

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > _DEADLINE:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random seeds
        # Increase attempts if n is larger? No, just try many.
        for seed in range(2000):
            if time.time() > _DEADLINE:
                break
            
            result = _try_construct_bitmask(n, required_min_deg, seed)
            if result is not None:
                return {'n': n, 'edges': result}
        
    return {'n': 39, 'edges': []}

def _try_construct_bitmask(n, required_min_deg, seed):
    random.seed(seed)
    adj = [0] * n
    
    for _ in range(1000): # Max steps
        if time.time() > _DEADLINE:
            return None
            
        # Calculate degrees
        degrees = [adj[u].bit_count() for u in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= required_min_deg:
            edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    mask ^= lsb
            return edges
        
        # Find vertices with min degree
        min_vertices = [u for u in range(n) if degrees[u] == min_deg]
        random.shuffle(min_vertices)
        
        added = False
        for u in min_vertices:
            if time.time() > _DEADLINE:
                return None
            
            # Find candidates for u
            mask_u = adj[u]
            candidates = []
            # Optimization: Instead of checking all n, we can just iterate
            # But n is small.
            for v in range(n):
                if v != u and not (mask_u & (1 << v)):
                    candidates.append(v)
            
            if not candidates:
                continue
                
            # Sort by degree
            candidates.sort(key=lambda v: degrees[v])
            
            # Try top few
            for v in candidates[:5]:
                if not _check_c4_bitmask(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            return None
            
    return None

def _check_c4_bitmask(adj, u, v):
    su = adj[u] & ~(1 << v)
    sv = adj[v] & ~(1 << u)
    
    if su.bit_count() < sv.bit_count():
        mask_iter = su
        mask_other = sv
        while mask_iter:
            lsb = mask_iter & -mask_iter
            x = lsb.bit_length() - 1
            mask_iter ^= lsb
            if adj[x] & mask_other:
                return True
    else:
        mask_iter = sv
        mask_other = su
        while mask_iter:
            lsb = mask_iter & -mask_iter
            y = lsb.bit_length() - 1
            mask_iter ^= lsb
            if adj[y] & mask_other:
                return True
    return False

# EVOLVE-BLOCK-END
