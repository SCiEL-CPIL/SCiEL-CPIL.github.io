"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Map edges to indices for bitmask representation
        edge_to_idx = {}
        for i, e in enumerate(edges):
            edge_to_idx[e] = i
        
        E = len(edges)
        n_I = n - k
        
        # Bitmasks for adjacency
        # N_C[u]: bitmask of neighbors in C for u in C (0..k-1)
        # N_I[v]: bitmask of neighbors in C for v in I (0..n_I-1)
        N_C = [0] * k
        N_I = [0] * n_I
        
        for u, v in edges:
            if u < k and v < k:
                N_C[u] |= (1 << v)
                N_C[v] |= (1 << u)
            elif u < k and v >= k:
                idx_v = v - k
                N_I[idx_v] |= (1 << u)
            elif u >= k and v < k:
                idx_u = u - k
                N_I[idx_u] |= (1 << v)
        
        candidates = []
        
        # 1. Clique C
        mask_C = 0
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in edge_to_idx:
                    mask_C |= (1 << edge_to_idx[(u, v)])
        
        if mask_C:
            candidates.append({
                'mask': mask_C,
                'verts': list(range(k))
            })
        
        # 2. Maximal bicliques
        if k <= 20:
            subsets = range(1, 1 << k)
        else:
            # Generate subsets of size 1, 2, 3
            subsets = []
            for size in range(1, 4):
                for comb in itertools.combinations(range(k), size):
                    mask = 0
                    for u in comb:
                        mask |= (1 << u)
                    subsets.append(mask)
        
        for S_mask in subsets:
            if S_mask == 0:
                continue
            
            # Compute T_mask
            T_mask = 0
            for v_idx in range(n_I):
                if (N_I[v_idx] & S_mask) == S_mask:
                    T_mask |= (1 << v_idx)
            
            if T_mask == 0:
                continue
            
            # Maximality check
            is_maximal = True
            
            # Check C side
            for u in range(k):
                if not (S_mask & (1 << u)):
                    if (N_C[u] & T_mask) == T_mask:
                        is_maximal = False
                        break
            
            if is_maximal:
                # Check I side
                for v_idx in range(n_I):
                    if not (T_mask & (1 << v_idx)):
                        if (N_I[v_idx] & S_mask) == S_mask:
                            is_maximal = False
                            break
            
            if not is_maximal:
                continue
            
            # Compute edge mask
            c_mask = 0
            
            # Vertices in S
            verts_S = [u for u in range(k) if S_mask & (1 << u)]
            
            # Edges in S
            for i in range(len(verts_S)):
                for j in range(i + 1, len(verts_S)):
                    u, v = verts_S[i], verts_S[j]
                    if u > v: u, v = v, u
                    if (u, v) in edge_to_idx:
                        c_mask |= (1 << edge_to_idx[(u, v)])
            
            # Vertices in T
            verts_T = [v_idx + k for v_idx in range(n_I) if T_mask & (1 << v_idx)]
            
            # Edges between S and T
            for u in verts_S:
                for v in verts_T:
                    if u < v:
                        e = (u, v)
                    else:
                        e = (v, u)
                    if e in edge_to_idx:
                        c_mask |= (1 << edge_to_idx[e])
            
            if c_mask:
                candidates.append({
                    'mask': c_mask,
                    'verts': verts_S + verts_T
                })
        
        # Greedy Set Cover
        uncovered = (1 << E) - 1
        partition = []
        
        while uncovered:
            best_idx = -1
            best_count = 0
            
            for i, c in enumerate(candidates):
                count = (uncovered & c['mask']).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                # Fallback: if no candidate covers any edge, we have a problem.
                # But candidates should cover all edges.
                break
            
            partition.append(candidates[best_idx]['verts'])
            uncovered &= ~candidates[best_idx]['mask']
            
        return partition

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        res = solve_instance(*spec)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
