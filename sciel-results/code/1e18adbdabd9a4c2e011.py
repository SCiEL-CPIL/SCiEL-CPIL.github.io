"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 145
    n = 16

    # Precompute all Q3 embeddings as sets of edge indices
    # Q3 has 8 vertices and 12 edges
    # Q3 edges: (u,v) where u^v is a power of 2
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            d = u ^ v
            if d & (d-1) == 0:  # power of 2
                q3_edges.append((u, v))
    
    # Map Q3 edge positions to indices
    q3_edge_set = set(q3_edges)
    q3_edge_count = len(q3_edges)
    
    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
    
    # Precompute all Q3 embeddings: for each bijection from Q3 vertices to K16 vertices
    # that preserves the Q3 structure, get the set of edge indices
    # This is expensive, so we'll use a smarter approach
    
    # Actually, let's use the incremental approach but with better search strategy
    # Precompute Q3 neighbor masks
    q3_mask = [0] * 8
    for u in range(8):
        for v in range(8):
            if u != v and (u ^ v) & ((u ^ v) - 1) == 0:
                q3_mask[u] |= (1 << v)
    
    # Fast Q3 check for adding edge (u, v) to graph defined by adj_bitmasks
    def has_q3_fast(adj, u, v):
        mask_u = adj[u] & ~(1 << u) & ~(1 << v)
        mask_v = adj[v] & ~(1 << u) & ~(1 << v)
        
        cands_u = []
        m2 = mask_u
        while m2:
            lsb = m2 & -m2
            idx = lsb.bit_length() - 1
            cands_u.append(idx)
            m2 ^= lsb
            
        cands_v = []
        m2 = mask_v
        while m2:
            lsb = m2 & -m2
            idx = lsb.bit_length() - 1
            cands_v.append(idx)
            m2 ^= lsb
        
        for i in range(len(cands_u)):
            c2 = cands_u[i]
            for j in range(i+1, len(cands_u)):
                c4 = cands_u[j]
                common_24 = adj[c2] & adj[c4]
                common_24 &= ~(1 << u) & ~(1 << v) & ~(1 << c2) & ~(1 << c4)
                if common_24 == 0:
                    continue
                
                for k in range(len(cands_v)):
                    c3 = cands_v[k]
                    if c3 == c2 or c3 == c4:
                        continue
                    if not (adj[c2] & (1 << c3)):
                        continue
                    
                    for l in range(k+1, len(cands_v)):
                        c5 = cands_v[l]
                        if c5 == c2 or c5 == c4 or c5 == c3:
                            continue
                        if not (adj[c4] & (1 << c5)):
                            continue
                        
                        common_35 = adj[c3] & adj[c5]
                        common_35 &= ~(1 << u) & ~(1 << v) & ~(1 << c3) & ~(1 << c5)
                        if common_35 == 0:
                            continue
                        
                        c6_candidates = common_24 & ~(1 << c3) & ~(1 << c5)
                        if c6_candidates == 0:
                            continue
                        
                        c7_candidates = common_35 & ~(1 << c2) & ~(1 << c4)
                        if c7_candidates == 0:
                            continue
                        
                        m6 = c6_candidates
                        while m6:
                            lsb = m6 & -m6
                            c6 = lsb.bit_length() - 1
                            if adj[c6] & c7_candidates:
                                return True
                            m6 ^= lsb
        
        return False

    # Build initial graph: K_{3,13}
    def build_initial():
        adj = [0] * n
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.add((a, b))
        return adj, edges

    best_adj, best_edges = build_initial()
    best_count = len(best_edges)
    
    random.seed(42)
    
    # Multiple randomized greedy passes with different orderings
    for trial in range(50):
        if time.time() > deadline - 30:
            break
            
        adj, edges = build_initial()
        
        missing = [e for e in all_edges if e not in edges]
        
        # Different random orderings
        if trial % 3 == 0:
            random.shuffle(missing)
        elif trial % 3 == 1:
            # Sort by some heuristic
            missing.sort(key=lambda e: random.random())
        else:
            # Random with bias toward high-degree vertices
            missing.sort(key=lambda e: -random.random())
        
        for e in missing:
            if time.time() > deadline - 25:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]

    # Local Search: Remove one, try to add one (or more)
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            for e in edge_list:
                if time.time() > deadline - 10:
                    break
                
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                missing = [ne for ne in all_edges if ne not in edges]
                random.shuffle(missing)
                
                added = False
                for ne in missing:
                    if time.time() > deadline - 10:
                        break
                    nu, nv = ne
                    if has_q3_fast(adj, nu, nv):
                        continue
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    added = True
                    
                    # Try to add more edges greedily
                    for ne2 in missing:
                        if time.time() > deadline - 10:
                            break
                        if ne2 in edges:
                            continue
                        nu2, nv2 = ne2
                        if has_q3_fast(adj, nu2, nv2):
                            continue
                        adj[nu2] |= (1 << nv2)
                        adj[nv2] |= (1 << nu2)
                        edges.add(ne2)
                    
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break

    # Try different initial structures
    if time.time() < deadline - 15:
        for k in [2, 4, 5]:
            if time.time() > deadline - 15:
                break
            adj = [0] * n
            edges = set()
            # K_{k, 16-k}
            for a in range(k):
                for b in range(k, 16):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    edges.add((a, b))
            
            missing = [e for e in all_edges if e not in edges]
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline - 15:
                    break
                u, v = e
                if has_q3_fast(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add(e)
            
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = set(edges)
                best_adj = adj[:]

    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
