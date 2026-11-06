import time
import random
from itertools import combinations

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj):
    n = len(adj)
    # Convert to bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    max_cliques = []
    
    def expand(R_mask, P_mask, X_mask):
        if P_mask == 0 and X_mask == 0:
            if R_mask != 0:
                # Extract vertices
                verts = []
                m = R_mask
                while m:
                    lsb = m & (-m)
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                if len(verts) >= 2:
                    max_cliques.append(frozenset(verts))
            return
        if P_mask == 0:
            return
        # Choose pivot: vertex in P ∪ X with max neighbors in P
        # Branch only on vertices in P not adjacent to pivot
        PX = P_mask | X_mask
        best_v = -1
        best_count = -1
        m = PX
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            cnt = (P_mask & adj_mask[v]).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_v = v
            m ^= lsb
        
        # Candidates: vertices in P not adjacent to best_v
        if best_v >= 0:
            candidates = P_mask & ~adj_mask[best_v]
        else:
            candidates = P_mask
        
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            v_bit = 1 << v
            new_R = R_mask | v_bit
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            expand(new_R, new_P, new_X)
            P_mask &= ~v_bit
            X_mask |= v_bit
            m &= ~(1 << v)
    
    expand(0, (1 << n) - 1, 0)
    return max_cliques

def _greedy_transversal(cliques, n, rng):
    """Greedy set cover: repeatedly pick vertex in most uncovered cliques."""
    if not cliques:
        return set()
    
    cliques_list = list(cliques)
    num_cliques = len(cliques_list)
    
    # For each vertex, bitmask of cliques it covers
    vertex_clique_mask = [0] * n
    for i, c in enumerate(cliques_list):
        bit = 1 << i
        for v in c:
            vertex_clique_mask[v] |= bit
    
    uncovered = (1 << num_cliques) - 1
    cover = set()
    
    while uncovered:
        best_v = -1
        best_count = -1
        best_rnd = 2.0
        for v in range(n):
            if v in cover:
                continue
            count = (vertex_clique_mask[v] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_rnd = rng.random()
                best_v = v
            elif count == best_count and count > 0:
                r = rng.random()
                if r < best_rnd:
                    best_rnd = r
                    best_v = v
        if best_v == -1 or best_count == 0:
            break
        cover.add(best_v)
        uncovered &= ~vertex_clique_mask[best_v]
    
    return cover

def _local_search(cover, cliques, n, rng, max_iters=5000):
    """Try to improve the transversal via removal, 1-1 swap, 2-1, and 2-2 moves."""
    cover = set(cover)
    cliques_list = list(cliques)
    num_cliques = len(cliques_list)
    
    if num_cliques == 0:
        return cover
    
    # Precompute for each vertex which cliques it hits (as set)
    vertex_hits = [set() for _ in range(n)]
    for i, c in enumerate(cliques_list):
        for v in c:
            vertex_hits[v].add(i)
    
    # For each clique, which vertices are in it (as set)
    clique_verts = [set(c) for c in cliques_list]
    
    # Precompute clique bitmask for fast validity check
    clique_mask = [0] * num_cliques
    for i, c in enumerate(cliques_list):
        m = 0
        for v in c:
            m |= (1 << v)
        clique_mask[i] = m
    
    def is_valid(cov_mask):
        for i in range(num_cliques):
            if not (clique_mask[i] & cov_mask):
                return False
        return True
    
    cover_mask = 0
    for v in cover:
        cover_mask |= (1 << v)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try removal: for each vertex in cover, check if removing it keeps validity
        for v in sorted(cover):
            new_mask = cover_mask & ~(1 << v)
            if is_valid(new_mask):
                cover.discard(v)
                cover_mask = new_mask
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-1 swap: remove v, add u (same size, may enable future removals)
        for v in sorted(cover):
            # Cliques where v is the only cover
            only_v = set()
            for ci in vertex_hits[v]:
                if not (clique_mask[ci] & (cover_mask & ~(1 << v))):
                    only_v.add(ci)
            
            if not only_v:
                continue
            
            for u in range(n):
                if u in cover:
                    continue
                if only_v <= vertex_hits[u]:
                    cover.discard(v)
                    cover.add(u)
                    cover_mask = (cover_mask & ~(1 << v)) | (1 << u)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-1 move: remove two vertices, add one
        cover_list = sorted(cover)
        for i in range(len(cover_list)):
            for j in range(i + 1, len(cover_list)):
                v1, v2 = cover_list[i], cover_list[j]
                remaining_mask = cover_mask & ~(1 << v1) & ~(1 << v2)
                # Cliques that need v1 or v2 (not covered by remaining)
                needed = set()
                for ci in vertex_hits[v1]:
                    if not (clique_mask[ci] & remaining_mask):
                        needed.add(ci)
                for ci in vertex_hits[v2]:
                    if not (clique_mask[ci] & remaining_mask):
                        needed.add(ci)
                
                if not needed:
                    cover.discard(v1)
                    cover.discard(v2)
                    cover_mask = remaining_mask
                    improved = True
                    break
                
                for u in range(n):
                    if u in cover:
                        continue
                    if needed <= vertex_hits[u]:
                        new_mask = remaining_mask | (1 << u)
                        if is_valid(new_mask):
                            cover.discard(v1)
                            cover.discard(v2)
                            cover.add(u)
                            cover_mask = new_mask
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-2 move: remove two vertices, add two different vertices
        # This can escape local optima where no single move improves
        cover_list = sorted(cover)
        for i in range(len(cover_list)):
            for j in range(i + 1, len(cover_list)):
                v1, v2 = cover_list[i], cover_list[j]
                remaining_mask = cover_mask & ~(1 << v1) & ~(1 << v2)
                # Cliques that need v1 or v2
                needed = set()
                for ci in vertex_hits[v1]:
                    if not (clique_mask[ci] & remaining_mask):
                        needed.add(ci)
                for ci in vertex_hits[v2]:
                    if not (clique_mask[ci] & remaining_mask):
                        needed.add(ci)
                
                if not needed:
                    cover.discard(v1)
                    cover.discard(v2)
                    cover_mask = remaining_mask
                    improved = True
                    break
                
                # Try to cover needed with two vertices u, w
                # Only try if needed is small enough
                if len(needed) > 8:
                    continue
                
                for u in range(n):
                    if u in cover:
                        continue
                    # Cliques still uncovered after adding u
                    still_needed = needed - vertex_hits[u]
                    if not still_needed:
                        # One vertex suffices (would have been caught by 2-1)
                        new_mask = remaining_mask | (1 << u)
                        if is_valid(new_mask):
                            cover.discard(v1)
                            cover.discard(v2)
                            cover.add(u)
                            cover_mask = new_mask
                            improved = True
                            break
                    elif len(still_needed) <= 6:
                        for w in range(n):
                            if w in cover or w == u:
                                continue
                            if still_needed <= vertex_hits[w]:
                                new_mask = remaining_mask | (1 << u) | (1 << w)
                                if is_valid(new_mask):
                                    cover.discard(v1)
                                    cover.discard(v2)
                                    cover.add(u)
                                    cover.add(w)
                                    cover_mask = new_mask
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                break
    
    return cover

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        adj = _generate_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        # Multiple greedy restarts with randomization
        best_cover = None
        best_size = n + 1
        rng = random.Random(seed + 1000)
        
        # Scale restarts and local search iterations with problem size
        if n <= 30:
            num_restarts = 30
            ls_iters = 5000
        elif n <= 35:
            num_restarts = 30
            ls_iters = 4500
        else:
            num_restarts = 35
            ls_iters = 4000
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Fast greedy with bitmask and random tie-breaking
            num_cliques = len(cliques)
            vertex_clique_mask = [0] * n
            for i, c in enumerate(cliques):
                bit = 1 << i
                for v in c:
                    vertex_clique_mask[v] |= bit
            
            uncovered = (1 << num_cliques) - 1
            cover = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                best_rnd = 2.0
                for v in range(n):
                    if v in cover:
                        continue
                    count = (vertex_clique_mask[v] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_rnd = rng.random()
                        best_v = v
                    elif count == best_count and count > 0:
                        r2 = rng.random()
                        if r2 < best_rnd:
                            best_rnd = r2
                            best_v = v
                if best_v == -1 or best_count == 0:
                    break
                cover.add(best_v)
                uncovered &= ~vertex_clique_mask[best_v]
            
            # Local search
            cover = _local_search(cover, cliques, n, rng, max_iters=ls_iters)
            
            if len(cover) < best_size:
                best_size = len(cover)
                best_cover = set(cover)
            
            # Early exit if we found a small cover
            if best_size <= 3:
                break
        
        if best_cover is None:
            # Fallback
            best_cover = set(range(n))
        
        results.append(sorted(best_cover))
    
    return results
