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
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    max_cliques = []
    
    def popcount(m):
        return bin(m).count('1')
    
    def expand(R_mask, P_mask, X_mask):
        if P_mask == 0 and X_mask == 0:
            if R_mask != 0:
                verts = []
                m = R_mask
                while m:
                    lsb = m & (-m)
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                if len(verts) >= 2:
                    max_cliques.append(frozenset(verts))
            return
        
        # Pivot selection: choose u from P ∪ X that maximizes |P ∩ N(u)|
        union_mask = P_mask | X_mask
        if union_mask == 0:
            return
        
        # Find pivot with maximum |P ∩ N(u)|
        best_pivot = -1
        best_count = -1
        m = union_mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            count = popcount(P_mask & adj_mask[u])
            if count > best_count:
                best_count = count
                best_pivot = u
            m ^= lsb
        
        # Only branch on vertices in P \ N(pivot)
        candidates = P_mask & ~adj_mask[best_pivot]
        
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
            m = candidates & P_mask  # recompute from updated P_mask
    
    expand(0, (1 << n) - 1, 0)
    return max_cliques

def _greedy_transversal(cliques, n, rng):
    """Greedy set cover with weighted scoring."""
    if not cliques:
        return set()
    
    cliques_list = list(cliques)
    num_cliques = len(cliques_list)
    
    # For each vertex, count uncovered cliques it hits
    vertex_to_cliques = {v: [] for v in range(n)}
    for i, c in enumerate(cliques_list):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    uncovered = set(range(num_cliques))
    cover = set()
    
    while uncovered:
        best_v = -1
        best_score = -1
        for v in range(n):
            if v in cover:
                continue
            count = 0
            for ci in vertex_to_cliques[v]:
                if ci in uncovered:
                    count += 1
            if count > 0:
                # Score: number of cliques covered, with small random tiebreak
                score = count + rng.random() * 0.001
                if score > best_score:
                    best_score = score
                    best_v = v
        
        if best_v == -1:
            break
        cover.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            if ci in uncovered:
                uncovered.discard(ci)
    
    return cover

def _local_search(cover, cliques, n, rng, max_iters=20000):
    """Improved local search with efficient validity and richer moves."""
    cover = set(cover)
    cliques_list = list(cliques)
    num_cliques = len(cliques_list)
    
    if num_cliques == 0:
        return cover
    
    # Precompute: for each vertex, which clique indices it hits
    vertex_hits = [set() for _ in range(n)]
    for i, c in enumerate(cliques_list):
        for v in c:
            vertex_hits[v].add(i)
    
    # For each clique, which vertices in cover hit it
    # We'll maintain this dynamically
    
    def compute_cover_cliques(cov):
        """Return dict: clique_idx -> set of vertices in cov that hit it"""
        cc = {}
        for v in cov:
            for ci in vertex_hits[v]:
                if ci not in cc:
                    cc[ci] = set()
                cc[ci].add(v)
        return cc
    
    def is_valid(cov):
        for ci in range(num_cliques):
            if not any(v in cov for v in cliques_list[ci]):
                return False
        return True
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Compute which cliques each cover vertex is essential for
        cover_cliques = compute_cover_cliques(cover)
        
        # Try removal: for each vertex in cover, check if removing it keeps validity
        for v in list(cover):
            # Check if v is the only cover vertex for any clique
            essential = False
            for ci in vertex_hits[v]:
                if ci in cover_cliques and cover_cliques[ci] == {v}:
                    essential = True
                    break
            if not essential:
                cover.discard(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one vertex v, add one vertex u
        for v in list(cover):
            # Find cliques that v is the only cover for
            only_v_cliques = set()
            for ci in vertex_hits[v]:
                if ci in cover_cliques and cover_cliques[ci] == {v}:
                    only_v_cliques.add(ci)
            
            if not only_v_cliques:
                continue
            
            # Try adding a new vertex u not in cover that covers all only_v_cliques
            best_u = -1
            best_gain = 0
            for u in range(n):
                if u in cover:
                    continue
                # How many of only_v_cliques does u cover?
                covered_by_u = only_v_cliques & vertex_hits[u]
                if covered_by_u == only_v_cliques:
                    # u can replace v
                    # Check if this leads to a smaller cover (it won't, same size)
                    # But it might enable further removals
                    # Score by how many additional cliques u covers
                    extra = len(vertex_hits[u] & set(cover_cliques.keys())) - len(only_v_cliques)
                    if extra > best_gain:
                        best_gain = extra
                        best_u = u
            
            if best_u != -1:
                cover.discard(v)
                cover.add(best_u)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-for-1: remove two vertices v1, v2, add one vertex u
        cover_list = list(cover)
        for i in range(len(cover_list)):
            for j in range(i + 1, len(cover_list)):
                v1, v2 = cover_list[i], cover_list[j]
                # Find cliques that are only covered by v1 or v2
                only_cliques = set()
                for ci in range(num_cliques):
                    c = cliques_list[ci]
                    # Check which cover vertices hit this clique
                    hit_by = set()
                    for v in cover:
                        if v in c:
                            hit_by.add(v)
                    if hit_by <= {v1, v2} and hit_by:
                        # This clique is only covered by v1 and/or v2
                        only_cliques.add(ci)
                
                if not only_cliques:
                    # Both can be removed
                    cover.discard(v1)
                    cover.discard(v2)
                    improved = True
                    break
                
                # Try to find u that covers all only_cliques
                best_u = -1
                for u in range(n):
                    if u in cover:
                        continue
                    if only_cliques <= vertex_hits[u]:
                        best_u = u
                        break
                
                if best_u != -1:
                    cover.discard(v1)
                    cover.discard(v2)
                    cover.add(best_u)
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            continue
        
        # Try 1-for-2: remove one vertex v, add two vertices u1, u2
        for v in list(cover):
            only_v_cliques = set()
            for ci in vertex_hits[v]:
                if ci in cover_cliques and cover_cliques[ci] == {v}:
                    only_v_cliques.add(ci)
            
            if not only_v_cliques:
                continue
            
            # Try pairs (u1, u2) not in cover
            candidates = [u for u in range(n) if u not in cover]
            for idx1 in range(len(candidates)):
                u1 = candidates[idx1]
                remaining = only_v_cliques - vertex_hits[u1]
                if not remaining:
                    continue  # u1 alone covers all, try swap instead
                for idx2 in range(idx1 + 1, len(candidates)):
                    u2 = candidates[idx2]
                    if remaining <= vertex_hits[u2]:
                        # u1 and u2 together cover all only_v_cliques
                        # But we're adding 2 and removing 1, so cover grows by 1
                        # Only do this if it enables further improvements
                        # For now, skip 1-for-2 as it increases size
                        pass
        
        # Try random perturbation to escape local optima
        if rng.random() < 0.05 and len(cover) > 0:
            # Remove a random vertex and try to repair
            v = rng.choice(list(cover))
            cover.discard(v)
            # Try to add back vertices to restore validity
            for _ in range(10):
                if is_valid(cover):
                    break
                # Find an uncovered clique
                uncovered_cliques = [ci for ci in range(num_cliques) 
                                     if not any(v2 in cover for v2 in cliques_list[ci])]
                if not uncovered_cliques:
                    break
                ci = rng.choice(uncovered_cliques)
                c = cliques_list[ci]
                candidates = [u for u in c if u not in cover]
                if candidates:
                    u = rng.choice(candidates)
                    cover.add(u)
            # If we couldn't repair, undo
            if not is_valid(cover):
                cover.add(v)
                # Try to repair properly
                for _ in range(20):
                    if is_valid(cover):
                        break
                    uncovered_cliques = [ci for ci in range(num_cliques) 
                                         if not any(v2 in cover for v2 in cliques_list[ci])]
                    if not uncovered_cliques:
                        break
                    ci = rng.choice(uncovered_cliques)
                    c = cliques_list[ci]
                    candidates = [u for u in c if u not in cover]
                    if candidates:
                        u = rng.choice(candidates)
                        cover.add(u)
                if not is_valid(cover):
                    # Last resort: add all vertices in uncovered cliques
                    for ci in range(num_cliques):
                        c = cliques_list[ci]
                        if not any(v2 in cover for v2 in c):
                            cover.update(c)
    
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
        
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Greedy with slight randomization
            cliques_list = list(cliques)
            vertex_to_cliques = {v: [] for v in range(n)}
            for i, c in enumerate(cliques_list):
                for v in c:
                    vertex_to_cliques[v].append(i)
            
            uncovered = set(range(len(cliques_list)))
            cover = set()
            
            while uncovered:
                best_v = -1
                best_score = -1
                for v in range(n):
                    if v in cover:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > 0:
                        # Add random tie-breaking
                        score = count + rng.random() * 0.01
                        if score > best_score:
                            best_score = score
                            best_v = v
                
                if best_v == -1:
                    break
                cover.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    if ci in uncovered:
                        uncovered.discard(ci)
            
            # Local search
            cover = _local_search(cover, cliques, n, rng, max_iters=5000)
            
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
