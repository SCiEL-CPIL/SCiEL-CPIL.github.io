import random
import time
import sys

sys.setrecursionlimit(10000)

def _time_up(deadline):
    return time.time() > deadline

def _has_triangle(edges_set, u, v):
    """Check if adding edge (u,v) creates a triangle."""
    pass

def _max_independent_set_bound(adj, n):
    pass

def _check_triangle_free(adj, n):
    for i in range(n):
        for j in adj[i]:
            if j < i:
                continue
            if adj[i] & adj[j]:
                return False
    return True

def _exact_alpha(adj, n, deadline):
    """
    Returns the size of the maximum independent set.
    Uses branch and bound.
    """
    # Sort vertices by degree descending for better pruning
    # But we need to map back to original indices or just work with sorted list
    # Let's use a recursive function that operates on a list of candidate vertices
    # and keeps track of the current independent set size.
    
    # To speed up, we can use bitmasks if n <= 64.
    # n=40 fits in 64 bits.
    # Let's use integers as bitmasks.
    
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
        
    best = [0]
    
    def dfs(candidates, current_size, current_mask):
        if _time_up(deadline):
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound: current_size + popcount(candidates)
        # If this bound <= best[0], prune
        if current_size + bin(candidates).count('1') <= best[0]:
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the induced subgraph?
        # Or just pick the first one.
        # Let's pick a vertex v from candidates.
        # To improve pruning, we can pick a vertex that maximizes the number of neighbors in candidates.
        
        # Simple heuristic: pick vertex with max degree in candidates
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = bin(candidates & masks[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        v = best_v
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Option 1: Include v
        # New candidates: candidates & ~masks[v] & ~bit_v (remove v and its neighbors)
        new_cands_1 = candidates & ~masks[v] & ~bit_v
        dfs(new_cands_1, current_size + 1, current_mask | bit_v)
        
        # Option 2: Exclude v
        # New candidates: candidates & ~bit_v
        new_cands_2 = candidates & ~bit_v
        dfs(new_cands_2, current_size, current_mask)
        
    all_mask = (1 << n) - 1
    dfs(all_mask, 0, 0)
    
    return best[0]

def construct():
    deadline = time.time() + 115.0 # Safety margin (150s limit, 120s hard stop maybe? "KILLS after 150s")
    best_n = 0
    best_edges = []
    
    # Try to find a graph on 40 vertices first?
    # R(3,10) <= 41, so a graph on 40 vertices would prove R(3,10) > 40, which is a major improvement over 39.
    # However, finding one is hard.
    # The prompt says "survey records 39 as the largest known order".
    # So 39 is the target. 40 is the absolute max possible if it exists.
    # Let's try 40 briefly, then 39.
    
    for target_n in [40, 39]:
        if _time_up(deadline):
            break
            
        found = False
        # Limit restarts to allow more time for exact check if needed
        # Or just run until deadline
        restart = 0
        while not _time_up(deadline):
            restart += 1
            
            # Initialize random graph
            n = target_n
            adj = [set() for _ in range(n)]
            
            # Randomly add edges
            # For triangle-free, we might want a specific structure or just random and fix.
            # Random graph G(n, p) is likely to have triangles if p is not small.
            # For n=39, to be triangle-free, p should be small.
            # Max edges in triangle-free graph on n vertices is n^2/4 (Mantel's theorem).
            # For n=39, max edges is ~380. Total possible ~750.
            # So p ~ 0.5 is too high.
            # Let's use p = 0.3 or so?
            # Or just generate random and remove triangles.
            
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.4:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Phase 1: Make triangle-free
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if _time_up(deadline):
                        break
                    for j in list(adj[i]):
                        if j < i: continue
                        common = adj[i] & adj[j]
                        if common:
                            # Remove edge (i,j)
                            adj[i].remove(j)
                            adj[j].remove(i)
                            changed = True
                if _time_up(deadline):
                    break
            
            if _time_up(deadline):
                break

            # Verify triangle-free (safety)
            if not _check_triangle_free(adj, n):
                continue

            # Check independence number
            # If alpha < 10, it's valid.
            # Use exact branch and bound.
            # For n=40, this might be slow.
            # We can add a quick heuristic check first to reject obvious invalid graphs.
            
            # Heuristic: Greedy
            best_heur = 0
            for _ in range(10):
                vertices = list(range(n))
                random.shuffle(vertices)
                ind_set = set()
                for v in vertices:
                    if not (ind_set & adj[v]):
                        ind_set.add(v)
                if len(ind_set) > best_heur:
                    best_heur = len(ind_set)
            
            if best_heur >= 10:
                continue # Likely invalid, skip exact check to save time
                
            # Exact check
            alpha = _exact_alpha(adj, n, deadline)
            
            if alpha < 10:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
                
        if found:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to a known valid graph if possible?
        # The previous fallback was n=26.
        # We should try to return a valid graph even if it's smaller.
        # But if we failed to find 39, returning 26 is safe.
        # Can we construct a specific triangle-free graph with small alpha?
        # C5 blow-ups?
        # For n=39, if we can't find random one, maybe a structured one?
        # But for now, return the best found or fallback.
        
        # Let's try a quick deterministic construction for fallback if time permits?
        # No, just return what we have.
        # If best_n is 0, we return the old fallback.
        n = 26
        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
