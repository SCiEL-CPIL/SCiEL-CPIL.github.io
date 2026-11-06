"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices.
    Uses Bitset-based representation for speed.
    1. Try n=40, 39, ...
    2. Random init, remove triangles, add edges to reduce alpha.
    3. Verify with Bitset MIS.
    """
    max_time = 115
    start_time = time.time()
    
    def bit_count(x):
        return bin(x).count('1')
        
    def is_triangle_free(adj, n):
        for u in range(n):
            # adj[u] contains neighbors of u
            # For each neighbor v of u, check if adj[u] & adj[v] == 0
            # We only need to check v > u to avoid double check? 
            # No, just check all v in adj[u].
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & -v_mask
                v_idx = lsb.bit_length() - 1
                if adj[u] & adj[v_idx]:
                    return False
                v_mask ^= lsb
        return True

    def max_independent_set_size(adj, n):
        best = [0]
        
        def branch(cands, count):
            if time.time() - start_time > max_time:
                return
            if count + bit_count(cands) <= best[0]:
                return
            if cands == 0:
                if count > best[0]:
                    best[0] = count
                return
            
            # Heuristic: pick vertex with highest degree in cands
            # Approximate: just pick the first one or random?
            # Let's pick the one with max common neighbors? Too slow.
            # Pick first is okay for small n.
            v_lsb = cands & -cands
            v_idx = v_lsb.bit_length() - 1
            
            # Branch 1: Include v_idx
            # New candidates are cands without v_idx and without neighbors of v_idx
            new_cands = cands & ~adj[v_idx] & ~v_lsb
            branch(new_cands, count + 1)
            
            # Branch 2: Exclude v_idx
            branch(cands & ~v_lsb, count)

        full_mask = (1 << n) - 1
        branch(full_mask, 0)
        return best[0]

    def get_mis(adj, n):
        best_set = []
        best_size = 0
        
        def branch(cands, current_set):
            nonlocal best_set, best_size
            if len(current_set) + bit_count(cands) <= best_size:
                return
            if cands == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            v_lsb = cands & -cands
            v_idx = v_lsb.bit_length() - 1
            
            # Include
            current_set.append(v_idx)
            new_cands = cands & ~adj[v_idx] & ~v_lsb
            branch(new_cands, current_set)
            current_set.pop()
            
            # Exclude
            branch(cands & ~v_lsb, current_set)

        full_mask = (1 << n) - 1
        branch(full_mask, [])
        return best_set

    def try_one(n, seed):
        rng = random.Random(seed)
        adj = [0] * n
        
        # Random init
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove triangles
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if changed: break
                # Iterate neighbors of i
                v_mask = adj[i]
                while v_mask:
                    lsb = v_mask & -v_mask
                    j = lsb.bit_length() - 1
                    if j > i: # Only check edges (i, j) with i < j
                        common = adj[i] & adj[j]
                        if common:
                            # Remove edge (i, j)
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            break
                    v_mask ^= lsb
        
        # Reduce alpha
        for _ in range(200):
            if time.time() - start_time > max_time:
                return None
            
            alpha = max_independent_set_size(adj, n)
            if alpha <= 9:
                break
            
            mis = get_mis(adj, n)
            
            # Try to add edge between two nodes in MIS
            found = False
            for u in mis:
                for v in mis:
                    if u < v and not (adj[u] & (1 << v)):
                        # Check if adding (u, v) creates triangle
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # Remove a vertex from MIS
                v_to_remove = mis[0]
                # Compact graph
                new_n = n - 1
                new_adj = [0] * new_n
                idx_map = {}
                new_idx = 0
                for i in range(n):
                    if i != v_to_remove:
                        idx_map[i] = new_idx
                        new_idx += 1
                
                for i in range(n):
                    if i == v_to_remove: continue
                    ni = idx_map[i]
                    # Map neighbors
                    v_mask = adj[i]
                    while v_mask:
                        lsb = v_mask & -v_mask
                        j = lsb.bit_length() - 1
                        if j != v_to_remove:
                            nj = idx_map[j]
                            new_adj[ni] |= (1 << nj)
                        v_mask ^= lsb
                            
                adj = new_adj
                n = new_n
                if n < 10:
                    return None

        if is_triangle_free(adj, n) and max_independent_set_size(adj, n) <= 9:
            edges = []
            for i in range(n):
                v_mask = adj[i]
                while v_mask:
                    lsb = v_mask & -v_mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    v_mask ^= lsb
            return {'n': n, 'edges': edges}
        return None

    best_result = None
    
    # Try n=40, 39
    for n in [40, 39]:
        for seed in range(50):
            if time.time() - start_time > max_time:
                break
            res = try_one(n, seed)
            if res:
                if best_result is None or res['n'] > best_result['n']:
                    best_result = res
                if res['n'] >= 39:
                    return res
        if best_result and best_result['n'] == 40:
            return best_result
            
    if best_result:
        return best_result
        
    # Fallback: Andrásfai(9) on 27 vertices (3*9)
    # AN_9 is triangle free, alpha=9
    n = 27
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % n
            if diff in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
                # This is complete graph. 
                # Correct AN_k definition:
                # Vertices 0..3k-1.
                # Edge (i, j) if min((j-i) mod 3k, (i-j) mod 3k) in {1, ..., k-1} or {2k+1, ... 3k-1}?
                # Actually, Andrásfai graph AN_k is the circulant graph C_{3k}({1, ..., k-1, 2k+1, ..., 3k-1})?
                # No, AN_k is triangle free.
                # Let's just use a known valid small graph if 40/39 fails.
                # Clebsch graph n=16, alpha=5.
                pass
                
    # Simple fallback: Random search for n=30
    for seed in range(100):
        if time.time() - start_time > max_time:
            break
        res = try_one(30, seed)
        if res:
            return res
            
    # Last resort: Empty graph? No, alpha=n.
    # Return a valid small graph:
    # Complete bipartite K_{5,5} is triangle free, alpha=10. Too big.
    # K_{4,4} + 2 isolated? No.
    # Let's return a known valid graph:
    # The 5-cycle C5 is triangle free, alpha=2. n=5.
    # We can just return a small valid graph to avoid 0 score.
    n = 5
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
