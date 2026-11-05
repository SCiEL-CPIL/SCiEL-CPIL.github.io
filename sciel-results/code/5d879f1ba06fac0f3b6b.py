"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with symmetric connection set S is triangle-free."""
    S_set = set(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            if (b - a) % n in S_set:
                return True
    return False

def _alpha_circulant(n, S):
    """Compute independence number of circulant graph using branch and bound."""
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def dfs(candidates, current_size):
        if current_size + candidates.bit_count() <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Heuristic: pick vertex with fewest neighbors in candidate set
        # This is equivalent to picking vertex with most non-neighbors (comp_mask)
        cands_list = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cands_list.append(v)
            tmp ^= lsb
        
        best_v = -1
        best_score = -1
        for v in cands_list:
            score = (candidates & comp_mask[v]).bit_count()
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        dfs(candidates & comp_mask[v] & ~(1 << v), current_size + 1)
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _build_edges_circulant(n, S):
    edges = []
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _find_valid_circulant(n, deadline):
    """Find a triangle-free circulant graph on n vertices with alpha < 10."""
    max_k = n // 2
    candidates = list(range(1, max_k + 1))
    # Sort candidates to try "dense" connections first? 
    # Actually, let's just iterate.
    
    S = set()
    # We use a recursive search
    # To avoid deep recursion overhead, we can use a stack or just standard recursion with limit
    # Given n<=40, depth <= 20. Recursion is fine.
    
    def backtrack(remaining_candidates):
        if time.time() > deadline:
            return None
        
        # Check if current S is a valid solution
        # We only check alpha if S is "maximal" or if we've added enough edges?
        # Checking alpha is expensive. 
        # Heuristic: Only check alpha if len(S) is large enough to potentially have alpha < 10.
        # For n=40, avg degree needs to be high.
        # If len(S) is small, alpha is likely large.
        # Let's check alpha at every node? No, too slow.
        # Let's check alpha only when we cannot add more candidates (maximal S).
        
        can_add = False
        for k in remaining_candidates:
            # Check if k can be added
            # Conflict if (k - a) % n in S or (a - k) % n in S for any a in S
            # Since S is symmetric, (a-k) = -(k-a). So just check (k-a) % n in S.
            ok = True
            for a in S:
                if (k - a) % n in S:
                    ok = False
                    break
            if ok:
                can_add = True
                break
        
        if not can_add:
            # Maximal triangle-free set
            alpha = _alpha_circulant(n, S)
            if alpha < 10:
                return set(S)
            return None

        # Try adding candidates
        # Order: try candidates that "block" fewest future candidates?
        # Or just iterate.
        for i, k in enumerate(remaining_candidates):
            if time.time() > deadline:
                return None
            
            # Check if k can be added
            ok = True
            for a in S:
                if (k - a) % n in S:
                    ok = False
                    break
            
            if ok:
                S.add(k)
                S.add((-k) % n)
                new_remaining = remaining_candidates[:i] + remaining_candidates[i+1:]
                result = backtrack(new_remaining)
                if result is not None:
                    return result
                S.remove(k)
                S.remove((-k) % n)
        
        return None

    return backtrack(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try from n=40 down
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        
        S = _find_valid_circulant(n, deadline)
        if S is not None:
            best_n = n
            best_edges = _build_edges_circulant(n, S)
            break
    
    # Fallback
    if best_n < 26:
        # Use a simple construction that's guaranteed to work
        n0 = 25
        best_edges = []
        # C5 blow-up or similar
        # Simple: use circulant with S = {1, 4} mod 5 pattern repeated
        # For n=25, S = {1, 4, 6, 9, 11, 14, 16, 19, 21, 24}? 
        # Let's just use a known small one if we fail.
        # The previous fallback was weak. 
        # Let's try n=20 with S={1,2,3}? No, triangle.
        # S={1,3}? 
        # If we fail to find any, we return the best we found or a valid small one.
        # Since R(3,10) is likely > 25, we should find something.
        # If not, we return a valid graph on n=10 (trivial) or similar.
        # But the score will be low.
        # Let's assume we find one for n>=26.
        
        # Fallback: n=10, C5 x C2?
        # Just return empty graph for n=1? No, alpha=1. 
        # We need alpha < 10.
        # A complete graph K_n is triangle-free? No.
        # An empty graph has alpha=n.
        # We need a graph with alpha < 10.
        # For n=9, alpha can be 3 (e.g. 3 disjoint triangles? No, triangles forbidden).
        # Triangle-free graph on 9 vertices.
        # C5 + 4 isolated? Alpha = 1 + 4 = 5 < 10.
        # So n=9 is valid.
        if best_n == 0:
             best_n = 9
             best_edges = []
             # C5 on 0..4
             best_edges.append([0,1])
             best_edges.append([1,2])
             best_edges.append([2,3])
             best_edges.append([3,4])
             best_edges.append([4,0])
             # Vertices 5,6,7,8 isolated.
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
