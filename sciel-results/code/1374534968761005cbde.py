"""Triangle-free graphs without independent 10-sets (Ramsey number R(3,10) > n).

Combines structured circulant graph search with general local search.
The circulant approach exploits the high symmetry to find valid graphs
faster than random search, which is crucial for n=39 and n=40.
"""
import time
import random

DEADLINE = 145.0  # Safety margin before 150s limit

def _has_triangle_bitset(adj, n):
    """Check if graph with adjacency bitsets has a triangle."""
    for i in range(n):
        ai = adj[i]
        # Only check neighbors j > i to avoid double counting, 
        # but for triangle check, we just need common neighbors.
        # If i is connected to j and k, and j is connected to k, it's a triangle.
        # We can check: for each neighbor j of i, does adj[j] share any neighbor with i?
        # Specifically, if adj[i] & adj[j] has any bit set, there is a triangle.
        # But we must ensure we don't count the edge (i,j) itself if we are not careful.
        # Actually, adj[i] contains j. adj[j] contains i. 
        # Common neighbors of i and j are (adj[i] & adj[j]) & ~(1<<i) & ~(1<<j).
        # If this is non-zero, triangle exists.
        j = ai
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            # Check common neighbors of i and jj
            # adj[i] & adj[jj] will include i and jj themselves if we are not careful?
            # No, adj[i] has bit jj. adj[jj] has bit i.
            # So adj[i] & adj[jj] has bits i and jj.
            # We need to mask them out.
            common = (adj[i] & adj[jj]) & ~(1 << i) & ~(1 << jj)
            if common:
                return True
            j ^= lb
    return False

def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set size via branch and bound. Returns size, capped at limit for speed if found."""
    best = [0]
    full_mask = (1 << n) - 1
    
    # Precompute degrees for ordering? Or just use bit_count.
    
    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        
        # Heuristic: pick vertex with max degree in candidates subgraph
        # To speed up, we can just pick the first one, or one with high degree.
        # Let's pick the vertex with the highest degree in the remaining candidates.
        # This is O(n^2) per node, which might be slow. 
        # Alternative: just pick the lowest index. This is faster but less effective pruning.
        # Given the small n (<=40), let's try a middle ground: pick vertex with max popcount(adj[v] & candidates).
        
        max_deg = -1
        pivot = -1
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
            c ^= lb
        
        if pivot == -1:
            return

        v_bit = 1 << pivot
        rest = candidates ^ v_bit
        
        # Branch 1: Exclude pivot
        bnb(chosen_count, rest)
        
        # Branch 2: Include pivot
        # New candidates must be non-neighbors of pivot
        new_cands = rest & ~adj[pivot]
        bnb(chosen_count + 1, new_cands)

    bnb(0, full_mask)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_bitset(adj, n):
        return False
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True

def _search_circulant(n, start_time, max_iter=2000):
    """Search for a valid circulant graph on n vertices."""
    # S is a set of offsets. i~j iff (j-i) mod n in S.
    # Symmetry: if k in S, then n-k in S.
    # We represent S by its minimal positive representatives in 1..n//2.
    max_off = n // 2
    
    def build_adj_from_S(S):
        adj = [0] * n
        for s in S:
            for i in range(n):
                j = (i + s) % n
                adj[i] |= (1 << j)
                # adj[j] |= (1 << i) is implied by symmetry if we add n-s, 
                # but we only store minimal reps. 
                # If s != n-s, we need to add n-s as well to make it undirected.
                # Wait, if S contains s, the edge (i, i+s) exists.
                # The edge (i+s, i) is the same. 
                # In a circulant graph defined by connection set C, 
                # i~j if (j-i) mod n in C.
                # For undirected, C must be symmetric: k in C iff -k in C.
                # So if we pick s in 1..n/2, we must also have n-s in C.
                # So we should add both s and n-s to the adjacency.
                if s != n - s:
                    j2 = (i + (n - s)) % n
                    adj[i] |= (1 << j2)
        # Ensure no self loops (s != 0 is guaranteed)
        for i in range(n):
            adj[i] &= ~(1 << i)
        return adj

    def is_triangle_free_circulant(S):
        # Triangle i, i+a, i+b.
        # Edges: (i, i+a) => a in S_sym.
        # (i+a, i+b) => (b-a) in S_sym.
        # (i, i+b) => b in S_sym.
        # So we need a, b, b-a all in S_sym.
        # S_sym is the set of all offsets.
        S_sym = set()
        for s in S:
            S_sym.add(s)
            S_sym.add((n - s) % n)
        
        for a in S_sym:
            for b in S_sym:
                if a == 0 or b == 0: continue
                c = (b - a) % n
                if c in S_sym:
                    # Check if this forms a non-degenerate triangle
                    # i, i+a, i+b must be distinct.
                    # a != 0, b != 0.
                    # i+a != i+b => a != b.
                    # If a == b, then c=0, which is not in S_sym.
                    # So we just need to check if a, b, c are non-zero and distinct?
                    # Actually, if a=b, c=0, not in S.
                    # If a=0, not in S.
                    # So any a,b in S_sym with (b-a)%n in S_sym forms a triangle.
                    return True
        return False

    def get_alpha_circulant(S):
        adj = build_adj_from_S(S)
        return _alpha_exact_bnb(adj, n, limit=10)

    # Local search on S
    best_S = set()
    best_alpha = 100
    best_adj = None
    best_valid = False

    # Initialize with random S
    for _ in range(50): # 50 restarts
        if time.time() > start_time + DEADLINE - 5:
            break
            
        S = set()
        # Random density ~0.3
        for off in range(1, max_off + 1):
            if random.random() < 0.3:
                S.add(off)
        
        if not S:
            continue
            
        if not is_triangle_free_circulant(S):
            continue
            
        alpha = get_alpha_circulant(S)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_S = set(S)
            best_adj = build_adj_from_S(S)
            if alpha < 10:
                best_valid = True
                # Don't break immediately, maybe we can find lower alpha? 
                # But we only need < 10. So we can return immediately.
                return best_adj
        
        # Local search: flip elements in S
        improved = True
        while improved and time.time() < start_time + DEADLINE - 5:
            improved = False
            # Try adding an offset
            for off in range(1, max_off + 1):
                if off in S:
                    continue
                new_S = S | {off}
                if not is_triangle_free_circulant(new_S):
                    continue
                new_alpha = get_alpha_circulant(new_S)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_S = new_S
                    best_adj = build_adj_from_S(new_S)
                    if new_alpha < 10:
                        return best_adj
                    improved = True
                    break
            # Try removing an offset
            if not improved:
                for off in list(S):
                    if len(S) <= 1: break
                    new_S = S - {off}
                    if not is_triangle_free_circulant(new_S):
                        continue
                    new_alpha = get_alpha_circulant(new_S)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_S = new_S
                        best_adj = build_adj_from_S(new_S)
                        if new_alpha < 10:
                            return best_adj
                        improved = True
                        break

    if best_valid:
        return best_adj
    return None

def _random_local_search(n, start_time):
    """General local search fallback."""
    for trial in range(100):
        if time.time() > start_time + DEADLINE - 5:
            return None
        
        # Generate random triangle-free graph
        adj = [0] * n
        order = list(range(n))
        random.shuffle(order)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if _is_valid(adj, n):
            return adj
        
        # Local search: try to reduce alpha by adding edges
        # We can only add edge (u,v) if it doesn't create a triangle.
        # Adding edges reduces alpha.
        current_alpha = _alpha_exact_bnb(adj, n, limit=10)
        
        improved = True
        while improved and time.time() < start_time + DEADLINE - 5:
            improved = False
            # Shuffle pairs to avoid bias
            random.shuffle(pairs)
            for u, v in pairs:
                if adj[u] & (1 << v):
                    continue
                # Check triangle
                if adj[u] & adj[v]:
                    continue
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                new_alpha = _alpha_exact_bnb(adj, n, limit=10)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < 10:
                        return adj
                    improved = True
                    break
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
    return None

def construct():
    start_time = time.time()
    best_n = 0
    best_adj = None
    
    # Try n from 40 down to 26
    for n in range(40, 25, -1):
        if time.time() > start_time + DEADLINE - 5:
            break
            
        # 1. Try Circulant Search
        adj = _search_circulant(n, start_time)
        if adj is not None:
            best_n = n
            best_adj = adj
            break
            
        # 2. Try Random Local Search (limited time)
        # Give it a small budget, e.g., 2 seconds
        local_start = time.time()
        adj = _random_local_search(n, start_time + 2.0) 
        if adj is not None:
            best_n = n
            best_adj = adj
            break

    # If nothing found, return empty or a small valid graph?
    # The task says 1 <= n <= 41.
    # We should at least return a valid graph if we found one.
    # If we didn't find any, we should try to find a small one to get some score?
    # "Maximize n". If we return n=0, score is 0?
    # Let's ensure we return at least a small valid graph.
    # E.g., C5 is triangle free, alpha=2. n=5.
    if best_n == 0:
        # Fallback to a known small graph
        # C5: 0-1-2-3-4-0
        n = 5
        edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
        adj = [0]*n
        for u,v in edges:
            adj[u] |= 1<<v
            adj[v] |= 1<<u
        best_n = n
        best_adj = adj

    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}

