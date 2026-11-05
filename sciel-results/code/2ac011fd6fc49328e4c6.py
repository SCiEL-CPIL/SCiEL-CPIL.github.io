import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Fallback: The seed graph (Andrasfai graph on 26 vertices)
    def get_seed_graph():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # Target: n = 39
    target_n = 39
    
    # Helper: Check if graph is triangle-free
    def is_triangle_free(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in adj[u]:
                if v > u: # Check each edge once? No, check triangles (u,v,w)
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True

    # Helper: Check if alpha < 10 using Branch and Bound
    def has_independent_set_of_size_k(n, edges, k):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # We want to know if there is an independent set of size k.
        # This is equivalent to finding a clique of size k in the complement.
        # Or just search for independent set in G.
        
        # B&B for Max Independent Set
        # State: current set, candidates
        # Pruning: if len(current) + len(candidates) < k, return False
        
        # Order vertices by degree descending for better pruning?
        # For independent set, picking high degree vertices removes many candidates,
        # which might be good or bad.
        
        # Let's use a simple recursive search
        # To speed up, we can use bitmasks if n <= 64. n=39 fits in int.
        
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        # candidates mask
        # We want to find if there is a set of k vertices with no edges between them.
        
        def dfs(candidates, current_size):
            if current_size >= k:
                return True
            if not candidates:
                return False
            # Pruning: if current_size + popcount(candidates) < k
            # Python int bit_count
            if current_size + bin(candidates).count('1') < k:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with min degree in the induced subgraph of candidates?
            # Or just pick the lowest set bit.
            # Let's pick the lowest set bit for simplicity, or highest degree?
            # For independent set, we want to keep options open.
            # Actually, standard B&B for MIS picks a vertex, includes it (removes neighbors), or excludes it.
            
            # Let's pick vertex v = lowest set bit in candidates
            lsb = candidates & -candidates
            v = (lsb.bit_length() - 1)
            
            # Branch 1: Include v
            # New candidates: candidates without v and without neighbors of v
            new_candidates_include = candidates & ~adj_mask[v] & ~(1 << v)
            if dfs(new_candidates_include, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            new_candidates_exclude = candidates & ~(1 << v)
            if dfs(new_candidates_exclude, current_size):
                return True
            
            return False
        
        initial_candidates = (1 << n) - 1
        return dfs(initial_candidates, 0)

    # Search for a valid circulant graph on n=39
    # Shifts 1..19
    # Triangle free condition for circulant graph C_n(S):
    # No a,b in S such that a+b in S or |a-b| in S (mod n, considering symmetry)
    # Since we use positive reps 1..19, we check:
    # 1. a+b <= 19 and a+b in S -> Triangle (0, a, a+b)
    # 2. a+b > 19: (a+b) - 39 = a+b-39. If a+b-39 in S? No, S is 1..19.
    #    If a+b > 39/2, the edge is (a+b)-39? No, edge is defined by diff.
    #    Diff is min(a+b, 39-(a+b)).
    #    So if min(a+b, 39-(a+b)) in S, then triangle.
    # 3. |a-b| in S -> Triangle (0, a, b)
    
    def is_valid_shift_set(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b: continue
                # Check a+b
                s = a + b
                if s > 19:
                    s = 39 - s
                if s in S_set:
                    return False
                # Check |a-b|
                d = abs(a - b)
                if d in S_set:
                    return False
        return True

    def get_edges_from_shifts(n, S):
        edges = []
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.append([i, j])
        return edges

    # Greedy search for S
    # We want |S| to be large to minimize alpha.
    # Max possible |S| for triangle-free circulant on 39?
    # Ramsey bound suggests avg degree ~ sqrt(n) or n/3?
    # Let's try to build S greedily.
    
    best_graph = None
    
    # Try randomized greedy
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Randomize order of shifts
        shifts = list(range(1, 20))
        random.shuffle(shifts)
        
        S = []
        for s in shifts:
            # Try to add s
            new_S = S + [s]
            if is_valid_shift_set(new_S):
                S = new_S
        
        # Check if this S gives alpha < 10
        # To save time, only check if |S| is "large enough"
        # A dense graph has smaller alpha.
        # If |S| < 10, alpha is likely > 9.
        if len(S) < 10:
            continue
            
        edges = get_edges_from_shifts(39, S)
        
        # Check triangle free (already done by construction, but verify)
        if not is_triangle_free(39, edges):
            continue
            
        # Check alpha < 10
        if not has_independent_set_of_size_k(39, edges, 10):
            # Valid!
            return {'n': 39, 'edges': edges}

    # If we didn't find one on 39, return seed
    return get_seed_graph()

if __name__ == "__main__":
    # Quick test
    g = construct()
    print(f"Found graph with n={g['n']}, edges={len(g['edges'])}")
