import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_fallback():
        n = 26
        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

    target_n = 39

    def has_independent_set_size_k(n, S, k):
        # S is set of differences
        # Build adjacency bitsets for complement graph
        # G is circulant. Edge (u,v) if diff in S.
        # Independent set in G is Clique in G_bar.
        # G_bar edge (u,v) if diff NOT in S.
        
        # We want to find a clique of size k in G_bar.
        # Bitset representation of G_bar adjacency.
        # adj[i] is a bitset of neighbors of i in G_bar.
        
        adj = [0] * n
        S_set = set(S)
        for i in range(n):
            mask = 0
            for j in range(n):
                if i == j: continue
                d = (j - i) % n
                if d > n//2: d = n - d
                if d not in S_set:
                    mask |= (1 << j)
            adj[i] = mask
            
        # Find clique of size k in G_bar
        # Use Bron-Kerbosch or simple backtracking with bitsets
        # Since we just need to know if one exists, we can use a heuristic search.
        
        def find_clique(candidates, size):
            if size == 0:
                return True
            if candidates == 0:
                return False
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in candidates
            # Or just pick lowest set bit
            # For speed, pick lowest set bit
            v_bit = 1
            while v_bit & candidates == 0:
                v_bit <<= 1
            v = (v_bit.bit_length() - 1)
            
            # Remove v from candidates
            new_candidates = candidates & ~(1 << v)
            
            # Intersect with neighbors of v
            new_candidates &= adj[v]
            
            if find_clique(new_candidates, size - 1):
                return True
            return False
        
        # Try all starting vertices?
        # By symmetry, we can fix 0 in the clique.
        # If a clique of size k exists, one containing 0 exists?
        # Not necessarily for general graphs, but for circulant, yes (by rotation).
        # So we fix 0.
        
        candidates = adj[0]
        return find_clique(candidates, k - 1)

    def is_valid_circulant(n, S):
        # Check triangle-free
        diffs = set()
        for u in S:
            for v in S:
                d = (v - u) % n
                if d != 0:
                    if d > n//2: d = n - d
                    diffs.add(d)
        if diffs & S:
            return False
            
        # Check alpha < 10
        if has_independent_set_size_k(n, S, 10):
            return False
        return True

    # Search
    best_result = None
    for restart in range(100):
        if time.time() > deadline:
            break
        S = set()
        diffs = list(range(1, target_n // 2 + 1))
        random.shuffle(diffs)
        
        for d in diffs:
            valid = True
            for s in S:
                d1 = (d - s) % target_n
                if d1 > target_n // 2: d1 = target_n - d1
                if d1 in S:
                    valid = False
                    break
                d2 = (s - d) % target_n
                if d2 > target_n // 2: d2 = target_n - d2
                if d2 in S:
                    valid = False
                    break
            if valid:
                S.add(d)
                
        if time.time() > deadline:
            break
            
        if is_valid_circulant(target_n, S):
            best_result = {'n': target_n, 'edges': [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if (j - i) % target_n in S or (i - j) % target_n in S]}
            break

    if best_result:
        return best_result
    return get_fallback()
