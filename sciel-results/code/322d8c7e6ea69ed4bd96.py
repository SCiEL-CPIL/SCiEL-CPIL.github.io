import random
from math import gcd
from itertools import combinations
import time

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
    
    # Precompute all bad triples
    bad_triples = []
    bad_triple_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.append((a, b, c))
                    bad_triple_set.add((a, b, c))
    
    # For each vertex, store which bad triples it's in
    vertex_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)
    
    # Greedy construction: start with all, remove vertices to break all bad triples
    def greedy_removal():
        current = set(range(1, N + 1))
        active = [True] * len(bad_triples)
        conflict_count = {i: len(vertex_to_triples[i]) for i in range(1, N + 1)}
        
        while True:
            max_count = -1
            max_vertex = -1
            for v in current:
                if conflict_count[v] > max_count:
                    max_count = conflict_count[v]
                    max_vertex = v
            if max_count <= 0:
                break
            current.discard(max_vertex)
            for t_idx in vertex_to_triples[max_vertex]:
                if active[t_idx]:
                    active[t_idx] = False
                    a, b, c = bad_triples[t_idx]
                    for v in (a, b, c):
                        if v != max_vertex:
                            conflict_count[v] -= 1
        return current
    
    def can_add(x, s):
        """Check if x can be added to set s without creating a bad triple."""
        # For each pair (u, v) in s, check if {x, u, v} is a bad triple
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            u = s_list[i]
            g_xu = gcd_table[x][u]
            for j in range(i + 1, n):
                v = s_list[j]
                if gcd_table[x][v] == g_xu and gcd_table[u][v] == g_xu:
                    return False
        return True
    
    def local_search(s, deadline):
        """Improve set s by adding elements and trying swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any single element
            candidates = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, s):
                    s.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one or more
            s_list = list(s)
            random.shuffle(s_list)
            for u in s_list:
                if time.time() >= deadline:
                    break
                new_s = s - {u}
                addable = [x for x in range(1, N + 1) if x not in s and x != u]
                random.shuffle(addable)
                added = 0
                for x in addable:
                    if can_add(x, new_s):
                        new_s.add(x)
                        added += 1
                if len(new_s) > len(s):
                    s = new_s
                    improved = True
                    break
        return s
    
    # Branch and bound: try to find a larger valid set
    def branch_and_bound(initial_set, deadline):
        """
        Use the structure of bad triples to search for larger sets.
        We process elements in a fixed order and decide include/exclude.
        """
        best = set(initial_set)
        best_size = len(best)
        
        # Order elements by their "constraint" - elements in more bad triples first
        element_order = sorted(range(1, N + 1), key=lambda x: -len(vertex_to_triples[x]))
        
        # For B&B, we need to track which bad triples are still "active" (not yet broken)
        # A bad triple is active if all 3 of its elements are still "possibly included"
        # We process in order; once we decide to exclude an element, any bad triple containing it is broken.
        
        # This is complex; let's use a simpler approach: iterative local search with multiple restarts.
        return best
    
    # Strategy: multiple greedy + local search restarts
    best_set = greedy_removal()
    best_set = local_search(best_set, deadline)
    best_size = len(best_set)
    
    # Try randomized greedy removals
    while time.time() < deadline:
        current = set(range(1, N + 1))
        active = [True] * len(bad_triples)
        conflict_count = {i: len(vertex_to_triples[i]) for i in range(1, N + 1)}
        
        while True:
            max_count = -1
            candidates = []
            for v in current:
                if conflict_count[v] > max_count:
                    max_count = conflict_count[v]
                    candidates = [v]
                elif conflict_count[v] == max_count:
                    candidates.append(v)
            if max_count <= 0:
                break
            # Random choice among max
            max_vertex = random.choice(candidates)
            current.discard(max_vertex)
            for t_idx in vertex_to_triples[max_vertex]:
                if active[t_idx]:
                    active[t_idx] = False
                    a, b, c = bad_triples[t_idx]
                    for v in (a, b, c):
                        if v != max_vertex:
                            conflict_count[v] -= 1
        
        current = local_search(current, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current.copy()
    
    # Final local search on best
    best_set = local_search(best_set, deadline)
    
    return list(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    # Verify
    from itertools import combinations as comb
    for a, b, c in comb(result, 3):
        g_ab = gcd(a, b)
        g_bc = gcd(b, c)
        g_ca = gcd(c, a)
        if g_ab == g_bc == g_ca:
            print(f"BAD TRIPLE: {a}, {b}, {c} with gcd={g_ab}")
            break
    else:
        print("Valid!")
