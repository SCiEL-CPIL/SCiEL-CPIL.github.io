import time
import random
from itertools import combinations

def _has_triangle_circulant(n, S):
    """
    Check if circulant graph C(n, S) has a triangle.
    S is a set of shifts. The graph is symmetric, so if s in S, n-s is effectively in S.
    Triangle exists if there are a, b, c in S (with a,b,c not necessarily distinct, 
    but representing edges) such that a+b+c = 0 mod n, or a+b = c mod n, etc.
    More precisely, vertices 0, a, a+b form a triangle if a in S, b in S, and a+b in S.
    Also 0, a, -b form a triangle if a in S, b in S, and a-b in S (i.e., -(a-b) in S).
    """
    S_mod = set(s % n for s in S)
    # Ensure symmetry for checking: if s in S, n-s is in the neighbor set.
    # The neighbor set is N = S ∪ {n-s for s in S}.
    N = set()
    for s in S_mod:
        if s != 0:
            N.add(s)
            N.add((n - s) % n)
    
    # Check for triangles 0, u, v.
    # Edge 0-u if u in N.
    # Edge 0-v if v in N.
    # Edge u-v if v-u in N.
    # So we need u, v in N such that (v-u) % n in N.
    # This is equivalent to finding a, b, c in N such that a+b=c mod n?
    # Let u=a, v=c. Then c-a = b. So a+b=c.
    # We need to check if there exist a, b, c in N such that (a+b)%n == c.
    # Note: a, b, c can be any elements of N.
    
    N_list = list(N)
    for a in N_list:
        for b in N_list:
            c = (a + b) % n
            if c in N:
                # Check if a, b, c form a valid triangle (distinct vertices)
                # Vertices are 0, a, a+b.
                # If a == 0 or b == 0, it's not a triangle (loop or multi-edge).
                # But N does not contain 0.
                # If a == b, vertices are 0, a, 2a. Triangle if 2a in N.
                # If a != b, vertices are 0, a, a+b. Triangle if a+b in N.
                # We must ensure vertices are distinct: a != 0, b != 0, a+b != 0.
                # Since N has no 0, a,b != 0.
                # If a+b == 0, then c=0, which is not in N. So no triangle.
                return True
    return False

def _max_independent_set_size_bitset(adj, n, k):
    """
    Check if there exists an independent set of size k.
    """
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    
    def dfs(candidates_mask, current_size):
        if found[0]:
            return
        if current_size == k:
            found[0] = True
            return
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - current_size:
            return
        
        # Pick vertex with most non-neighbors in candidates
        mask = candidates_mask
        best_v = None
        best_count = -1
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_count = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_count > best_count:
                best_count = nn_count
                best_v = v
            mask ^= lsb
        
        if best_v is None:
            return
        
        # Include
        new_cands = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_cands, current_size + 1)
        
        if found[0]:
            return
        
        # Exclude
        dfs(candidates_mask & ~(1 << best_v), current_size)
    
    dfs(full_mask, 0)
    return found[0]

def _circulant_to_bitset(n, S):
    adj = [0] * n
    S_mod = set(s % n for s in S)
    N = set()
    for s in S_mod:
        if s != 0:
            N.add(s)
            N.add((n - s) % n)
            
    for i in range(n):
        for s in N:
            j = (i + s) % n
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _bitset_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Target n=39
    n = 39
    
    # Search for a circulant graph C(39, S) that is triangle-free and alpha < 10
    # We try S of size 4 and 5 (degree 8 and 10)
    # This covers a large portion of the "dense" triangle-free circulants
    
    found_graph = None
    
    for size in [4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Generate all subsets of {1, ..., 19} of given size
        # We restrict to s <= n//2 to avoid duplicates due to symmetry
        candidates = list(range(1, n // 2 + 1))
        
        # To speed up, we can skip sets that obviously contain triangles
        # e.g. if 1, 2 in S, then 1+1=2 is a triangle.
        # So S cannot contain {1, 2}.
        # If 1, 3 in S, 1+1=2 (ok if 2 not in S), 1+3=4 (ok if 4 not in S).
        
        for S in combinations(candidates, size):
            if time.time() > deadline:
                break
            
            # Quick check: if S contains a, b, and a+b is in S (mod n), it has a triangle.
            # Also if a-b is in S.
            S_set = set(S)
            has_tri = False
            for a in S_set:
                for b in S_set:
                    if (a + b) % n in S_set:
                        # Check if a+b != 0 and vertices distinct
                        if (a + b) % n != 0:
                            has_tri = True
                            break
                    if (a - b) % n in S_set:
                        if (a - b) % n != 0:
                            has_tri = True
                            break
                if has_tri:
                    break
            
            if has_tri:
                continue
            
            # Full triangle check
            if _has_triangle_circulant(n, S):
                continue
            
            # Check independence number
            adj = _circulant_to_bitset(n, S)
            if not _max_independent_set_size_bitset(adj, n, 10):
                found_graph = {'n': n, 'edges': _bitset_to_edges(adj, n)}
                break
        
        if found_graph:
            break
    
    if found_graph:
        return found_graph
    
    # Fallback: Try n=38
    n = 38
    for size in [4, 5, 6]:
        if time.time() > deadline:
            break
        candidates = list(range(1, n // 2 + 1))
        for S in combinations(candidates, size):
            if time.time() > deadline:
                break
            if _has_triangle_circulant(n, S):
                continue
            adj = _circulant_to_bitset(n, S)
            if not _max_independent_set_size_bitset(adj, n, 10):
                return {'n': n, 'edges': _bitset_to_edges(adj, n)}
                
    # Final fallback: n=9 empty graph
    return {'n': 9, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
