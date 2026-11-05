import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Helper: Check if a set of steps S defines a triangle-free circulant graph
    def is_triangle_free_circulant(n, S):
        if not S:
            return True
        # D contains all directed differences
        D = set()
        for s in S:
            D.add(s)
            D.add(n - s)
        D.discard(0)
        
        # Check if any a, b, c in D sum to 0 mod n
        # a + b + c = 0 => c = -a - b
        # We iterate over pairs a, b in D
        # To avoid duplicates, we can just check all pairs
        # n is small (<=40), so D is small.
        for a in D:
            for b in D:
                c = (-a - b) % n
                if c in D:
                    # Need to ensure distinct vertices for the triangle
                    # Vertices 0, a, a+b
                    # Edges (0,a), (a, a+b), (a+b, 0)
                    # (0,a) exists if a in D
                    # (a, a+b) exists if b in D
                    # (a+b, 0) exists if -(a+b) in D => c in D
                    # Distinctness:
                    # a != 0 (true)
                    # b != 0 (true)
                    # c != 0 (true)
                    # a != b? If a=b, then 2a+c=0.
                    # Vertices: 0, a, 2a.
                    # If a=b, the edge (0,a) and (a,2a) and (2a,0).
                    # This is a valid triangle if vertices are distinct.
                    # 0 != a (true)
                    # a != 2a => a != 0 (true)
                    # 0 != 2a => 2a != 0 mod n.
                    # If n is even and a = n/2, then 2a = 0. Vertices 0, n/2, 0. Not distinct.
                    # So we must check if 2*a % n != 0.
                    if 2 * a % n != 0:
                        return False
        return True

    def build_adj_circulant(n, S):
        adj = [0] * n
        S_set = set(S)
        for u in range(n):
            for s in S_set:
                v = (u + s) % n
                adj[u] |= (1 << v)
                v2 = (u - s) % n
                adj[u] |= (1 << v2)
        # Remove self loops
        for u in range(n):
            adj[u] &= ~(1 << u)
        return adj

    def find_independent_set_bitmask(adj, n, k):
        """
        Find an independent set of size k in the graph defined by adj (list of bitmasks).
        Returns a list of vertices if found, None otherwise.
        """
        non_neighbors = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            # non-neighbors are those not in adj[i] and not i itself
            mask = full_mask ^ adj[i]
            mask &= ~(1 << i)
            non_neighbors[i] = mask
            
        found = [False]
        best_set = []
        
        def dfs(candidates_mask, current_set):
            if found[0]:
                return
            if len(current_set) == k:
                best_set.extend(current_set)
                found[0] = True
                return
            
            # Pruning: if not enough candidates left
            if bin(candidates_mask).count('1') < k - len(current_set):
                return
            
            if candidates_mask == 0:
                return
                
            # Heuristic: pick vertex with smallest index (or most constrained?)
            # Smallest index is fast.
            lsb = candidates_mask & -candidates_mask
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            new_candidates = candidates_mask & non_neighbors[v] & ~lsb
            dfs(new_candidates, current_set + [v])
            
            if found[0]:
                return
            
            # Branch 2: Exclude v
            dfs(candidates_mask & ~lsb, current_set)
            
        dfs(full_mask, [])
        
        if found[0]:
            return best_set
        return None

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_result = None
    best_n = 0

    # Try circulant graphs first for high n
    for n in [40, 39, 38]:
        if time.time() > deadline:
            break
            
        max_step = n // 2
        steps_range = list(range(1, max_step + 1))
        
        # Random local search over step sets
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            # Initialize S
            # We want high density, so start with many steps
            # But must be triangle-free.
            # Let's start with a random subset and repair.
            S = set(random.sample(steps_range, random.randint(3, max_step)))
            
            # Phase 1: Make triangle-free
            # Loop until triangle-free
            for _ in range(50):
                if not is_triangle_free_circulant(n, S):
                    # Find a conflict and remove a step
                    # For simplicity, just remove a random step
                    if S:
                        S.remove(random.choice(list(S)))
                else:
                    break
            
            if not is_triangle_free_circulant(n, S):
                continue
                
            # Phase 2: Check Alpha
            adj = build_adj_circulant(n, S)
            iso = find_independent_set_bitmask(adj, n, 10)
            
            if iso is None:
                # Valid
                edges = extract_edges(adj, n)
                if n > best_n:
                    best_n = n
                    best_result = {'n': n, 'edges': edges}
                break # Found valid for this n, try next n (smaller) or stop?
                # The goal is to maximize n. So if we found 40, we can return.
                # If we found 39, we might still find 40? No, we tried 40 first.
                # So if we found valid for n, we can return immediately if n is max possible?
                # R(3,10) <= 41, so max n is 40.
                # If we find 40, return.
                # If we find 39, we can try to improve? No, 40 is better.
                # So if we find valid for n, and n is the current target, we return.
                # But maybe we can find 40 in a later attempt?
                # We loop attempts. If we find valid, we should return immediately as 40 is the max possible.
                # Wait, the loop is `for n in [40, 39...]`.
                # If we find valid for 40, we return.
                # If we fail 40, we go to 39.
                # If we find valid for 39, we return 39?
                # Yes, because we are trying to maximize n. 39 > 38.
                # So we return the first valid graph found in the descending order of n.
                return {'n': n, 'edges': edges}

    # Fallback to general random graph repair if circulant search fails
    # This handles cases where the optimal graph is not circulant
    for n in [39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break
            
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Phase 1: Make triangle-free
            for _ in range(2000):
                if time.time() > deadline:
                    break
                tri_found = False
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            common = adj[u] & adj[v]
                            if common:
                                lsb = common & -common
                                w = lsb.bit_length() - 1
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                tri_found = True
                                break
                    if tri_found:
                        break
                if not tri_found:
                    break
            
            # Phase 2: Reduce Alpha
            for _ in range(500):
                if time.time() > deadline:
                    break
                iso = find_independent_set_bitmask(adj, n, 10)
                if iso is None:
                    edges = extract_edges(adj, n)
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    break
                else:
                    improved = False
                    for i in range(10):
                        for j in range(i+1, 10):
                            u = iso[i]
                            v = iso[j]
                            if not ((adj[u] >> v) & 1):
                                if (adj[u] & adj[v]) == 0:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                    improved = True
                                    break
                        if improved:
                            break
                    if not improved:
                        break

    if best_result is not None:
        return best_result
    
    # Fallback small graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
