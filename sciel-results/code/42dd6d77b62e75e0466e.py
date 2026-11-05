import time
import random

def _has_c5_circulant(n, S):
    """
    Check if the circulant graph C_n(S) contains a 5-cycle.
    S is a set of positive integers < n/2.
    """
    # Generate all possible steps
    steps = list(S) + [-s for s in S]
    
    # We look for s1, s2, s3, s4, s5 in steps such that:
    # sum(si) = 0 mod n
    # partial sums p0=0, p1=s1, p2=s1+s2, p3=s1+s2+s3, p4=s1+s2+s3+s4 are distinct and non-zero mod n
    
    # Optimization: iterate s1, s2, s3.
    # target = -(s1+s2+s3)
    # Check if there exist s4, s5 in steps such that s4+s5 = target.
    # And check distinctness of partial sums.
    
    # Precompute sums of pairs in steps
    # pair_sums[target] = list of (s4, s5) pairs
    pair_sums = {}
    for s4 in steps:
        for s5 in steps:
            s = (s4 + s5) % n
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((s4, s5))
            
    for s1 in steps:
        p1 = s1 % n
        if p1 == 0: continue
        for s2 in steps:
            p2 = (p1 + s2) % n
            if p2 == 0 or p2 == p1: continue
            for s3 in steps:
                p3 = (p2 + s3) % n
                if p3 == 0 or p3 == p1 or p3 == p2: continue
                
                target = (-p3) % n
                if target in pair_sums:
                    for s4, s5 in pair_sums[target]:
                        p4 = (p3 + s4) % n
                        if p4 == 0 or p4 == p1 or p4 == p2 or p4 == p3:
                            continue
                        # Check if p4 + s5 == 0 mod n (guaranteed by target)
                        # Check distinctness of all 5 vertices: 0, p1, p2, p3, p4
                        # They are all distinct and non-zero by construction.
                        # Also need to ensure the cycle closes correctly, which it does by sum=0.
                        # One more check: the vertices are 0, s1, s1+s2, s1+s2+s3, s1+s2+s3+s4.
                        # These are 0, p1, p2, p3, p4.
                        # Are they all distinct? Yes, checked.
                        # Is the last step s5 valid? Yes, it's in steps.
                        return True
    return False

def _calc_alpha(adj, n):
    """Calculate independence number using branch and bound."""
    adj_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_sets[i].add(j)
                adj_sets[j].add(i)
    
    best = [0]
    
    def dfs(candidates, size):
        if size + len(candidates) <= best[0]:
            return
        if not candidates:
            best[0] = max(best[0], size)
            return
        
        # Use a better heuristic: pick vertex with minimum degree in the induced subgraph
        min_deg = float('inf')
        v = -1
        for c in candidates:
            deg = len(adj_sets[c] & candidates)
            if deg < min_deg:
                min_deg = deg
                v = c
        
        remaining = set(candidates)
        remaining.discard(v)
        
        # Include v
        new_cands = set(remaining)
        for nb in adj_sets[v]:
            new_cands.discard(nb)
        dfs(new_cands, size + 1)
        
        # Exclude v
        dfs(remaining, size)
    
    dfs(set(range(n)), 0)
    return best[0]

def construct():
    deadline = time.time() + 145
    
    # Try n=33 with circulant graphs
    n = 33
    max_half = n // 2
    possible_diffs = list(range(1, max_half + 1))
    
    rng = random.Random(42)
    
    found = None
    
    # Try random subsets of possible_diffs
    # Bias towards sizes that might work
    # For alpha < 9, we need enough edges.
    # Turan bound?
    # Let's try sizes from 8 to 14
    
    for attempt in range(20000):
        if time.time() > deadline - 10:
            break
            
        size = rng.randint(8, 14)
        S = set(rng.sample(possible_diffs, size))
        
        if _has_c5_circulant(n, S):
            continue
            
        # Build graph
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                diff = (v - u) % n
                if diff in S or (-diff) % n in S:
                    adj[u][v] = True
                    adj[v][u] = True
        
        alpha = _calc_alpha(adj, n)
        if alpha < 9:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges.append([u, v])
            found = {'n': n, 'edges': edges}
            break
            
    if found:
        return found
        
    # Fallback n=32
    n = 32
    edges = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges.append([verts[i], verts[j]])
                
    return {'n': n, 'edges': edges}
