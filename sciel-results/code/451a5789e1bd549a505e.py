"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import random
    from itertools import combinations

    N = 128
    # Precompute GCDs
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g

    # Identify bad triples: (a, b, c) such that gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We store for each vertex a bitmask of its neighbors in the 3-uniform hypergraph?
    # No, we need to check if adding v to S creates a bad triple.
    # A bad triple is formed by v, x, y in S if gcd(v,x)==gcd(v,y)==gcd(x,y).
    # Let's precompute for each pair (x,y) the set of z that form a bad triple with x,y.
    # But checking validity of adding v to S requires checking all pairs (x,y) in S.
    # If S is large, this is O(|S|^2).
    
    # Optimization: 
    # For a fixed v, a pair (x,y) in S is "bad" with v if gcd(v,x)==gcd(v,y)==gcd(x,y).
    # Let's precompute for each v, a list of pairs (x,y) such that (v,x,y) is a bad triple.
    # Then adding v to S is invalid iff there exists (x,y) in that list such that x in S and y in S.
    
    # bad_pairs_for_v[v] = list of (x, y) where x < y, x != v, y != v, and (v,x,y) is bad.
    # Note: x and y can be in any order, but we store x < y.
    bad_pairs_for_v = [[] for _ in range(N + 1)]
    
    for v in range(1, N + 1):
        for x in range(1, N + 1):
            if x == v: continue
            for y in range(x + 1, N + 1):
                if y == v: continue
                # Check if (v, x, y) is a bad triple
                g_vx = gcds[v][x]
                g_vy = gcds[v][y]
                g_xy = gcds[x][y]
                if g_vx == g_vy == g_xy:
                    # We want to store unique pairs (min, max) for x,y
                    # But we need to know if x and y are both in S.
                    # It's better to store the pair (x,y) directly.
                    bad_pairs_for_v[v].append((x, y))
    
    # To make lookup faster, we can convert bad_pairs_for_v[v] to a set of frozensets?
    # Or just a list. The number of bad triples is not huge.
    # Let's estimate: N=128.
    # Total triples ~ 128^3 / 6 ~ 350k.
    # Bad triples are a subset.
    # For each v, the number of bad pairs is the degree of v in the hypergraph projected to pairs.
    
    def is_valid(v, S):
        """Check if v can be added to set S."""
        for x, y in bad_pairs_for_v[v]:
            if x in S and y in S:
                return False
        return True

    def greedy_construct(seed):
        rng = random.Random(seed)
        # Calculate degree: number of bad triples involving v
        # This is roughly len(bad_pairs_for_v[v]) / 2? No, each bad triple contributes to 3 vertices.
        # For v, the number of bad triples is the number of pairs (x,y) in bad_pairs_for_v[v].
        # Wait, if (v,x,y) is bad, it appears in bad_pairs_for_v[v] as (x,y).
        # It also appears in bad_pairs_for_v[x] as (v,y) and bad_pairs_for_v[y] as (v,x).
        # So len(bad_pairs_for_v[v]) is exactly the number of bad triples containing v.
        
        degrees = [len(bad_pairs_for_v[v]) for v in range(N + 1)]
        
        # Sort by degree, with random tie-breaking
        vertices = list(range(1, N + 1))
        # Add small random noise to break ties and allow different restarts
        scored = [(degrees[v] + rng.random() * 1.0, v) for v in vertices]
        scored.sort()
        
        S = set()
        for _, v in scored:
            if is_valid(v, S):
                S.add(v)
        return S

    def local_search(S, max_iters=100):
        S = set(S)
        current_size = len(S)
        
        for _ in range(max_iters):
            improved = False
            
            # 1. Try to add any vertex
            # Sort candidates by degree to try "safer" additions first?
            # Or just iterate.
            for v in range(1, N + 1):
                if v not in S:
                    if is_valid(v, S):
                        S.add(v)
                        current_size += 1
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. Try 1-for-1 swap
            # Remove u, add v.
            # We want to swap if it allows adding more later, or if it's just a shuffle.
            # Actually, if we can't add anything, a 1-for-1 swap doesn't increase size.
            # It might help escape local minima for 1-for-2.
            
            # 3. Try 1-for-2 swap
            # Remove u, add v1, v2.
            # We need to check if S \ {u} + {v1, v2} is valid.
            # This is equivalent to:
            # - v1 and v2 are not in S \ {u} (i.e., not in S, or one is u? No, u is removed).
            # - is_valid(v1, S \ {u}) is True
            # - is_valid(v2, S \ {u} + {v1}) is True
            
            best_swap = None
            best_gain = 0
            
            # To speed up, we can iterate over u in S.
            # For each u, find all v not in S such that is_valid(v, S \ {u}).
            # Then try to add pairs of such v's.
            
            for u in list(S):
                S_without_u = S - {u}
                
                # Find candidates v that can be added to S_without_u
                candidates = []
                for v in range(1, N + 1):
                    if v not in S_without_u:
                        if is_valid(v, S_without_u):
                            candidates.append(v)
                
                if len(candidates) < 2:
                    continue
                
                # Try to find two candidates v1, v2 that can be added simultaneously
                # We can greedily try to add them.
                # Sort candidates by degree?
                candidates.sort(key=lambda x: len(bad_pairs_for_v[x]))
                
                S_temp = set(S_without_u)
                added = []
                for v in candidates:
                    if is_valid(v, S_temp):
                        S_temp.add(v)
                        added.append(v)
                
                if len(added) >= 2:
                    gain = len(added) - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (u, added)
            
            if best_swap:
                u, added = best_swap
                S.remove(u)
                for v in added:
                    S.add(v)
                current_size += best_gain
                improved = True
                continue
            
            # 4. If no improvement, break
            break
        
        return S

    best_S = set()
    best_size = 0
    num_restarts = 50
    
    for i in range(num_restarts):
        S = greedy_construct(seed=i)
        S = local_search(S, max_iters=50)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    result = sorted(best_S)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
