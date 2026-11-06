import itertools
import random
import time

def construct():
    # 1. Generate all 4-subsets of [1, 16]
    # Represent sets as frozensets or bitmasks. Bitmasks are faster for intersection.
    # Map element 1..16 to bit 0..15.
    all_sets = []
    set_to_idx = {}
    for combo in itertools.combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
        set_to_idx[mask] = len(all_sets) - 1
    
    n_sets = len(all_sets)
    
    # Precompute popcount for 16-bit ints? Or just use bin().count('1')
    # For speed, precompute popcount for 0..2^16-1
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # 2. Precompute bad_pairs
    # bad_pairs[X] = list of (u, v) such that all_sets[u], all_sets[v], all_sets[X] form a sunflower.
    bad_pairs = [[] for _ in range(n_sets)]
    
    # Iterate all pairs (u, v)
    for u in range(n_sets):
        Au = all_sets[u]
        for v in range(u + 1, n_sets):
            Av = all_sets[v]
            K = Au & Av
            k_size = popcount[K]
            
            # X must contain K and be disjoint from (Au \ K) U (Av \ K)
            # Au \ K = Au & ~K
            # Av \ K = Av & ~K
            forbidden = (Au | Av) & ~K
            # X = K | P, P subset of [16] \ (K | forbidden)
            # [16] \ (K | forbidden) is the set of available elements for P.
            # Note: K and forbidden are disjoint.
            available = ((1 << 16) - 1) & ~(K | forbidden)
            
            # We need to choose (4 - k_size) elements from 'available' to form P.
            # Then X = K | P.
            # We need to find indices X such that X is in all_sets.
            
            # To avoid iterating all subsets of 'available' (which can be large),
            # we can just iterate all_sets and check? No, that's O(N^3).
            # Better: Iterate subsets of 'available'.
            # Size of 'available' is 16 - |K| - |forbidden|.
            # |forbidden| = |Au \ K| + |Av \ K| = (4 - k_size) + (4 - k_size) = 8 - 2*k_size.
            # |available| = 16 - k_size - (8 - 2*k_size) = 8 + k_size.
            # We need to choose p_size = 4 - k_size from available.
            
            p_size = 4 - k_size
            if p_size < 0: continue
            
            # Get list of bits in 'available'
            avail_bits = []
            temp = available
            while temp:
                lsb = temp & -temp
                avail_bits.append(lsb)
                temp ^= lsb
            
            # Generate combinations of p_size from avail_bits
            # If p_size is 0, just one combo (empty)
            # If p_size is 1, 11 combos max
            # If p_size is 2, 45 combos max
            # If p_size is 3, 84 combos max
            # If p_size is 4, 70 combos max
            
            for P_combo in itertools.combinations(avail_bits, p_size):
                P = 0
                for b in P_combo:
                    P |= b
                X_mask = K | P
                X_idx = set_to_idx.get(X_mask, -1)
                if X_idx != -1:
                    bad_pairs[X_idx].append((u, v))
                    # Also, the pair (u, v) conflicts with X.
                    # We store this in bad_pairs[X].
    
    # 3. Local Search
    # Initialize F greedily
    F = set()
    in_F = [False] * n_sets
    
    # Greedy initialization
    # Order sets by some heuristic? Random is fine.
    indices = list(range(n_sets))
    random.shuffle(indices)
    
    for idx in indices:
        # Check if idx can be added
        valid = True
        for u, v in bad_pairs[idx]:
            if in_F[u] and in_F[v]:
                valid = False
                break
        if valid:
            F.add(idx)
            in_F[idx] = True
            
    best_size = len(F)
    
    # Hill Climbing / Simulated Annealing
    start_time = time.time()
    time_limit = 85.0 # 5s margin
    
    # Parameters
    T = 1.0
    cooling = 0.9999
    
    while time.time() - start_time < time_limit:
        # Pick a random set not in F
        # If F is large, picking a random index and checking is O(1)
        X = random.randint(0, n_sets - 1)
        if in_F[X]:
            continue
            
        # Check conflicts
        conflicts = []
        for u, v in bad_pairs[X]:
            if in_F[u] and in_F[v]:
                conflicts.append((u, v))
                
        if not conflicts:
            # Add X
            F.add(X)
            in_F[X] = True
            if len(F) > best_size:
                best_size = len(F)
        else:
            # Try to resolve conflicts
            # Find a small vertex cover of the conflict graph
            # Conflict graph vertices are the sets in conflicts.
            # Edges are the pairs.
            
            # Simple heuristic: Pick a random conflicting pair, remove one of them.
            # Or try to remove a set that is in multiple conflicts.
            
            # Count degree of each vertex in conflicts
            deg = {}
            for u, v in conflicts:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            # Pick vertex with max degree
            if deg:
                rem_candidates = [k for k, v in deg.items()]
                # Sort by degree
                rem_candidates.sort(key=lambda x: deg[x], reverse=True)
                
                # Try removing the top candidate
                # Check if removing it resolves all conflicts?
                # No, it only resolves conflicts involving it.
                # We need to remove a set of vertices that hits all edges.
                
                # Greedy vertex cover
                edges_to_cover = conflicts[:]
                cover = set()
                remaining_edges = edges_to_cover
                while remaining_edges:
                    # Pick edge
                    u, v = remaining_edges[0]
                    # Pick one of u, v to remove
                    # Prefer the one with higher degree in the REMAINING graph?
                    # For simplicity, just pick u.
                    # Better: pick the one that covers more other edges.
                    # Let's just pick the one with higher global degree for now.
                    if deg.get(u, 0) >= deg.get(v, 0):
                        cover.add(u)
                        # Remove all edges incident to u
                        remaining_edges = [(a, b) for a, b in remaining_edges if a != u and b != u]
                    else:
                        cover.add(v)
                        remaining_edges = [(a, b) for a, b in remaining_edges if a != v and b != v]
                
                # We have a cover 'cover'.
                # If we remove 'cover' and add X, new size is len(F) - len(cover) + 1.
                # We accept if:
                # 1. len(cover) == 0 (not possible here)
                # 2. len(cover) == 1 and (random() < T or we want to move sideways)
                # 3. len(cover) == 0 (size increase) - handled above
                
                if len(cover) == 1:
                    # Sideways move
                    # Accept with probability T?
                    if random.random() < T:
                        rem = list(cover)[0]
                        F.remove(rem)
                        in_F[rem] = False
                        F.add(X)
                        in_F[X] = True
                # If len(cover) > 1, size decreases. Skip for now to maintain size.
            
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor to allow some sideways moves always?
            # Or reheat?
            # Let's just keep it low.
            
    return [sorted([i+1 for i in range(16) if (all_sets[idx] >> i) & 1]) for idx in F]

# Test locally? No, just output.
