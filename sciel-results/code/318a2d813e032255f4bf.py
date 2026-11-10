import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # 2. Build Adjacency Bitmasks
    # adj[i] is an integer bitmask where bit j is 1 if i and j are connected (forbidden difference)
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        adj[i] = mask

    full_mask = (1 << m) - 1

    # Helper to extract list of nodes from mask
    def mask_to_list(mask):
        res = []
        c = mask
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            res.append(v)
            c ^= lsb
        return res

    # Helper to get list of nodes compatible with a set mask (not in set, not adjacent to any in set)
    def get_compatible_nodes(mask):
        blocked = 0
        s = mask
        while s:
            lsb = s & (-s)
            u = lsb.bit_length() - 1
            s ^= lsb
            blocked |= adj[u]
        return full_mask & ~blocked & ~mask

    # --- Strategy 1: CRT Construction ---
    # 205 = 5 * 41
    # Graph G_205 is the direct product of G_5 and G_41.
    # Independent set in product: If A is indep in G_5 and B is indep in G_41,
    # then A x B is indep in G_205.
    # Size = |A| * |B|.
    
    # Solve G_5 (C5): Alpha = 2. E.g., {0, 2}.
    # Verify: 2-0=2 (not square mod 5, squares are 1,4). 0-2=3 (not square).
    A5 = [0, 2]
    
    # Solve G_41: Find max independent set.
    m41 = 41
    forb41 = set()
    for x in range(1, m41):
        forb41.add(pow(x, 2, m41))
    forb41.discard(0)
    
    adj41 = [0] * m41
    for i in range(m41):
        mask = 0
        for j in range(m41):
            if i == j: continue
            if (i - j) % m41 in forb41:
                mask |= (1 << j)
        adj41[i] = mask
        
    full41 = (1 << m41) - 1
    best41_mask = 0
    best41_size = 0
    
    # Heuristic search for G_41
    # Since 41 is small, we can do many restarts quickly.
    for seed in range(2000):
        rng = random.Random(seed)
        cand = full41
        sel = 0
        
        # Randomized Greedy
        # Order nodes by degree in remaining graph? Or just random order?
        # Let's try random sequential greedy with a twist: pick node with min degree in remaining.
        # But for 41, simple random greedy is fast.
        # Let's use min-degree greedy for better initial sets.
        
        for _ in range(m41):
            if cand == 0: break
            min_deg = m41 + 1
            min_nodes = []
            c = cand
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj41[v] & cand).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            if not min_nodes: break
            v = rng.choice(min_nodes)
            sel |= (1 << v)
            cand &= ~(adj41[v] | (1 << v))
            
        # Local Improvement: Add any compatible node
        improved = True
        while improved:
            improved = False
            blocked = 0
            s = sel
            while s:
                lsb = s & (-s)
                u = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj41[u]
            comp = full41 & ~blocked & ~sel
            if comp:
                # Pick one (e.g., first)
                lsb = comp & (-comp)
                v = lsb.bit_length() - 1
                sel |= (1 << v)
                improved = True
        
        if sel.bit_count() > best41_size:
            best41_size = sel.bit_count()
            best41_mask = sel

    A41 = mask_to_list(best41_mask)
    
    # Construct A205 from A5 and A41
    # x = a5 * 41 * inv(41, 5) + a41 * 5 * inv(5, 41)
    # 41 % 5 = 1, inv(1, 5) = 1
    # 5 % 41 = 5, inv(5, 41) = 33 (5*33=165=4*41+1)
    # x = (a5 * 41 + a41 * 165) % 205
    
    init_mask = 0
    for a5 in A5:
        for a41 in A41:
            x = (a5 * 41 + a41 * 165) % 205
            init_mask |= (1 << x)
            
    best_mask = init_mask
    best_size = best_mask.bit_count()

    # --- Strategy 2: Local Search on G_205 ---
    # Start from best_mask and try to improve.
    # Also run independent randomized searches on G_205 to see if we can beat the CRT bound.
    # (Note: Alpha of direct product can be larger than product of alphas, but usually close.
    #  For Cayley graphs, it's often exactly the product if the groups are cyclic and S is product.)
    # Actually, for cyclic groups, if S = S1 x S2, then Alpha(G) = Alpha(G1) * Alpha(G2) is a known result for certain classes, 
    # but let's verify empirically by searching.
    
    def improve_mask(mask, rng, max_iter=1000):
        """
        Local search: 
        1. Add any compatible node.
        2. Try swaps: remove u, add v if size increases or same (with some probability) to escape local optima.
        """
        current = mask
        current_size = current.bit_count()
        
        for _ in range(max_iter):
            # 1. Add
            comp = get_compatible_nodes(current)
            if comp:
                # Pick node with min degree in comp? Or random?
                # Random is fine for diversity.
                # Let's pick one that has fewest neighbors in comp to preserve options?
                # For speed, just pick first or random.
                # Let's pick random from comp.
                # To pick random from bitmask efficiently:
                # We can't easily do uniform random from bitmask without list.
                # Since comp is sparse-ish, list is ok.
                comp_list = mask_to_list(comp)
                v = rng.choice(comp_list)
                current |= (1 << v)
                current_size += 1
                continue
            
            # 2. Swap / Kick
            # If no add possible, try a kick.
            # Remove a random node u.
            cur_list = mask_to_list(current)
            if not cur_list:
                break
            u = rng.choice(cur_list)
            new_mask = current & ~(1 << u)
            
            # Try to add nodes to new_mask
            # We want to add at least 1 node to maintain size, or 2 to grow.
            # Let's run a greedy fill on new_mask.
            
            # Calculate blocked for new_mask
            blocked = 0
            s = new_mask
            while s:
                lsb = s & (-s)
                w = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj[w]
            
            valid_v = full_mask & ~blocked & ~new_mask & ~(1 << u)
            
            if valid_v:
                # Pick a valid v. 
                # Ideally one that allows further growth.
                # Heuristic: pick v with min degree in valid_v?
                # Or just pick one and run improve.
                valid_list = mask_to_list(valid_v)
                v = rng.choice(valid_list)
                candidate = new_mask | (1 << v)
                
                # Try to add more to candidate
                # Simple loop: add any compatible
                improved = True
                while improved:
                    improved = False
                    comp_c = get_compatible_nodes(candidate)
                    if comp_c:
                        comp_list = mask_to_list(comp_c)
                        v2 = rng.choice(comp_list)
                        candidate |= (1 << v2)
                        improved = True
                
                if candidate.bit_count() > current_size:
                    current = candidate
                    current_size = candidate.bit_count()
                elif candidate.bit_count() == current_size:
                    # Accept with some probability to explore?
                    # Or just reject to ensure progress?
                    # Let's reject unless it's a different configuration that might lead to growth later.
                    # For simplicity, reject ties to keep the search "hill climbing".
                    # But this might get stuck.
                    # Let's accept ties with prob 0.1?
                    if rng.random() < 0.1:
                        current = candidate
                        current_size = candidate.bit_count()
            else:
                # No valid swap found? Restore u?
                # Actually, if valid_v is empty, we can't swap u with anything.
                # Try removing a different node?
                # For now, just break or try next iteration.
                pass

        return current

    # Run local search on the initial CRT set
    rng = random.Random(123)
    best_mask = improve_mask(best_mask, rng, max_iter=5000)
    best_size = best_mask.bit_count()

    # Run multiple independent searches on G_205
    # This might find sets larger than the CRT product bound.
    
    while time.time() < deadline - 10:
        seed = int(time.time() * 1000) % 100000
        rng = random.Random(seed)
        
        # Random Greedy Start
        cand = full_mask
        sel = 0
        
        # Randomized Min-Degree Greedy
        for _ in range(m):
            if cand == 0: break
            min_deg = m + 1
            min_nodes = []
            c = cand
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & cand).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            if not min_nodes: break
            v = rng.choice(min_nodes)
            sel |= (1 << v)
            cand &= ~(adj[v] | (1 << v))
            
        # Local Search
        sel = improve_mask(sel, rng, max_iter=2000)
        
        if sel.bit_count() > best_size:
            best_size = sel.bit_count()
            best_mask = sel

    # Final extraction
    res = mask_to_list(best_mask)
    return np.array(res, dtype=np.int64)
