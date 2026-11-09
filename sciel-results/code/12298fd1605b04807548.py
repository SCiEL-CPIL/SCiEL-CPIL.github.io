import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for return/eval overhead
    
    N = 16
    E = N * (N - 1) // 2  # 120
    
    # Precompute edge index mapping
    def edge_idx(u, v):
        if u > v:
            u, v = v, u
        # Lexicographic order: (0,1)..(0,15), (1,2)..(1,15), ...
        # Index = sum_{k=0}^{u-1} (N - 1 - k) + (v - u - 1)
        # Sum = u * (N - 1 + N - u) / 2 = u * (2N - 1 - u) / 2
        return u * (2 * N - 1 - u) // 2 + (v - u - 1)

    # Precompute K4s and their edge indices
    k4s = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    idxs = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4s.append(idxs)
    
    num_k4s = len(k4s)

    def count_violations(colors):
        # colors is a list of 120 ints
        violations = 0
        for idxs in k4s:
            c0 = colors[idxs[0]]
            c1 = colors[idxs[1]]
            c2 = colors[idxs[2]]
            c3 = colors[idxs[3]]
            c4 = colors[idxs[4]]
            c5 = colors[idxs[5]]
            
            # Count distinct colors
            # d = 6
            # d -= (c0==c1) + (c0==c2) + ...
            d = 6
            if c0 == c1: d -= 1
            if c0 == c2: d -= 1
            if c0 == c3: d -= 1
            if c0 == c4: d -= 1
            if c0 == c5: d -= 1
            if c1 == c2: d -= 1
            if c1 == c3: d -= 1
            if c1 == c4: d -= 1
            if c1 == c5: d -= 1
            if c2 == c3: d -= 1
            if c2 == c4: d -= 1
            if c2 == c5: d -= 1
            if c3 == c4: d -= 1
            if c3 == c5: d -= 1
            if c4 == c5: d -= 1
            
            if d < 5:
                violations += 1
                
        return violations

    def repair_search(num_colors, max_iters=20000):
        # Strategy: Start with a random coloring.
        # If there are violations, pick a random violated K4.
        # Pick a random edge in that K4.
        # Change its color to a random one (or try all to find improvement).
        # This focuses search on fixing constraints.
        
        colors = [random.randint(0, num_colors - 1) for _ in range(E)]
        current_bad = count_violations(colors)
        
        if current_bad == 0:
            return colors
            
        # Precompute for each edge, which K4s it belongs to?
        # Not strictly necessary if we just pick a random K4, but might help.
        
        # To pick a violated K4 efficiently, we might need to scan.
        # Scanning 1820 K4s is fast.
        
        for i in range(max_iters):
            if time.time() > deadline:
                return None
            
            if current_bad == 0:
                return colors
                
            # Find a violated K4
            # We can just iterate until we find one
            # Since current_bad > 0, we will find one.
            # But we don't know which one without checking.
            # Let's just check all K4s until we find a violation.
            # This is O(num_k4s) per step. 1820 steps * 20000 iters = 36M checks.
            # Might be too slow in Python if not optimized.
            
            # Optimization: Maintain a list of violated K4 indices?
            # Hard to update incrementally without complex bookkeeping.
            
            # Alternative: Just pick a random K4. If it's valid, skip?
            # If we pick a random K4, prob of it being violated is current_bad/num_k4s.
            # If current_bad is small, we waste time.
            
            # Let's try to be smarter:
            # 1. Check if current_bad is small.
            # 2. If so, scan to find one.
            
            # Actually, let's just scan for the first violation.
            # Worst case 1820 checks.
            # 20000 iters * 1820 = 36.4M simple ops.
            # Python can do ~10M-50M simple ops/sec.
            # 36M ops might take 1-3 seconds. This is acceptable.
            
            viol_k4_idx = -1
            for k_idx, idxs in enumerate(k4s):
                c0 = colors[idxs[0]]
                c1 = colors[idxs[1]]
                c2 = colors[idxs[2]]
                c3 = colors[idxs[3]]
                c4 = colors[idxs[4]]
                c5 = colors[idxs[5]]
                
                d = 6
                if c0 == c1: d -= 1
                if c0 == c2: d -= 1
                if c0 == c3: d -= 1
                if c0 == c4: d -= 1
                if c0 == c5: d -= 1
                if c1 == c2: d -= 1
                if c1 == c3: d -= 1
                if c1 == c4: d -= 1
                if c1 == c5: d -= 1
                if c2 == c3: d -= 1
                if c2 == c4: d -= 1
                if c2 == c5: d -= 1
                if c3 == c4: d -= 1
                if c3 == c5: d -= 1
                if c4 == c5: d -= 1
                
                if d < 5:
                    viol_k4_idx = k_idx
                    break
            
            if viol_k4_idx == -1:
                # Should not happen if current_bad > 0, but safety check
                # Maybe current_bad was stale?
                current_bad = count_violations(colors)
                if current_bad == 0:
                    return colors
                continue

            idxs = k4s[viol_k4_idx]
            
            # Pick a random edge in this K4
            e_pos = random.randint(0, 5)
            e_idx = idxs[e_pos]
            old_c = colors[e_idx]
            
            # Try to find a better color for this edge
            # We want to reduce violations.
            # Let's try all num_colors and pick the one that minimizes violations.
            # To do this efficiently, we only need to check K4s affected by this edge.
            # But we don't have the list of affected K4s precomputed.
            
            # Fallback: Just try random changes or a few specific ones.
            # If we try all 15 colors, we need to evaluate the change.
            # Evaluating the whole board is expensive.
            
            # Let's just try a random new color first.
            # If that doesn't work after some tries, try others.
            
            # Actually, a "hill climbing" step:
            # Try all colors. For each, calculate the delta in violations.
            # To calculate delta, we need to know which K4s change status.
            # A K4 changes status if it contains this edge.
            # How many K4s contain a specific edge (u,v)?
            # We need to pick 2 more vertices from the remaining 14.
            # C(14,2) = 91 K4s.
            # So changing one edge affects 91 K4s.
            # Checking 91 K4s is much faster than 1820.
            
            # Let's precompute for each edge, the list of K4 indices it belongs to.
            # This is a structural change to speed up the search.
            # But we are inside the loop. We can precompute this once outside.
            
            # Wait, I am writing the code now. I should precompute edge_to_k4s.
            
            # Let's restart the logic with precomputed edge_to_k4s.
            
            return None # Placeholder, will rewrite function

    # Precompute edge_to_k4s
    edge_to_k4s = [[] for _ in range(E)]
    for k_idx, idxs in enumerate(k4s):
        for e_idx in idxs:
            edge_to_k4s[e_idx].append(k_idx)

    def repair_search_v2(num_colors, max_iters=20000):
        colors = [random.randint(0, num_colors - 1) for _ in range(E)]
        current_bad = count_violations(colors)
        
        if current_bad == 0:
            return colors
            
        for i in range(max_iters):
            if time.time() > deadline:
                return None
            
            if current_bad == 0:
                return colors
                
            # Find a violated K4
            # Instead of scanning all, we can pick a random edge that is in a violated K4?
            # Or just scan for a violated K4 as before.
            # Scanning is O(1820).
            
            viol_k4_idx = -1
            for k_idx, idxs in enumerate(k4s):
                c0 = colors[idxs[0]]
                c1 = colors[idxs[1]]
                c2 = colors[idxs[2]]
                c3 = colors[idxs[3]]
                c4 = colors[idxs[4]]
                c5 = colors[idxs[5]]
                
                d = 6
                if c0 == c1: d -= 1
                if c0 == c2: d -= 1
                if c0 == c3: d -= 1
                if c0 == c4: d -= 1
                if c0 == c5: d -= 1
                if c1 == c2: d -= 1
                if c1 == c3: d -= 1
                if c1 == c4: d -= 1
                if c1 == c5: d -= 1
                if c2 == c3: d -= 1
                if c2 == c4: d -= 1
                if c2 == c5: d -= 1
                if c3 == c4: d -= 1
                if c3 == c5: d -= 1
                if c4 == c5: d -= 1
                
                if d < 5:
                    viol_k4_idx = k_idx
                    break
            
            if viol_k4_idx == -1:
                # Stale current_bad?
                current_bad = count_violations(colors)
                if current_bad == 0:
                    return colors
                continue

            idxs = k4s[viol_k4_idx]
            
            # Pick a random edge in this K4
            e_pos = random.randint(0, 5)
            e_idx = idxs[e_pos]
            old_c = colors[e_idx]
            
            # Get K4s affected by this edge
            affected_k4s = edge_to_k4s[e_idx]
            
            # Try all colors for this edge
            best_new_c = old_c
            best_bad = current_bad
            
            for new_c in range(num_colors):
                if new_c == old_c:
                    continue
                
                # Calculate new badness
                # We need to check the status of affected_k4s
                # But other K4s remain unchanged.
                # So new_bad = current_bad - (violations in affected_k4s with old_c) + (violations in affected_k4s with new_c)
                
                # To do this, we need to know the violation status of each affected K4 with old_c.
                # We can just check them.
                
                delta = 0
                for k_idx in affected_k4s:
                    k_idxs = k4s[k_idx]
                    
                    # Check old status
                    # We have colors[e_idx] = old_c currently
                    # So we can check the current status.
                    c0 = colors[k_idxs[0]]
                    c1 = colors[k_idxs[1]]
                    c2 = colors[k_idxs[2]]
                    c3 = colors[k_idxs[3]]
                    c4 = colors[k_idxs[4]]
                    c5 = colors[k_idxs[5]]
                    
                    d_old = 6
                    if c0 == c1: d_old -= 1
                    if c0 == c2: d_old -= 1
                    if c0 == c3: d_old -= 1
                    if c0 == c4: d_old -= 1
                    if c0 == c5: d_old -= 1
                    if c1 == c2: d_old -= 1
                    if c1 == c3: d_old -= 1
                    if c1 == c4: d_old -= 1
                    if c1 == c5: d_old -= 1
                    if c2 == c3: d_old -= 1
                    if c2 == c4: d_old -= 1
                    if c2 == c5: d_old -= 1
                    if c3 == c4: d_old -= 1
                    if c3 == c5: d_old -= 1
                    if c4 == c5: d_old -= 1
                    
                    old_viol = 1 if d_old < 5 else 0
                    
                    # Check new status
                    # We need to simulate the change.
                    # The edge e_idx is one of the 6 edges in k_idxs.
                    # We need to know which position it is.
                    # This is inefficient.
                    
                    # Let's just do a full check of the affected K4s with the new color temporarily set?
                    # No, that modifies state.
                    
                    # Better: Precompute for each K4, the position of each edge?
                    # Or just extract the 6 colors, replace the one at e_idx, and check.
                    
                    # We need to map e_idx to position in k_idxs.
                    # k_idxs is a list of 6 edge indices.
                    # We can find the position.
                    # Since k_idxs is small, linear search is fine.
                    
                    # Actually, we can just check the 6 colors.
                    # But we need to know which one changes.
                    
                    # Let's store the colors of the 6 edges.
                    # Then replace the one corresponding to e_idx.
                    
                    # To avoid linear search every time, we can precompute edge_pos_in_k4.
                    # But let's just do it simply for now.
                    
                    # Find position of e_idx in k_idxs
                    pos = -1
                    for j in range(6):
                        if k_idxs[j] == e_idx:
                            pos = j
                            break
                            
                    if pos == -1:
                        # Should not happen
                        continue
                        
                    # Get the 6 colors
                    # We already have c0..c5.
                    # Let's put them in a list for easy modification?
                    # Or just handle it.
                    
                    # Let's just create a tuple of 6 colors, replace pos with new_c, and check distinct count.
                    # This is faster than manual if/else if we use a helper?
                    # No, manual if/else is fastest in Python for small N.
                    # But we need to replace one value.
                    
                    # Let's just re-evaluate the K4 with the new color.
                    # We can do this by checking the 6 values, one of which is new_c.
                    
                    # Values:
                    # v0 = colors[k_idxs[0]] if k_idxs[0] != e_idx else new_c
                    # ...
                    
                    # This is 6 conditionals.
                    
                    v0 = colors[k_idxs[0]] if k_idxs[0] != e_idx else new_c
                    v1 = colors[k_idxs[1]] if k_idxs[1] != e_idx else new_c
                    v2 = colors[k_idxs[2]] if k_idxs[2] != e_idx else new_c
                    v3 = colors[k_idxs[3]] if k_idxs[3] != e_idx else new_c
                    v4 = colors[k_idxs[4]] if k_idxs[4] != e_idx else new_c
                    v5 = colors[k_idxs[5]] if k_idxs[5] != e_idx else new_c
                    
                    d_new = 6
                    if v0 == v1: d_new -= 1
                    if v0 == v2: d_new -= 1
                    if v0 == v3: d_new -= 1
                    if v0 == v4: d_new -= 1
                    if v0 == v5: d_new -= 1
                    if v1 == v2: d_new -= 1
                    if v1 == v3: d_new -= 1
                    if v1 == v4: d_new -= 1
                    if v1 == v5: d_new -= 1
                    if v2 == v3: d_new -= 1
                    if v2 == v4: d_new -= 1
                    if v2 == v5: d_new -= 1
                    if v3 == v4: d_new -= 1
                    if v3 == v5: d_new -= 1
                    if v4 == v5: d_new -= 1
                    
                    new_viol = 1 if d_new < 5 else 0
                    
                    delta += new_viol - old_viol
                    
                new_bad = current_bad + delta
                
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_new_c = new_c
                    
                if new_bad == 0:
                    colors[e_idx] = new_c
                    return colors
                    
            colors[e_idx] = best_new_c
            current_bad = best_bad
            
            # If no improvement, accept a random move to escape local minima?
            # Or just continue.
            # If best_new_c == old_c, we are stuck on this edge.
            # We can try a random move on a random edge.
            if best_new_c == old_c:
                # Random perturbation
                if random.random() < 0.1:
                    e_idx2 = random.randint(0, E - 1)
                    colors[e_idx2] = random.randint(0, num_colors - 1)
                    current_bad = count_violations(colors)
                    
        return None

    # Try 14 colors first?
    # The problem says 13 <= f(16) <= 15.
    # Seed uses 17.
    # Let's try 14 first, then 15.
    
    # To increase chance of success, run multiple attempts with different seeds?
    # Or just one long run.
    
    # Let's try 14 colors.
    # If it fails, try 15.
    
    # Note: The random seed in local_search was fixed to 42.
    # I should use a different seed or time-based for diversity?
    # The task allows standard library.
    
    # Let's use a fixed seed for reproducibility in the "best" case, 
    # but maybe try a few seeds if time permits.
    
    # For now, one run with seed 42.
    random.seed(42)
    sol14 = repair_search_v2(14)
    if sol14 is not None:
        return sol14

    # If 14 fails, try 15
    # Reset seed? Or continue?
    # Let's use a different seed for the second attempt to explore different space.
    random.seed(123)
    sol15 = repair_search_v2(15)
    if sol15 is not None:
        return sol15
    
    # Fallback: Seed (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
