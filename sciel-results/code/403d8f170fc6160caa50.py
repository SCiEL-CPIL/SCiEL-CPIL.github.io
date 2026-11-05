import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Precompute 4-AP conflicts
    # conflicts[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            # For a: conflicts are (b, c, e)
            conflicts[a].append((b, c, e))
            # For b: conflicts are (a, c, e)
            conflicts[b].append((a, c, e))
            # For c: conflicts are (a, b, e)
            conflicts[c].append((a, b, e))
            # For e: conflicts are (a, b, c)
            conflicts[e].append((a, b, c))

    # 2. Precompute 3-AP pairs for each element
    # pairs_3ap[x] is a list of pairs (u, v) such that {x, u, v} is a 3-AP
    # We need to be careful about direction. 
    # A 3-AP is defined by (start, diff). The set is {s, s+d, s+2d}.
    # For a given x, it can be the start, middle, or end.
    # Let's just store the other two elements.
    pairs_3ap = [[] for _ in range(N)]
    
    for s in range(N):
        max_d = (N - 1 - s) // 2
        for d in range(1, max_d + 1):
            m = s + d
            e = s + 2 * d
            # The set is {s, m, e}
            # For s: partners are m, e
            pairs_3ap[s].append((m, e))
            # For m: partners are s, e
            pairs_3ap[m].append((s, e))
            # For e: partners are s, m
            pairs_3ap[e].append((s, m))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if (mask >> i) & 1:
                for (u, v) in pairs_3ap[i]:
                    if (mask >> u) & 1 and (mask >> v) & 1:
                        # Each 3-AP is counted 3 times (once for each element)
                        # We can just count and divide by 3, or use a set of tuples.
                        # For speed in delta calculation, we just need the change.
                        # For absolute count, let's just count pairs and divide by 3.
                        count += 1
        return count // 3

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        for (u, v) in pairs_3ap[x]:
            if (mask >> u) & 1 and (mask >> v) & 1:
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        # Same as add, because the APs broken are exactly the ones formed by adding x to the rest
        return delta_3aps_add(mask, x)

    # 3. Initialization: Residue Class based
    def init_residue():
        # Try different moduli and residue sets
        # A 4-AP modulo m is a set of 4 residues in AP.
        # We want to pick a set of residues R such that no 4 residues in R form a 4-AP mod m.
        # And ideally, many 3 residues in R form a 3-AP mod m.
        
        best_mask = 0
        best_score = -1
        
        for m in [3, 4, 5, 6, 7, 8, 9, 10, 12, 16]:
            # Generate all 4-APs modulo m
            bad_sets_4 = set()
            for a in range(m):
                for d in range(1, m):
                    b = (a + d) % m
                    c = (a + 2*d) % m
                    e = (a + 3*d) % m
                    bad_sets_4.add(frozenset([a, b, c, e]))
            
            # Try random subsets of residues of size ~ m/2
            for _ in range(50):
                res_size = random.randint(m//3, m//2)
                residues = random.sample(range(m), res_size)
                res_set = frozenset(residues)
                
                # Check if this residue set contains a 4-AP mod m
                is_bad = False
                for bs in bad_sets_4:
                    if bs.issubset(res_set):
                        is_bad = True
                        break
                if is_bad:
                    continue
                
                # Construct mask
                mask = 0
                for i in range(N):
                    if i % m in residues:
                        mask |= (1 << i)
                
                # Verify no 4-AP in [1, N] (mod m check is necessary but not sufficient for small N? 
                # Actually, if no 4-AP mod m, then no 4-AP in integers? 
                # Yes, if a, a+d, a+2d, a+3d is a 4-AP in integers, their residues mod m form a 4-AP in Z_m.
                # So checking mod m is sufficient to guarantee no 4-AP in [1, N] if we pick residues carefully?
                # Wait. If residues form a 4-AP mod m, it doesn't mean they form a 4-AP in integers.
                # But if integers form a 4-AP, residues MUST form a 4-AP mod m.
                # So if we exclude all residue sets that contain a 4-AP mod m, we guarantee no integer 4-AP.
                
                score = count_3aps(mask)
                if score > best_score:
                    best_score = score
                    best_mask = mask
        
        if best_mask == 0:
            # Fallback to random valid set
            mask = 0
            elems = list(range(N))
            random.shuffle(elems)
            for x in elems:
                if is_valid_add(mask, x):
                    mask |= (1 << x)
            return mask
            
        return best_mask

    # 4. Local Search: Greedy Hill Climbing
    def local_search(mask, score):
        improved = True
        while improved:
            improved = False
            # Try all single bit flips
            best_delta = 0
            best_move = None
            best_new_mask = mask
            best_new_score = score
            
            # Try adding
            for x in range(N):
                if not (mask >> x) & 1:
                    if is_valid_add(mask, x):
                        d = delta_3aps_add(mask, x)
                        if d > best_delta:
                            best_delta = d
                            best_move = ('add', x)
                            best_new_mask = mask | (1 << x)
                            best_new_score = score + d
            
            # Try removing
            for x in range(N):
                if (mask >> x) & 1:
                    d = -delta_3aps_remove(mask, x)
                    if d > best_delta:
                        best_delta = d
                        best_move = ('remove', x)
                        best_new_mask = mask & ~(1 << x)
                        best_new_score = score + d
            
            # Try swapping (remove x, add y)
            # This is more expensive. We can limit this to a few random swaps or skip in tight loops.
            # Let's do a limited number of random swaps to keep it fast.
            for _ in range(50):
                x = random.randint(0, N-1)
                if not (mask >> x) & 1: continue
                y = random.randint(0, N-1)
                if (mask >> y) & 1: continue
                
                temp_mask = mask & ~(1 << x)
                if is_valid_add(temp_mask, y):
                    d_rem = delta_3aps_remove(mask, x)
                    d_add = delta_3aps_add(temp_mask, y)
                    d = d_add - d_rem
                    if d > best_delta:
                        best_delta = d
                        best_move = ('swap', x, y)
                        best_new_mask = temp_mask | (1 << y)
                        best_new_score = score + d
            
            if best_delta > 0:
                mask = best_new_mask
                score = best_new_score
                improved = True
                # Optional: break if too slow? No, local search is fast.
                
        return mask, score

    # 5. Iterated Local Search
    best_mask = init_residue()
    best_score = count_3aps(best_mask)
    
    current_mask = best_mask
    current_score = best_score
    
    # Run ILS
    while time.time() < deadline - 5.0:
        # Local search to convergence
        current_mask, current_score = local_search(current_mask, current_score)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        # Perturb
        # Randomly flip k bits
        k = random.randint(2, 5)
        new_mask = current_mask
        
        # Ensure we don't create 4-APs immediately? 
        # Better to just flip and then repair? Or just flip and let local search fix it?
        # If we create a 4-AP, the set is invalid. 
        # The local search function assumes the input is valid? 
        # My local_search only checks validity for ADD. It assumes REMOVE is always valid.
        # It does NOT check if the initial mask is valid.
        # If I perturb and create a 4-AP, the subsequent "add" checks will be flawed?
        # is_valid_add checks if adding x creates a 4-AP. If the mask ALREADY has a 4-AP, 
        # is_valid_add might return True for some x, but the set is still invalid.
        # So we must ensure the perturbed mask is valid, or repair it.
        
        # Strategy: Randomly remove k elements, then try to add back different elements greedily?
        # Or: Randomly flip k bits, then if invalid, remove elements from 4-APs until valid.
        
        # Simpler Perturbation:
        # 1. Randomly remove r elements.
        # 2. Randomly add a elements (if valid).
        
        r = random.randint(1, 3)
        elems_in_mask = [i for i in range(N) if (current_mask >> i) & 1]
        elems_not_in_mask = [i for i in range(N) if not (current_mask >> i) & 1]
        
        if not elems_in_mask:
            continue
            
        removed = random.sample(elems_in_mask, min(r, len(elems_in_mask)))
        for x in removed:
            new_mask &= ~(1 << x)
            
        # Try to add back some elements
        for _ in range(r):
            if not elems_not_in_mask: break
            y = random.choice(elems_not_in_mask)
            if is_valid_add(new_mask, y):
                new_mask |= (1 << y)
        
        # If new_mask is invalid (shouldn't be if we only added validly to a valid set), 
        # but wait: removing elements from a valid set keeps it valid. Adding validly keeps it valid.
        # So new_mask is always valid.
        
        # Calculate score of new_mask
        # We can update score incrementally or just recalculate.
        # Recalculating is O(N * avg_pairs). N=256, avg_pairs ~ 50. 256*50 = 12800 ops. Fast enough.
        new_score = count_3aps(new_mask)
        
        # Accept if better or with some probability? 
        # ILS usually accepts unconditionally and relies on local search to recover.
        # But if new_score is much worse, we might waste time.
        # Let's accept if new_score >= current_score - 10 (allow some degradation)
        if new_score >= current_score - 10:
            current_mask = new_mask
            current_score = new_score
        else:
            # Keep current
            pass

    # Final local search on best
    final_mask, final_score = local_search(best_mask, best_score)
    if final_score > best_score:
        best_mask = final_mask
        best_score = final_score

    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)

