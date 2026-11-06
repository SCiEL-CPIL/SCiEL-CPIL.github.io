import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Fallback: Paley tournament on 83 vertices
    def get_paley_83():
        p = 83
        squares = {x * x % p for x in range(1, p)}
        return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
    
    # Try to find a smaller valid tournament
    best_n = 83
    best_orientation = None
    paley = get_paley_83()
    best_orientation = paley['orientation']
    
    # We will try n from 82 down to 31
    # But we should prioritize larger n first? Or smaller?
    # Smaller n is harder to find. Larger n is easier.
    # Since we want to minimize n, we should try to find a valid one for small n.
    # But if we start with n=31, it might take a long time to find a valid one (if it exists) or prove it doesn't.
    # Actually, we just need to find ONE valid tournament for a given n.
    # If we find one for n=67, score is 83/67.
    # If we find one for n=50, score is 83/50.
    
    # Let's iterate n from 82 down to 31.
    # If we find a valid one, we record it and continue to smaller n?
    # No, if we find a valid one for n=82, we can return it immediately?
    # No, we want the MINIMUM n.
    # So we should try to find a valid tournament for the smallest n possible.
    # But we don't know which n has a valid tournament.
    # We know 83 works.
    # We know 67 works (from problem statement).
    # So we can try n=67 first?
    # If we find a valid tournament for n=67, we can return it.
    # But maybe n=60 works?
    # To be safe and efficient, we can try a few specific values known to be close to the bound, or just sweep down.
    
    # Let's try a few random restarts for each n.
    # To maximize the chance of finding a small n, we should try smaller n first?
    # No, smaller n is much harder to satisfy. The density of constraints is higher.
    # It is easier to find a valid tournament for larger n.
    # So if we sweep from 82 down, we will likely find a valid one quickly at n=82, 81, etc.
    # But those give a small improvement (83/82 = 1.01).
    # We want a BIG improvement.
    # The problem states "smallest known has 67".
    # So there is a high probability that a valid tournament exists for n=67.
    # Let's try to find one for n=67, then 66, etc.
    # But if we get stuck at n=67, we might waste time.
    
    # Alternative: Try to find valid tournaments for a set of candidate n values.
    # Candidates: 67, 60, 50, 40, 31.
    # If we find one for 67, return it.
    # If not, try 60, etc.
    # But what if 82 is easy to find and 67 is hard?
    # The "hardness" is not monotonic in a simple way for random search.
    
    # Let's stick to a simple strategy:
    # 1. Try n=67 with many restarts.
    # 2. If found, return.
    # 3. If not found within time, try n=70, 75, 80...
    # 4. If any found, return the smallest n found.
    # 5. If none found, return Paley 83.
    
    # Actually, the most robust way to get *some* improvement is to find *any* valid tournament with n < 83.
    # Let's try to find a valid tournament for n=82, 81, 80...
    # If we find one for 82, score is 1.012.
    # If we find one for 67, score is 1.238.
    # The jump to 67 is much better.
    
    # Let's define a function to check if a tournament is valid.
    def is_valid(n, orientation):
        # orientation is a list of bits for i<j
        # Reconstruct out-masks
        out = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    out[i] |= (1 << j)
                else:
                    out[j] |= (1 << i)
                idx += 1
        
        # Check all 4-sets
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if any of a,b,c,d dominates the other 3
                        # a dominates b,c,d if out[a] has bits for b,c,d
                        mask_bcd = (1 << b) | (1 << c) | (1 << d)
                        if (out[a] & mask_bcd) == mask_bcd:
                            continue
                        mask_acd = (1 << a) | (1 << c) | (1 << d)
                        if (out[b] & mask_acd) == mask_acd:
                            continue
                        mask_abd = (1 << a) | (1 << b) | (1 << d)
                        if (out[c] & mask_abd) == mask_abd:
                            continue
                        mask_abc = (1 << a) | (1 << b) | (1 << c)
                        if (out[d] & mask_abc) == mask_abc:
                            continue
                        return False
        return True

    def count_bad(n, orientation):
        out = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    out[i] |= (1 << j)
                else:
                    out[j] |= (1 << i)
                idx += 1
        
        bad_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        mask_bcd = (1 << b) | (1 << c) | (1 << d)
                        if (out[a] & mask_bcd) == mask_bcd:
                            continue
                        mask_acd = (1 << a) | (1 << c) | (1 << d)
                        if (out[b] & mask_acd) == mask_acd:
                            continue
                        mask_abd = (1 << a) | (1 << b) | (1 << d)
                        if (out[c] & mask_abd) == mask_abd:
                            continue
                        mask_abc = (1 << a) | (1 << b) | (1 << c)
                        if (out[d] & mask_abc) == mask_abc:
                            continue
                        bad_count += 1
        return bad_count

    def local_search(n, max_iters=10000):
        # Random init
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        
        # Helper to get index
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
            # Formula: sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
            # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
            # Wait, let's use a simpler mapping or precomputed table.
            pass
        
        # Precompute indices
        idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_map[(i, j)] = idx
                idx += 1

        current_bad = count_bad(n, orientation)
        if current_bad == 0:
            return orientation
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx_ij = idx_map[(i, j)]
            
            # Flip edge
            orientation[idx_ij] = 1 - orientation[idx_ij]
            new_bad = count_bad(n, orientation)
            
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad == 0:
                    return orientation
            else:
                # Revert
                orientation[idx_ij] = 1 - orientation[idx_ij]
                
        return None

    # Try specific n values
    # We know 83 works.
    # Let's try to find a valid tournament for n < 83.
    # We'll try a few values.
    target_ns = [67, 60, 50, 40, 35, 31]
    # Also try some values close to 83 in case smaller ones are too hard
    target_ns.extend([82, 81, 80])
    
    # Sort by n ascending? No, we want the smallest valid n.
    # But finding a valid one for small n is harder.
    # Let's try the known bound 67 first.
    # If we find it, we return it.
    # If not, we try others.
    
    found_n = None
    found_orient = None
    
    for n in target_ns:
        if n >= 83:
            continue
        if time.time() > deadline:
            break
            
        # Try a few restarts for this n
        for restart in range(5):
            if time.time() > deadline:
                break
            result = local_search(n, max_iters=2000)
            if result is not None:
                # Verify
                if is_valid(n, result):
                    found_n = n
                    found_orient = result
                    break
        
        if found_n is not None:
            break
            
    if found_n is not None:
        return {'n': found_n, 'orientation': found_orient}
    else:
        return paley

# EVOLVE-BLOCK-END
