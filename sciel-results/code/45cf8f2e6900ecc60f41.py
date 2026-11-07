import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    def count_comparables(family_masks):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        L = len(family_masks)
        for i in range(L):
            ai = family_masks[i]
            for j in range(i + 1, L):
                bj = family_masks[j]
                if (ai & ~bj) == 0 or (bj & ~ai) == 0:
                    cnt += 1
        return cnt
    
    # 1. Initialize with the longest chain (length 13)
    family = []
    cur = 0
    family.append(0)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    used = set(family)
    
    # 2. Greedily add sets that maximize comparable pairs with current family
    while len(family) < m:
        best_gain = -1
        best_mask = None
        
        for s in all_masks:
            if s in used:
                continue
            
            gain = 0
            for f in family:
                if (s & ~f) == 0 or (f & ~s) == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        
        if best_mask is None:
            break
            
        family.append(best_mask)
        used.add(best_mask)
    
    # 3. Local Search (Hill Climbing) with fast delta computation
    best_family = family[:]
    best_score = count_comparables(best_family)
    
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Precompute comparability: comp[a][b] = 1 if a is strict subset of b or vice versa
    # This is a 4096x4096 boolean matrix - too large. Instead, use bit operations on the fly.
    # But we can precompute for each mask, which other masks it's comparable with.
    # Actually, let's just use the bit trick efficiently.
    
    # Fast local search: for each position i, compute gain of replacing family[i] with s
    # gain(s, i) = num_comparables(s, family without i) - num_comparables(family[i], family without i)
    
    # Precompute: for current family, comp_count[i] = number of j != i where family[i] and family[j] are comparable
    # Then for candidate s: new_comp = sum over j != i of is_comparable(s, family[j])
    # delta = new_comp - comp_count[i]
    
    def compute_comp_counts(fam):
        """For each i, count how many j != i are comparable with fam[i]."""
        L = len(fam)
        counts = [0] * L
        for i in range(L):
            ai = fam[i]
            for j in range(L):
                if j == i:
                    continue
                bj = fam[j]
                if (ai & ~bj) == 0 or (bj & ~ai) == 0:
                    counts[i] += 1
        return counts
    
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        fam_set = set(best_family)
        comp_counts = compute_comp_counts(best_family)
        
        for i in range(m):
            if time.time() > deadline:
                break
            
            old_mask = best_family[i]
            old_comp = comp_counts[i]
            
            best_gain = 0
            best_repl = None
            
            for s in all_masks:
                if s in fam_set:
                    continue
                
                new_comp = 0
                for j in range(m):
                    if j == i:
                        continue
                    if (s & ~best_family[j]) == 0 or (best_family[j] & ~s) == 0:
                        new_comp += 1
                
                delta = new_comp - old_comp
                if delta > best_gain:
                    best_gain = delta
                    best_repl = s
            
            if best_repl is not None and best_gain > 0:
                best_family[i] = best_repl
                fam_set.discard(old_mask)
                fam_set.add(best_repl)
                improved = True
                # Recompute comp_counts after change (or update incrementally)
                comp_counts = compute_comp_counts(best_family)
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
