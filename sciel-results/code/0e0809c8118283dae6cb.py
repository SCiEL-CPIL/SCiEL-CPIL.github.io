import time
import random
import sys

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # Precompute comparability
    # comparables[s] is a set of masks t such that s is a strict subset of t or t is a strict subset of s
    comparables = [set() for _ in range(total_masks)]
    
    # Efficiently build comparables
    # For each s, enumerate supersets
    for s in range(total_masks):
        complement = (~s) & (total_masks - 1)
        u = complement
        while u:
            t = s | u
            comparables[s].add(t)
            comparables[t].add(s)
            u = (u - 1) & complement

    def calc_score(fam):
        count = 0
        n_fam = len(fam)
        for i in range(n_fam):
            for j in range(i + 1, n_fam):
                if fam[i] in comparables[fam[j]]:
                    count += 1
        return count

    def local_search(fam, fam_set, candidate_pool, deadline):
        current_family = fam[:]
        current_fam_set = fam_set.copy()
        current_score = calc_score(current_family)
        
        # Precompute degrees for current family
        def get_degs(fam):
            degs = [0] * len(fam)
            for i in range(len(fam)):
                for j in range(len(fam)):
                    if i != j and fam[i] in comparables[fam[j]]:
                        degs[i] += 1
            return degs

        improved = True
        while improved and time.time() < deadline:
            improved = False
            degs = get_degs(current_family)
            
            # Try swapping each element with a candidate from the pool
            # To speed up, we can sample the pool or iterate if pool is small
            # Pool size for C of size 6 is ~1600. 40 * 1600 = 64k checks per pass.
            # 64k checks * (12 comparisons) = ~768k ops. Very fast.
            
            indices = list(range(len(current_family)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                A = current_family[i]
                dA = degs[i]
                
                best_gain = 0
                best_B = -1
                
                # Iterate through candidate pool
                # We can optimize by precomputing comparability of pool elements with current_family?
                # But pool is static, family changes.
                # Just compute on the fly.
                
                for B in candidate_pool:
                    if B in current_fam_set:
                        continue
                    
                    # Calculate gain of replacing A with B
                    # Gain = (edges(B, Family\A)) - (edges(A, Family\A))
                    # edges(A, Family\A) = dA
                    # edges(B, Family\A) = sum(1 for x in Family if x != A and x in comparables[B])
                    
                    count_comp = 0
                    for x in current_family:
                        if x == A:
                            continue
                        if x in comparables[B]:
                            count_comp += 1
                    
                    gain = count_comp - dA
                    if gain > best_gain:
                        best_gain = gain
                        best_B = B
                
                if best_B != -1 and best_gain > 0:
                    current_fam_set.remove(A)
                    current_fam_set.add(best_B)
                    current_family[i] = best_B
                    current_score += best_gain
                    improved = True
                    # Restart the pass to recompute degrees?
                    # Or just continue with stale degrees?
                    # Stale degrees might lead to suboptimal swaps, but restarting is expensive.
                    # Let's restart the inner loop to be safe, or just break and restart outer while.
                    break
            
            # If we broke due to a swap, the outer while will restart and recompute degrees.
            # If no swap happened, improved remains False and we exit.

        return current_family, current_score

    best_family = None
    best_score_val = -1

    # Strategy: Try several centers C
    # Sizes 5, 6, 7 are good candidates
    center_sizes = [5, 6, 7]
    
    # Pre-generate some random centers
    centers = []
    for size in center_sizes:
        for _ in range(20): # 20 random centers per size
            c_mask = 0
            while True:
                c_mask = random.randint(0, total_masks - 1)
                if bin(c_mask).count('1') == size:
                    centers.append(c_mask)
                    break

    random.shuffle(centers)

    for C in centers:
        if time.time() > deadline:
            break
            
        # Build candidate pool: all subsets and supersets of C (including C? No, C is fixed)
        # Actually, we want to select 40 sets. One is C.
        # The other 39 are from Subsets(C) \ {C} and Supersets(C) \ {C}.
        
        sub_sets = []
        comp = (~C) & (total_masks - 1)
        # Enumerate subsets of C
        u = C
        while True:
            if u != C:
                sub_sets.append(u)
            if u == 0:
                break
            u = (u - 1) & C
            
        sup_sets = []
        # Enumerate supersets of C
        u = comp
        while True:
            s = C | u
            if s != C:
                sup_sets.append(s)
            if u == 0:
                break
            u = (u - 1) & comp
        
        candidate_pool = sub_sets + sup_sets
        
        # Initialize family with C and 39 random elements from pool
        if len(candidate_pool) < 39:
            continue
            
        init_family = [C] + random.sample(candidate_pool, 39)
        init_set = set(init_family)
        
        fam, score = local_search(init_family, init_set, candidate_pool, deadline)
        
        if score > best_score_val:
            best_score_val = score
            best_family = fam[:]

    # Fallback if no center worked (shouldn't happen)
    if best_family is None:
        best_family = [i for i in range(40)]
        
    # Convert to lists
    return _convert(best_family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
