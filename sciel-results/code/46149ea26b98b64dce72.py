import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    all_masks = list(range(1 << n))
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        m = len(family)
        for i in range(m):
            mi = family[i]
            for j in range(i + 1, m):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    count += 1
                elif (mi & mj) == mj and mi != mj:
                    count += 1
        return count
    
    def score_candidate(cand, family):
        """Count how many sets in family are comparable with cand."""
        s = 0
        for fs in family:
            if (cand & fs) == cand and cand != fs:
                s += 1
            elif (cand & fs) == fs and cand != fs:
                s += 1
        return s
    
    def initial_structure_chain_star():
        """Chain + star hybrid: empty, all singletons, then build up."""
        family = []
        used = set()
        
        # Empty set
        family.append(0)
        used.add(0)
        
        # All 12 singletons
        for i in range(n):
            mask = 1 << i
            family.append(mask)
            used.add(mask)
        
        # Now greedily add sets to maximize comparability
        remaining = [m for m in all_masks if m not in used]
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                if cand in used:
                    continue
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family
    
    def initial_structure_pure_chain():
        """Start with a chain, then greedily add."""
        family = []
        used = set()
        
        # Chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,...,11}
        chain_mask = 0
        for k in range(n + 1):
            family.append(chain_mask)
            used.add(chain_mask)
            if k < n:
                chain_mask |= (1 << k)
        
        # Greedily add remaining
        remaining = [m for m in all_masks if m not in used]
        
        for _ in range(m - len(family)):
            if time.time() > deadline:
                break
            best = None
            best_score = -1
            for cand in remaining:
                if cand in used:
                    continue
                s = score_candidate(cand, family)
                if s > best_score:
                    best_score = s
                    best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
            remaining.remove(best)
        
        return family
    
    def local_search(family, time_limit):
        """Try to improve by swapping out poorly-connected sets using delta evaluation."""
        current_score = count_comparable(family)
        end_time = time.time() + time_limit
        m = len(family)
        
        # Precompute comparability matrix for current family
        def build_comp_matrix(fam):
            comp = [[0]*m for _ in range(m)]
            for i in range(m):
                for j in range(i+1, m):
                    mi, mj = fam[i], fam[j]
                    if (mi & mj) == mi and mi != mj:
                        comp[i][j] = comp[j][i] = 1
                    elif (mi & mj) == mj and mi != mj:
                        comp[i][j] = comp[j][i] = 1
            return comp
        
        comp_mat = build_comp_matrix(family)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used = set(family)
            
            for i in range(m):
                if time.time() > end_time:
                    return family
                
                old_mask = family[i]
                # Current contribution of i to total score
                score_without_i = sum(comp_mat[i])
                
                best_replacement = None
                best_gain = 0
                
                # Find best replacement using delta
                for cand in all_masks:
                    if cand in used:
                        continue
                    # Compute new contribution of cand to the family without i
                    new_contrib = 0
                    for j in range(m):
                        if j == i:
                            continue
                        mj = family[j]
                        if (cand & mj) == cand and cand != mj:
                            new_contrib += 1
                        elif (cand & mj) == mj and cand != mj:
                            new_contrib += 1
                    
                    gain = new_contrib - score_without_i
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = cand
                
                if best_replacement is not None and best_gain > 0:
                    family[i] = best_replacement
                    current_score += best_gain
                    improved = True
                    used = set(family)
                    # Rebuild comparison matrix
                    comp_mat = build_comp_matrix(family)
        
        return family
    
    def simulated_annealing(family, time_limit):
        """Simulated annealing with delta evaluation."""
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        used = set(family)
        
        end_time = time.time() + time_limit
        T = 10.0
        cooling = 0.995  # Faster cooling for more iterations
        min_T = 0.1
        
        random.seed(42)
        
        while time.time() < end_time and T > min_T:
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Generate candidate
            if random.random() < 0.7:
                new_mask = old_mask
                for _ in range(random.randint(1, 3)):
                    bit = random.randint(0, n - 1)
                    new_mask ^= (1 << bit)
            else:
                new_mask = random.randint(0, max_mask)
            
            if new_mask == old_mask:
                continue
            
            if new_mask in used:
                continue
            
            # Calculate delta
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                mj = family[j]
                old_c = 0
                if (old_mask & mj) == old_mask and old_mask != mj:
                    old_c = 1
                elif (old_mask & mj) == mj and old_mask != mj:
                    old_c = 1
                
                new_c = 0
                if (new_mask & mj) == new_mask and new_mask != mj:
                    new_c = 1
                elif (new_mask & mj) == mj and new_mask != mj:
                    new_c = 1
                
                delta += new_c - old_c
            
            # Accept?
            if delta > 0 or random.random() < (2.71828 ** (delta / T)):
                # Accept
                used.remove(old_mask)
                used.add(new_mask)
                family[i] = new_mask
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            
            T *= cooling
        
        return best_family, best_score
    
    # Try multiple initial structures and pick the best
    best_family = None
    best_score = -1
    
    # Structure 1: Chain + greedy
    fam1 = initial_structure_pure_chain()
    if len(fam1) == m:
        s1 = count_comparable(fam1)
        if s1 > best_score:
            best_score = s1
            best_family = fam1[:]
    
    if time.time() < deadline - 20:
        # Structure 2: Star + greedy
        fam2 = initial_structure_chain_star()
        if len(fam2) == m:
            s2 = count_comparable(fam2)
            if s2 > best_score:
                best_score = s2
                best_family = fam2[:]
    
    # Local search on the best family
    if best_family and time.time() < deadline - 10:
        time_remaining = deadline - 10 - time.time()
        best_family = local_search(best_family, time_remaining)
        final_score = count_comparable(best_family)
        if final_score > best_score:
            best_score = final_score
    
    # Simulated annealing polish
    if best_family and time.time() < deadline - 3:
        time_remaining = deadline - 3 - time.time()
        sa_family, sa_score = simulated_annealing(best_family, time_remaining)
        if sa_score > best_score:
            best_score = sa_score
            best_family = sa_family
    
    # Ensure exactly m sets
    if best_family is None or len(best_family) != m:
        # Fallback: use sorted masks
        best_family = sorted(all_masks, key=lambda x: (bin(x).count('1'), x))[:m]
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
    
    # Validate
    assert len(result) == m
    assert len(set(tuple(s) for s in result)) == m
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
            assert not isinstance(x, bool)
            assert 0 <= x <= 11
        assert len(s) == len(set(s))
    
    return result
