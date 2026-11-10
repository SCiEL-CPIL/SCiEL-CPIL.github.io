import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    n = 12
    m = 40
    
    # Precompute all subsets as bitmasks
    # all_masks[i] = bitmask for subset i
    # We'll work with bitmasks for speed
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        for i in range(len(family)):
            mi = family[i]
            for j in range(i + 1, len(family)):
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
    
    # Generate all 2^12 = 4096 subsets as bitmasks
    all_masks = list(range(1 << n))
    
    # Precompute popcount for sorting
    # Sort by (popcount, mask) for initial structure generation
    
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
        """Try to improve by swapping out poorly-connected sets."""
        current_score = count_comparable(family)
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used = set(family)
            
            for i in range(len(family)):
                if time.time() > end_time:
                    return family
                # Try removing family[i] and adding a better set
                without_i = family[:i] + family[i+1:]
                best_replacement = None
                best_gain = 0
                
                # Score of family without i
                score_without_i = count_comparable(without_i)
                
                # Find best replacement
                for cand in all_masks:
                    if cand in used:
                        continue
                    # Score with replacement
                    new_family = without_i + [cand]
                    new_score = count_comparable(new_family)
                    gain = new_score - current_score
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = cand
                
                if best_replacement is not None and best_gain > 0:
                    family[i] = best_replacement
                    current_score += best_gain
                    improved = True
                    used = set(family)
        
        return family
    
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
    
    if time.time() > deadline - 5:
        pass  # Not enough time for more
    else:
        # Structure 2: Star + greedy
        fam2 = initial_structure_chain_star()
        if len(fam2) == m:
            s2 = count_comparable(fam2)
            if s2 > best_score:
                best_score = s2
                best_family = fam2[:]
    
    # Local search on the best family
    if best_family and time.time() < deadline - 2:
        time_remaining = deadline - 2 - time.time()
        best_family = local_search(best_family, time_remaining)
        final_score = count_comparable(best_family)
        if final_score > best_score:
            best_score = final_score
    
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
