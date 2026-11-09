import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 12
    m = 40
    
    # All subsets as bitmasks
    all_subsets = list(range(1 << n))
    
    # Precompute subset relationships: for each pair, check if one is subset of other
    # Use bit operations: A is subset of B iff (A & B) == A
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                a, b = family[i], family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count
    
    # Initial solution: greedy construction
    # Start with empty set, then greedily add sets that maximize new comparable pairs
    def greedy_initial():
        family = [0]  # empty set
        remaining = set(all_subsets) - {0}
        for _ in range(m - 1):
            best_set = None
            best_gain = -1
            for s in remaining:
                # Count how many existing sets are comparable with s
                gain = 0
                for e in family:
                    if (s & e) == s and s != e:
                        gain += 1
                    elif (s & e) == e and s != e:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_set = s
            family.append(best_set)
            remaining.discard(best_set)
        return family
    
    # Better initial: use a structured approach
    # Build chains and merge
    def structured_initial():
        # Create multiple chains and pick from them
        # Chain 1: add elements in order 0,1,2,...,11
        # But we need 40 distinct sets
        
        # Strategy: take all sets of size <= k for some k, then fill
        # C(12,0)+C(12,1)+C(12,2)+C(12,3) = 1+12+66+220 = 299 > 40
        # So take all sets of size 0,1,2 (1+12+66=79) - too many
        # Take size 0,1 (13 sets), then 27 from size 2
        
        # Actually, let's use a different strategy:
        # Take a long "path" in the lattice
        # Start with {}, add 0, add 1, ..., add 11 (13 sets)
        # Then for each, also include some supersets
        
        # Simpler: use random restarts with local search
        return None
    
    # Local search: swap sets to improve score
    def local_search(family, time_limit=5.0):
        family = list(family)
        current_score = count_comparable(family)
        best_score = current_score
        best_family = list(family)
        
        start = time.time()
        no_improve_count = 0
        
        while time.time() - start < time_limit and no_improve_count < 500:
            # Try random swaps
            i, j = random.sample(range(m), 2)
            old_i = family[i]
            
            # Pick a new set not in family
            new_set = random.choice(all_subsets)
            if new_set in family:
                continue
            
            family[i] = new_set
            new_score = count_comparable(family)
            
            if new_score > best_score:
                best_score = new_score
                best_family = list(family)
                no_improve_count = 0
            else:
                family[i] = old_i
                no_improve_count += 1
        
        return best_family, best_score
    
    # Multi-start local search
    best_family = None
    best_score = -1
    
    # Start 1: greedy
    fam1 = greedy_initial()
    score1 = count_comparable(fam1)
    best_family = fam1
    best_score = score1
    
    # Try structured starts
    # Start with all small sets
    small_sets = []
    for r in range(0, 4):
        for combo in combinations(range(n), r):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            small_sets.append(mask)
    
    # Take first 40 from small_sets (sorted by size)
    small_sets.sort(key=lambda x: (bin(x).count('1'), x))
    if len(small_sets) >= m:
        fam2 = small_sets[:m]
        score2 = count_comparable(fam2)
        if score2 > best_score:
            best_score = score2
            best_family = fam2
    
    # Random restarts with local search
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Random initial family
        random.shuffle(all_subsets)
        fam = all_subsets[:m]
        score = count_comparable(fam)
        
        if score > best_score:
            best_score = score
            best_family = fam
        
        # Local search
        improved_fam, improved_score = local_search(fam, time_limit=2.0)
        if improved_score > best_score:
            best_score = improved_score
            best_family = improved_fam
        
        if time.time() > deadline - 10:
            break
    
    # Final polish: try more swaps on best family
    if time.time() < deadline - 5:
        final_fam, final_score = local_search(best_family, time_limit=3.0)
        if final_score > best_score:
            best_score = final_score
            best_family = final_fam
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        lst = [b for b in range(n) if (mask >> b) & 1]
        result.append(lst)
    
    return result
