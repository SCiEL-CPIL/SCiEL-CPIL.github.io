"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 12
M = 40
DEADLINE = 85.0  # safety margin before 90s kill

def score(family):
    """Count comparable pairs in the family."""
    n = len(family)
    count = 0
    for i in range(n):
        si = family[i]
        si_set = set(si)
        for j in range(i+1, n):
            sj = family[j]
            sj_set = set(sj)
            if si_set < sj_set or sj_set < si_set:
                count += 1
    return count

def random_subset(size, rng):
    return sorted(rng.sample(range(N), size))

def initial_solution(rng):
    """Create an initial solution with subsets of sizes 4-7."""
    sizes = []
    # Use sizes 4, 5, 6, 7 with some distribution
    for _ in range(M):
        s = rng.choice([4, 5, 6, 7])
        sizes.append(s)
    
    family = []
    seen = set()
    attempts = 0
    while len(family) < M and attempts < 10000:
        s = sizes[len(family)]
        sub = tuple(random_subset(s, rng))
        if sub not in seen:
            seen.add(sub)
            family.append(list(sub))
        attempts += 1
    
    # If we couldn't get M distinct subsets, fill with empty set and singletons
    while len(family) < M:
        for s in range(0, N+1):
            for sub in combinations(range(N), s):
                if sub not in seen:
                    seen.add(sub)
                    family.append(list(sub))
                    break
            if len(family) >= M:
                break
        if len(family) >= M:
            break
    
    return family[:M]

def hill_climb(family, rng, deadline):
    """Improve the family using local search."""
    current_score = score(family)
    family = [list(s) for s in family]
    
    # Precompute sets for faster comparison
    family_sets = [set(s) for s in family]
    
    def score_with_replacement(fam, fam_sets, idx, new_sub):
        """Score if we replace element at idx with new_sub."""
        new_set = set(new_sub)
        count = 0
        for i in range(M):
            if i == idx:
                continue
            si = fam_sets[i]
            if new_set < si or si < new_set:
                count += 1
        return count
    
    # Generate candidate subsets to try
    all_candidates = []
    for s in range(0, N+1):
        for sub in combinations(range(N), s):
            all_candidates.append(list(sub))
    
    improved = True
    no_improve_count = 0
    max_no_improve = 5000
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count = 0
        
        # Try random swaps
        for _ in range(200):
            if time.time() >= deadline:
                break
            
            idx = rng.randrange(M)
            # Try replacing family[idx] with a random candidate
            new_sub = rng.choice(all_candidates)
            if set(new_sub) == family_sets[idx]:
                continue
            
            new_score = score_with_replacement(family, family_sets, idx, new_sub)
            
            if new_score > current_score:
                # Accept
                old_set = family_sets[idx]
                family[idx] = new_sub
                family_sets[idx] = set(new_sub)
                current_score = new_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            break
    
    return family, current_score

def construct():
    rng = random.Random(42)
    start_time = time.time()
    
    # Generate initial solution
    family = initial_solution(rng)
    
    # Hill climb
    family, score_val = hill_climb(family, rng, DEADLINE)
    
    # Final verification and conversion
    result = []
    seen = set()
    for s in family:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            result.append([int(x) for x in t])
    
    # Ensure exactly 40 distinct subsets
    if len(result) < M:
        all_subs = []
        for s in range(0, N+1):
            for sub in combinations(range(N), s):
                if tuple(sub) not in seen:
                    all_subs.append(list(sub))
        for sub in all_subs:
            if len(result) >= M:
                break
            t = tuple(sorted(sub))
            if t not in seen:
                seen.add(t)
                result.append([int(x) for x in t])
    
    result = result[:M]
    
    # Ensure types are correct
    final = []
    for s in result:
        final.append([int(x) for x in s])
    
    return final
# EVOLVE-BLOCK-END
