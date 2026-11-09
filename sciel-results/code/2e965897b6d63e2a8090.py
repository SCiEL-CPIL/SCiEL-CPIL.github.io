import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    total_sets = 1 << n  # 4096
    
    # Helper to count comparable pairs
    def count_comparables(family):
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    count += 1
        return count

    # Greedy initialization
    def greedy_init():
        family = []
        family_set = set()
        family.append(0)
        family_set.add(0)
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            for s in range(total_sets):
                if s in family_set:
                    continue
                gain = 0
                for f in family:
                    if (f & s) == f or (f & s) == s:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        return family

    family = greedy_init()
    current_score = count_comparables(family)
    best_family = family[:]
    best_score = current_score

    # Try MILP approach using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build comparability matrix: comp[i][j] = 1 if i and j are comparable (one is strict subset of other)
        # For efficiency, precompute for all pairs
        # comp[i][j] = 1 if (i & j) == i or (i & j) == j, and i != j
        
        # We want to maximize sum_{i<j} comp[i][j] * x_i * x_j
        # subject to sum x_i = m, x_i in {0,1}
        
        # This is a quadratic binary program. MILP handles linear problems.
        # We can linearize: introduce y_ij for each pair, y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1
        # Maximize sum comp[i][j] * y_ij
        # subject to sum x_i = m
        
        # Number of variables: 4096 x_i + (4096*4095/2) y_ij ≈ 4096 + 8.3M ≈ 8.3M variables
        # This is too large for MILP. Let's think of another approach.
        
        # Alternative: Since we only need to select 40 sets, we can use a different formulation.
        # Or we can use the fact that the problem has special structure.
        
        # Actually, let's try a different approach: use the MILP with just x variables
        # and use a linearization that's more efficient.
        
        # For m=40, the number of pairs among selected sets is at most C(40,2) = 780.
        # But we don't know which 40 will be selected.
        
        # Let's try a heuristic: use the greedy result as a starting point and do
        # local search with a better move structure.
        
        # Actually, let's try to use the comparability structure more directly.
        # Key insight: if we have a chain, all pairs in the chain are comparable.
        # So we want to find 40 sets that form as many chains as possible.
        
        # Let's do a more sophisticated local search:
        # 1. Start with greedy solution
        # 2. Try to improve by swapping sets
        # 3. Use a smarter candidate generation: prefer sets that are comparable to many existing sets
        
        # Precompute comparability counts for quick lookup
        # For each set s, store how many sets in the current family are comparable to s
        
        # Let's implement a better SA with:
        # 1. Biased candidate generation
        # 2. Multiple restarts
        # 3. Final local search polish
        
        # First, let's do a full local search polish on the greedy solution
        def local_search_polish(fam, max_rounds=10, time_limit=None):
            """Full-scan local search: for each position, try all possible replacements."""
            fam = fam[:]
            score = count_comparables(fam)
            improved = True
            rounds = 0
            while improved and rounds < max_rounds:
                improved = False
                rounds += 1
                if time_limit and time.time() > time_limit:
                    break
                for i in range(m):
                    old_s = fam[i]
                    best_new = old_s
                    best_delta = 0
                    # Try all possible replacements
                    for s in range(total_sets):
                        if s == old_s:
                            continue
                        # Check if s is already in fam (excluding position i)
                        is_dup = False
                        for j, f in enumerate(fam):
                            if j != i and f == s:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        # Calculate delta
                        count_new = 0
                        count_old = 0
                        for j, f in enumerate(fam):
                            if j == i:
                                continue
                            if (f & s) == f or (f & s) == s:
                                count_new += 1
                            if (f & old_s) == f or (f & old_s) == old_s:
                                count_old += 1
                        delta = count_new - count_old
                        if delta > best_delta:
                            best_delta = delta
                            best_new = s
                    if best_delta > 0:
                        fam[i] = best_new
                        score += best_delta
                        improved = True
            return fam, score
        
        # Polish the greedy solution
        polished_family, polished_score = local_search_polish(family, max_rounds=5, time_limit=deadline - 10)
        if polished_score > best_score:
            best_score = polished_score
            best_family = polished_family[:]
        
        # Now do SA with biased candidate generation
        def get_comparable_count(family, s, exclude_index=-1):
            count = 0
            for i, f in enumerate(family):
                if i == exclude_index:
                    continue
                if (f & s) == f or (f & s) == s:
                    count += 1
            return count
        
        # SA with multiple restarts
        T = 5.0
        T_min = 0.01
        alpha = 0.9998
        max_iter = 100000
        
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Start from best found so far (or random for diversity)
            if restart == 0:
                fam = best_family[:]
            else:
                fam = best_family[:]
                # Perturb: swap a few random elements
                for _ in range(restart * 2):
                    i = random.randint(0, m - 1)
                    j = random.randint(0, m - 1)
                    if i != j:
                        fam[i], fam[j] = fam[j], fam[i]
            
            cur_score = count_comparables(fam)
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                i = random.randint(0, m - 1)
                old_s = fam[i]
                
                # Biased candidate: pick from sets that are comparable to many existing sets
                # With probability 0.7, pick a comparable set; with 0.3, pick random
                if random.random() < 0.7:
                    # Find sets comparable to existing family members
                    candidates = set()
                    for f in fam:
                        if f == old_s:
                            continue
                        # All sets that are subsets or supersets of f
                        # Subsets: all masks that are subset of f
                        sub = f
                        while True:
                            candidates.add(sub)
                            if sub == 0:
                                break
                            sub = (sub - 1) & f
                        # Supersets: all masks that are superset of f
                        # This is harder. Let's just use random for now.
                        # Actually, let's just pick random from a smaller pool.
                    
                    # Simpler: pick a random set that is comparable to at least one family member
                    found = False
                    for _ in range(20):  # Try up to 20 random picks
                        new_s = random.randint(0, total_sets - 1)
                        if new_s == old_s:
                            continue
                        # Check if comparable to at least one family member
                        for f in fam:
                            if f != old_s and ((f & new_s) == f or (f & new_s) == new_s):
                                found = True
                                break
                        if found:
                            break
                    
                    if not found:
                        new_s = random.randint(0, total_sets - 1)
                else:
                    new_s = random.randint(0, total_sets - 1)
                
                if new_s == old_s:
                    continue
                
                # Check duplicate
                is_dup = False
                for j, f in enumerate(fam):
                    if j != i and f == new_s:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Calculate delta
                count_new = 0
                count_old = 0
                for j, f in enumerate(fam):
                    if j == i:
                        continue
                    if (f & new_s) == f or (f & new_s) == new_s:
                        count_new += 1
                    if (f & old_s) == f or (f & old_s) == old_s:
                        count_old += 1
                
                delta = count_new - count_old
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    fam[i] = new_s
                    cur_score += delta
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = fam[:]
                
                T *= alpha
                if T < T_min:
                    T = T_min
        
        # Final polish
        polished_family, polished_score = local_search_polish(best_family, max_rounds=3, time_limit=deadline - 2)
        if polished_score > best_score:
            best_score = polished_score
            best_family = polished_family[:]
        
        family = best_family
    
    except Exception as e:
        # Fallback to original SA if anything goes wrong
        T = 10.0
        T_min = 0.01
        alpha = 0.9995
        max_iter = 200000
        
        best_family = family[:]
        best_score = current_score
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            i = random.randint(0, m - 1)
            old_s = family[i]
            new_s = random.randint(0, total_sets - 1)
            
            if new_s == old_s:
                continue
            
            is_dup = False
            for j, f in enumerate(family):
                if j != i and f == new_s:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            count_new = 0
            count_old = 0
            for j, f in enumerate(family):
                if j == i:
                    continue
                if (f & new_s) == f or (f & new_s) == new_s:
                    count_new += 1
                if (f & old_s) == f or (f & old_s) == old_s:
                    count_old += 1
            
            delta = count_new - count_old
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                family[i] = new_s
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        family = best_family

    # Convert bitmasks to lists
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
    
    return result
