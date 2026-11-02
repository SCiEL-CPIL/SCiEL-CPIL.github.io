"""Football pool covering code K_3(7,1).
Strategy: Vectorized genetic algorithm with NumPy for fast coverage tracking.
Uses a precomputed ball membership matrix for O(1) coverage updates.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0


def construct(q, n, radius):
    start = time.time()
    total = q ** n  # 2187 for q=3, n=7
    
    def digits_of(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % q
            x //= q
        return d
    
    power = [1] * n
    for i in range(1, n):
        power[i] = power[i - 1] * q
    
    # Precompute ball membership as a NumPy matrix: mem[c, w] = 1 if w in ball(c)
    # mem shape: (total, total)
    mem = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        d = digits_of(c)
        # The word itself
        mem[c, c] = 1
        # Words at distance 1
        for pos in range(n):
            cur = d[pos]
            for v in range(q):
                if v == cur:
                    continue
                w = c + (v - cur) * power[pos]
                mem[c, w] = 1
    
    def time_left():
        return start + _DEADLINE - _MARGIN - time.time()
    
    def build_cov(sel):
        """Build coverage array from selected codewords."""
        if not sel:
            return np.zeros(total, dtype=np.int32)
        return mem[np.array(sel, dtype=np.int64)].sum(axis=0).astype(np.int32)
    
    def is_valid(cov):
        return np.min(cov) >= 1
    
    def greedy_init(seed=None):
        """Greedy initialization with random tie-breaking."""
        rng = np.random.RandomState(seed)
        uncovered = np.ones(total, dtype=bool)
        selected = []
        while uncovered.any():
            if time_left() < 1.0:
                break
            # Count how many uncovered words each candidate covers
            counts = mem[:, uncovered].sum(axis=1)
            # Among candidates with max count, pick randomly
            max_cnt = np.max(counts)
            if max_cnt == 0:
                break
            candidates = np.where(counts == max_cnt)[0]
            best_idx = int(rng.choice(candidates))
            selected.append(best_idx)
            uncovered[mem[best_idx, :]] = False
        return selected
    
    def local_search(sel, max_iters=200):
        """Local search: try removals and swaps."""
        if not sel:
            return sel
        cov = build_cov(sel)
        best_size = len(sel)
        best_sel = list(sel)
        sel_set = set(sel)
        
        for iteration in range(max_iters):
            if time_left() < 2.0:
                break
            improved = False
            
            # Pass 1: Remove redundant codewords
            for k in range(len(sel) - 1, -1, -1):
                c = sel[k]
                # Check if any word in ball(c) has cov==1
                ball_words = np.where(mem[c, :])[0]
                if np.min(cov[ball_words]) > 1:
                    # Redundant
                    cov[ball_words] -= 1
                    sel.pop(k)
                    sel_set.discard(c)
                    improved = True
                    if len(sel) < best_size:
                        best_size = len(sel)
                        best_sel = list(sel)
            
            if not improved and time_left() < 5.0:
                break
            
            # Pass 2: Try swaps
            if time_left() < 5.0:
                break
            
            # Rebuild cov if changed
            cov = build_cov(sel)
            sel_set = set(sel)
            
            for i in range(len(sel)):
                if time_left() < 2.0:
                    break
                ci = sel[i]
                # Find words uniquely covered by ci
                ball_words = np.where(mem[ci, :])[0]
                uniq = ball_words[cov[ball_words] == 1]
                if len(uniq) == 0:
                    continue
                
                # Try to find a replacement j that covers all unique words
                # Use vectorized check: for each j, check if it covers all uniq
                # j covers all uniq if mem[j, uniq].all()
                # But we need to check this efficiently
                # For each candidate j not in sel_set, check if it covers all uniq
                candidates = np.setdiff1d(np.arange(total), np.array(list(sel_set), dtype=np.int64))
                if len(candidates) == 0:
                    continue
                
                # Check which candidates cover all unique words
                # mem[candidates, uniq] has shape (len(candidates), len(uniq))
                # A candidate covers all uniq if all entries in its row are 1
                cover_all = mem[candidates, uniq].all(axis=1)
                good_candidates = candidates[cover_all]
                
                if len(good_candidates) == 0:
                    continue
                
                # Try each good candidate (limit to first few for speed)
                for j in good_candidates[:10]:
                    if time_left() < 1.0:
                        break
                    j = int(j)
                    # Perform swap: remove ci, add j
                    new_sel = list(sel)
                    new_sel[i] = j
                    ncov = build_cov(new_sel)
                    
                    # Try removals after swap
                    changed = True
                    while changed:
                        changed = False
                        for k in range(len(new_sel) - 1, -1, -1):
                            c2 = new_sel[k]
                            ball_words2 = np.where(mem[c2, :])[0]
                            if np.min(ncov[ball_words2]) > 1:
                                ncov[ball_words2] -= 1
                                new_sel.pop(k)
                                changed = True
                    
                    if len(new_sel) < len(sel):
                        sel = new_sel
                        cov = ncov
                        sel_set = set(sel)
                        improved = True
                        if len(sel) < best_size:
                            best_size = len(sel)
                            best_sel = list(sel)
                        break
            # End swap pass
        
        return best_sel if len(best_sel) < len(sel) else sel
    
    def genetic_algorithm(pop_size=50, generations=50, seed=None):
        """Genetic algorithm for finding small covering codes."""
        rng = np.random.RandomState(seed)
        
        # Initialize population
        population = []
        for i in range(pop_size):
            sel = greedy_init(seed=seed * 1000 + i)
            population.append(sel)
        
        best_sel = min(population, key=len)
        best_size = len(best_sel)
        
        for gen in range(generations):
            if time_left() < 5.0:
                break
            
            # Evaluate fitness (size)
            population.sort(key=len)
            
            # Elitism: keep best 20%
            elite_count = max(1, pop_size // 5)
            new_population = list(population[:elite_count])
            
            # Crossover and mutation to fill rest
            while len(new_population) < pop_size:
                if time_left() < 2.0:
                    break
                
                # Select parents (tournament selection)
                p1_idx = int(rng.choice(pop_size, size=3, replace=False)[np.argmin([len(population[i]) for i in rng.choice(pop_size, size=3, replace=False)])])
                p2_idx = int(rng.choice(pop_size, size=3, replace=False)[np.argmin([len(population[i]) for i in rng.choice(pop_size, size=3, replace=False)])])
                
                p1 = list(population[p1_idx])
                p2 = list(population[p2_idx])
                
                # Crossover: random subset of p1 + fill from p2
                if len(p1) > 0 and len(p2) > 0:
                    cross_point = int(rng.randint(0, min(len(p1), len(p2))))
                    child = list(p1[:cross_point])
                    # Fill remaining from p2, avoiding duplicates
                    for c in p2:
                        if c not in child:
                            child.append(c)
                        if len(child) >= len(p1):
                            break
                else:
                    child = list(p1) if p1 else list(p2)
                
                # Mutation: randomly remove/add codewords
                if rng.random() < 0.3 and len(child) > 5:
                    # Remove a random codeword
                    rem_idx = int(rng.randint(0, len(child) - 1))
                    child.pop(rem_idx)
                
                if rng.random() < 0.3:
                    # Add a random codeword
                    add_idx = int(rng.randint(0, total - 1))
                    if add_idx not in child:
                        child.append(add_idx)
                
                new_population.append(child)
            
            population = new_population[:pop_size]
            
            # Local search on best few
            for i in range(min(5, len(population))):
                if time_left() < 3.0:
                    break
                improved = local_search(population[i], max_iters=50)
                if len(improved) < len(population[i]):
                    population[i] = improved
                    if len(improved) < best_size:
                        best_size = len(improved)
                        best_sel = list(improved)
        
        return best_sel
    
    # Main algorithm
    # Start with greedy initialization
    best_overall = greedy_init(seed=42)
    best_size = len(best_overall)
    
    # Local search on the greedy result
    improved = local_search(best_overall, max_iters=300)
    if len(improved) < best_size:
        best_size = len(improved)
        best_overall = improved
    
    # Try multiple random restarts with greedy
    for restart in range(10):
        if time_left() < 10.0:
            break
        sel = greedy_init(seed=100 + restart * 7)
        sel = local_search(sel, max_iters=100)
        if len(sel) < best_size:
            best_size = len(sel)
            best_overall = sel
    
    # Try genetic algorithm if time permits
    if time_left() > 30.0:
        ga_result = genetic_algorithm(pop_size=30, generations=20, seed=123)
        if len(ga_result) < best_size:
            best_size = len(ga_result)
            best_overall = ga_result
    
    # Final validity check and completion
    if best_overall is None or len(best_overall) == 0:
        best_overall = greedy_init(seed=999)
    
    cov = build_cov(best_overall)
    if np.min(cov) < 1:
        # Greedy completion
        uncovered = np.where(cov < 1)[0]
        while len(uncovered) > 0:
            if time_left() < 1.0:
                break
            counts = mem[:, uncovered].sum(axis=1)
            best_idx = int(np.argmax(counts))
            if counts[best_idx] == 0:
                break
            best_overall.append(best_idx)
            cov[mem[best_idx, :]] += 1
            uncovered = np.where(cov < 1)[0]
    
    # Deduplicate and convert
    seen = set()
    final = []
    for c in best_overall:
        if c in seen:
            continue
        seen.add(c)
        final.append(digits_of(c))
    
    if not final:
        return [list(x) for x in itertools.product(range(q), repeat=n)]
    
    return final
