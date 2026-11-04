"""
Combined approach: Structural insight (dynamic candidate generation from arithmetic neighborhoods)
+ Refinement machinery (efficient scoring, ternary/geometric search, local search, greedy, and GA).
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def _score_set_numpy(U):
    """Fast scoring using NumPy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    arr = np.array(U, dtype=np.int64)
    # Compute all sums (i<=j to avoid duplicates, but we need unique values)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    # Compute all differences (i<j)
    i_idx2, j_idx2 = np.tril_indices(n, k=-1)
    diffs_pos = arr[j_idx2] - arr[i_idx2]
    diffs = np.concatenate([diffs_pos, -diffs_pos, np.array([0])])
    
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    # Use fast numpy path for larger sets
    if n >= 15:
        return _score_set_numpy(U)
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def generate_parametric_candidates(deadline):
    candidates = []
    
    # 1. Geometric Progressions
    for B in range(2, 15):
        for K in range(3, 12):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            if time.time() > deadline - 10: return candidates
            candidates.append(U)
            
    # 2. "Powers of B" with offset or scaling
    for B in range(2, 10):
        for K in range(3, 10):
            for offset in range(0, 3):
                U = [offset]
                val = offset + 1
                for i in range(1, K):
                    U.append(val)
                    val = val * B + offset
                if 0 not in U:
                    U = sorted(set(U + [0]))
                candidates.append(U)
                
    # 3. Fibonacci-like
    for K in range(3, 10):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
        
    # 4. Randomized geometric
    random.seed(42)
    for _ in range(50):
        B = random.uniform(1.5, 5.0)
        K = random.randint(4, 12)
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        candidates.append(U)
        
    # 5. Perfect difference sets (cyclic)
    # Singer difference sets from projective planes
    for q in [2, 3, 4, 5, 7, 8, 9, 11, 13]:
        if time.time() > deadline - 10:
            break
        # Singer difference set: size k=q*q+q+1, v=q*q+q+1, lambda=q
        k = q * q + q + 1
        v = k
        # Use the standard construction
        U = [0]
        # For small q, use known constructions
        if q == 2:
            U = [0, 1, 3]
        elif q == 3:
            U = [0, 1, 3, 4, 9, 10, 12]
        elif q == 4:
            U = [0, 1, 4, 10, 12, 13, 15, 16, 19, 21, 25]
        elif q == 5:
            U = [0, 1, 4, 14, 16, 17, 22, 28, 29, 34, 35, 38, 39, 41, 44]
        elif q == 7:
            U = [0, 1, 3, 7, 12, 18, 21, 29, 32, 34, 37, 38, 46, 48, 49, 51, 53, 57, 58, 60, 63]
        elif q == 8:
            U = [0, 1, 7, 12, 18, 21, 29, 32, 34, 37, 38, 46, 48, 49, 51, 53, 57, 58, 60, 63, 64, 66, 70, 71, 73, 75, 79, 80, 82, 85, 86, 88, 92, 93, 95, 96, 98, 102, 103, 105, 108, 109, 111, 112, 114, 118]
        else:
            continue
        U = sorted(set(U))
        if 0 in U and len(U) >= 2:
            candidates.append(U)
        
    return candidates

def score_geometric(b, m):
    """Score a geometric progression U = {floor(b^k) for k=0..m} with 0."""
    if b <= 1.0 or m < 1:
        return -1.0
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
    return score_set(U)

def continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    # Phase 1: Coarse grid search to identify promising regions
    best_score = -1.0
    best_params = None
    
    # Coarse grid over b and m
    b_grid = np.linspace(1.5, 5.0, 20)
    m_grid = range(3, 15)
    
    for m in m_grid:
        if time.time() > deadline - 10:
            break
        for b in b_grid:
            s = score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return None, -1.0
    
    # Phase 2: Ternary search around the best coarse point
    b0, m0 = best_params
    b_lo = max(1.5, b0 - 0.5)
    b_hi = min(5.0, b0 + 0.5)
    
    lo = b_lo
    hi = b_hi
    for _ in range(40):  # 40 iterations for fine-grained search
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = score_geometric(m1, m0)
        s2 = score_geometric(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    s_opt = score_geometric(b_opt, m0)
    
    # Also try adjacent m values
    for m_adj in [m0 - 1, m0 + 1]:
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        # Quick ternary search for this m
        lo2 = max(1.5, b_opt - 0.3)
        hi2 = min(5.0, b_opt + 0.3)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            s1 = score_geometric(m1, m_adj)
            s2 = score_geometric(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = score_geometric(b_adj, m_adj)
        if s_adj > s_opt:
            s_opt = s_adj
            b_opt = b_adj
            m0 = m_adj
    
    # Construct the final set
    U = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    
    return U, s_opt

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This is the structural improvement: exploring the arithmetic neighborhood.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences of existing elements
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    
    # Add some random small integers to ensure diversity
    for _ in range(20):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def greedy_construct(deadline, max_n=25, max_val=200):
    """
    Greedy construction: add elements that maximize marginal |U-U|/|U+U| gain.
    """
    U = [0, 1]
    current_score = score_set(U)
    
    # Precompute current sums and diffs
    def _compute_sums_diffs(U):
        sums = set()
        diffs = set()
        for i in range(len(U)):
            for j in range(i, len(U)):
                sums.add(U[i] + U[j])
            for j in range(i + 1, len(U)):
                d = U[j] - U[i]
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        return sums, diffs
    
    current_sums, current_diffs = _compute_sums_diffs(U)
    
    for _ in range(max_n - 2):
        if time.time() > deadline - 10:
            break
            
        best_candidate = None
        best_gain = -1.0
        best_new_sums = None
        best_new_diffs = None
        
        # Generate candidates from arithmetic neighborhood
        candidates = set()
        for a in U:
            for b in U:
                s = a + b
                d = abs(a - b)
                if 0 < s <= max_val and s not in U:
                    candidates.add(s)
                if 0 < d <= max_val and d not in U:
                    candidates.add(d)
        
        # Add some random candidates
        for _ in range(30):
            r = random.randint(1, max_val)
            if r not in U:
                candidates.add(r)
        
        # Evaluate each candidate
        for c in candidates:
            new_U = U + [c]
            new_sums = set(current_sums)
            new_diffs = set(current_diffs)
            
            for x in U:
                new_sums.add(x + c)
                d = x - c
                new_diffs.add(d)
                new_diffs.add(-d)
            
            sum_count = len(new_sums)
            diff_count = len(new_diffs)
            
            if sum_count == 0:
                continue
                
            ratio = diff_count / sum_count
            if ratio <= 1:
                continue
                
            max_u = max(U + [c])
            D = 2 * max_u + 1
            if D <= 1:
                continue
                
            score = 1 + log(ratio) / log(D)
            
            if score > best_gain:
                best_gain = score
                best_candidate = c
                best_new_sums = new_sums
                best_new_diffs = new_diffs
        
        if best_candidate is not None and best_gain > current_score:
            U.append(best_candidate)
            U.sort()
            current_score = best_gain
            current_sums = best_new_sums
            current_diffs = best_new_diffs
        else:
            break
    
    return U, score_set(U)

def genetic_algorithm(population, deadline, max_val=500, max_n=30, pop_size=30, generations=50):
    """
    Genetic algorithm treating set membership as a genome.
    """
    random.seed(123)
    best_individual = None
    best_score = -1.0
    
    # Initialize population
    pop = []
    for ind in population:
        pop.append(list(ind))
    while len(pop) < pop_size:
        n = random.randint(4, max_n)
        elements = random.sample(range(1, max_val + 1), n - 1)
        U = [0] + elements
        U.sort()
        pop.append(U)
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
            
        # Evaluate population
        scored_pop = []
        for U in pop:
            s = score_set(U)
            scored_pop.append((s, U))
        
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_individual = list(scored_pop[0][1])
        
        # Elitism: keep top individuals
        elite_count = max(2, pop_size // 5)
        new_pop = [list(ind) for _, ind in scored_pop[:elite_count]]
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Select parents (tournament selection)
            def tournament_select(scored_pop, k=3):
                selected = random.sample(scored_pop, k)
                return max(selected, key=lambda x: x[0])[1]
            
            parent1 = tournament_select(scored_pop)
            parent2 = tournament_select(scored_pop)
            
            # Crossover: uniform crossover
            child = list(parent1)
            for i in range(len(parent2)):
                if random.random() < 0.5:
                    if i < len(child):
                        child[i] = parent2[i]
                    else:
                        child.append(parent2[i])
            
            # Mutation
            mutation_rate = 0.2
            if random.random() < mutation_rate:
                # Add an element
                if len(child) < max_n:
                    c = random.randint(1, max_val)
                    if c not in child:
                        child.append(c)
                        child.sort()
            if random.random() < mutation_rate:
                # Remove an element (not 0)
                if len(child) > 2:
                    idx = random.randint(1, len(child) - 1)
                    child.pop(idx)
            if random.random() < mutation_rate:
                # Replace an element
                if len(child) > 1:
                    idx = random.randint(1, len(child) - 1)
                    child[idx] = random.randint(1, max_val)
                    child = sorted(set(child))
            
            # Ensure 0 is in the set
            if 0 not in child:
                child.append(0)
                child.sort()
            
            new_pop.append(child)
        
        pop = new_pop
    
    return best_individual, best_score

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try some structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 2. Generate parametric candidates (including perfect difference sets)
    if time.time() < deadline - 30:
        param_candidates = generate_parametric_candidates(deadline)
        for cand in param_candidates:
            s = score_set(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
                structured_candidates.append(cand)
    
    # 3. Continuous geometric search
    if time.time() < deadline - 25:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            structured_candidates.append(geo_set)
    
    # 4. Greedy construction
    if time.time() < deadline - 20:
        greedy_set, greedy_score = greedy_construct(deadline, max_n=25, max_val=200)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = list(greedy_set)
            structured_candidates.append(greedy_set)
    
    # 5. Local Search with Dynamic Candidate Generation
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Choose a starting set
        if restarts % 3 == 0:
            # Start from a structured set
            base = random.choice(structured_candidates)
            U = list(base)
        else:
            # Start from random
            n = random.randint(4, 15)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # 6. Genetic Algorithm refinement
    if time.time() < deadline - 10:
        # Build initial population from best found and structured candidates
        ga_population = [list(best_set)]
        for cand in structured_candidates[:10]:
            ga_population.append(list(cand))
        
        ga_set, ga_score = genetic_algorithm(ga_population, deadline, max_val=500, max_n=30, pop_size=30, generations=40)
        if ga_set is not None and ga_score > best_score:
            best_score = ga_score
            best_set = list(ga_set)
    
    # Final polish: Try very small moves on the best set found
    if time.time() < deadline - 2:
        U = list(best_set)
        current_score = best_score
        U.sort()
        final_score, final_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set

# EVOLVE-BLOCK-END
