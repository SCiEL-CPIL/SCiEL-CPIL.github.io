import time
import random
import math
import numpy as np
from copy import deepcopy

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    max_u = max(U_set)
    # Compute sumset and diffset efficiently
    sums = set()
    diffs = set()
    diffs.add(0)
    U_list = sorted(U_set)
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n):
            d = U_list[j] - ui
            diffs.add(d)
            diffs.add(-d)
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max U + 1)
    if len(sums) == 0:
        return 0.0
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    if 2 * max_u + 1 <= 1:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _compute_sums_diffs(U_list):
    """Compute sumset and diffset for a sorted list U containing 0."""
    n = len(U_list)
    sums = set()
    diffs = set()
    diffs.add(0)
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n):
            d = U_list[j] - ui
            diffs.add(d)
            diffs.add(-d)
    return sums, diffs

def _marginal_ratio_gain(U_list, candidate):
    """
    Compute the marginal gain in ratio when adding `candidate` to U_list.
    Returns (new_ratio - old_ratio, new_sums, new_diffs).
    U_list must be sorted and contain 0.
    """
    if candidate in U_list:
        return 0.0, None, None
    
    n = len(U_list)
    # Current sums and diffs
    old_sums, old_diffs = _compute_sums_diffs(U_list)
    
    # New sums: old_sums + {candidate + u for u in U_list}
    new_sums = set(old_sums)
    for u in U_list:
        new_sums.add(candidate + u)
    
    # New diffs: old_diffs + {candidate - u, u - candidate for u in U_list}
    new_diffs = set(old_diffs)
    for u in U_list:
        new_diffs.add(candidate - u)
        new_diffs.add(u - candidate)
    
    old_ratio = len(old_diffs) / len(old_sums) if old_sums else 0.0
    new_ratio = len(new_diffs) / len(new_sums) if new_sums else 0.0
    
    return new_ratio - old_ratio, new_sums, new_diffs

def _greedy_construct(max_elements, max_val, deadline):
    """
    Greedy construction: start with {0}, then repeatedly add the element
    that maximizes the marginal gain in |U-U|/|U+U| ratio.
    Uses binary-structure-aware candidate generation to avoid carries.
    """
    U = [0]
    U_set = {0}
    
    # Generate initial candidates: powers of 2, and their combinations
    candidates = set()
    # Powers of 2 up to max_val
    for k in range(1, 20):
        p = 1 << k
        if p <= max_val:
            candidates.add(p)
        else:
            break
    
    # Also add some structured candidates
    for k in range(1, 20):
        p = 1 << k
        if p <= max_val:
            # p/2, 3p/2, 2p, 5p/2 etc. (binary-friendly)
            for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                v = int(p * mult)
                if 0 < v <= max_val and v not in U_set:
                    candidates.add(v)
    
    # Add some small integers
    for v in range(1, min(50, max_val + 1)):
        candidates.add(v)
    
    current_sums, current_diffs = _compute_sums_diffs(U)
    current_ratio = len(current_diffs) / len(current_sums) if current_sums else 0.0
    
    while len(U) < max_elements and time.time() < deadline - 5:
        # Generate new candidates based on current structure
        new_candidates = set()
        U_sorted = sorted(U)
        
        # From existing elements, generate sum/diff candidates
        for a in U_sorted:
            for b in U_sorted:
                s = a + b
                d = abs(b - a)
                if 0 < s <= max_val and s not in U_set:
                    new_candidates.add(s)
                if 0 < d <= max_val and d not in U_set:
                    new_candidates.add(d)
        
        # Binary-structure candidates: next power of 2 above max
        if U_sorted:
            max_u = U_sorted[-1]
            # Next power of 2
            if max_u > 0:
                next_pow2 = 1 << (max_u.bit_length())
                if next_pow2 <= max_val:
                    new_candidates.add(next_pow2)
                # Also 2*max_u, 3*max_u (careful about carries)
                for mult in [2, 3]:
                    v = max_u * mult
                    if 0 < v <= max_val and v not in U_set:
                        new_candidates.add(v)
        
        # Add to candidate pool
        candidates = candidates | new_candidates
        candidates -= U_set
        
        if not candidates:
            break
        
        # Evaluate top candidates (limit to avoid O(n^3) blowup)
        candidate_list = sorted(candidates)[:100]
        random.shuffle(candidate_list)
        candidate_list = candidate_list[:50]
        
        best_candidate = None
        best_gain = -float('inf')
        best_new_sums = None
        best_new_diffs = None
        
        for c in candidate_list:
            gain, new_sums, new_diffs = _marginal_ratio_gain(U, c)
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                best_new_sums = new_sums
                best_new_diffs = new_diffs
        
        if best_candidate is None or best_gain <= 0:
            # No positive gain; try random exploration
            candidate_list = sorted(candidates)[:20]
            random.shuffle(candidate_list)
            best_candidate = None
            best_gain = -float('inf')
            for c in candidate_list:
                gain, new_sums, new_diffs = _marginal_ratio_gain(U, c)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                    best_new_sums = new_sums
                    best_new_diffs = new_diffs
            
            if best_candidate is None or best_gain < -0.01:
                break
        
        # Add the best candidate
        U.append(best_candidate)
        U.sort()
        U_set.add(best_candidate)
        current_sums = best_new_sums
        current_diffs = best_new_diffs
        current_ratio = len(current_diffs) / len(current_sums)
        
        # Remove used candidate
        candidates.discard(best_candidate)
    
    return U

def _local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = _score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = _generate_candidates(U, new_limit)
        
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
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
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
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def _generate_candidates(U, limit_val, random_small_count=15):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    Also adds some random small integers for diversity.
    """
    U_set = set(U)
    candidates = set()
    U_list = sorted(U_set)
    n = len(U_list)
    for i in range(n):
        a = U_list[i]
        for j in range(i, n):
            b = U_list[j]
            s = a + b
            d = b - a
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    # Add random small integers for diversity
    for _ in range(random_small_count):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
    return [c for c in candidates if 0 < c <= limit_val]

def _score_geometric(b, m):
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
    return _score_set(U)

def _continuous_geometric_search(deadline):
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
            s = _score_geometric(b, m)
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
        s1 = _score_geometric(m1, m0)
        s2 = _score_geometric(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    s_opt = _score_geometric(b_opt, m0)
    
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
            s1 = _score_geometric(m1, m_adj)
            s2 = _score_geometric(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = _score_geometric(b_adj, m_adj)
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

def _cyclic_difference_set_seed(q, deadline):
    """
    Generate a seed from a cyclic difference set of order q (prime power).
    A (v, k, lambda) difference set has v = q^2 + q + 1, k = q + 1, lambda = 1.
    For prime power q, Singer difference sets exist.
    """
    if q < 2 or q > 20:  # Limit to reasonable sizes
        return [0, 1, 2]
    
    # For small q, use known constructions
    # q=2: (7,3,1) - Fano plane
    if q == 2:
        return [0, 1, 3]
    # q=3: (13,4,1)
    if q == 3:
        return [0, 1, 3, 9]
    # q=4: (21,5,1)
    if q == 4:
        return [0, 1, 4, 10, 16]
    # q=5: (31,6,1)
    if q == 5:
        return [0, 1, 5, 11, 20, 26]
    # q=7: (57,8,1)
    if q == 7:
        return [0, 1, 7, 18, 28, 40, 50, 55]
    
    # For larger q, use a greedy approach inspired by difference sets
    # Try to construct a set where all pairwise differences are distinct
    U = [0]
    U_set = {0}
    diffs_seen = set()
    
    for candidate in range(1, q*q + q + 2):
        if time.time() > deadline - 10:
            break
        # Check if adding candidate maintains distinct differences
        valid = True
        for u in U:
            d = abs(candidate - u)
            if d in diffs_seen:
                valid = False
                break
        if valid and len(U) <= q + 1:
            U.append(candidate)
            U_set.add(candidate)
            for u in U[:-1]:
                diffs_seen.add(abs(candidate - u))
            if len(U) == q + 1:
                break
    
    U.sort()
    return U

def _genetic_algorithm(U_seed, deadline, pop_size=30, generations=50):
    """
    Genetic algorithm treating the set's membership as a genome.
    Genome: binary vector of length max_val+1 indicating membership.
    """
    U_seed = sorted(set(U_seed))
    if not U_seed or 0 not in U_seed:
        U_seed = [0, 1]
    
    max_val = max(U_seed) * 2 + 100
    max_val = min(max_val, 500)  # Cap for performance
    
    def genome_to_set(genome):
        """Convert binary genome to set, ensuring 0 is included."""
        U = [0]
        for i, bit in enumerate(genome):
            if i > 0 and bit:
                U.append(i)
        return sorted(U)
    
    def set_to_genome(U_set):
        """Convert set to binary genome."""
        genome = [0] * (max_val + 1)
        genome[0] = 1
        for u in U_set:
            if u <= max_val:
                genome[u] = 1
        return genome
    
    def fitness(U_list):
        """Fitness = score, with penalty for too large or too small sets."""
        if not U_list or 0 not in U_list or len(U_list) < 2:
            return 0.0
        n = len(U_list)
        if n > 30:
            return _score_set(U_list) * 0.8  # Penalty for large sets
        if n < 3:
            return _score_set(U_list) * 0.9
        return _score_set(U_list)
    
    # Initialize population
    population = []
    best_seed_genome = set_to_genome(U_seed)
    population.append(best_seed_genome)
    
    for _ in range(pop_size - 1):
        # Mutate the seed
        genome = list(best_seed_genome)
        for _ in range(random.randint(1, 5)):
            pos = random.randint(1, max_val)
            genome[pos] = 1 - genome[pos]
        population.append(genome)
    
    best_fitness = -1.0
    best_genome = best_seed_genome
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
        
        # Evaluate fitness
        scored_pop = []
        for genome in population:
            U_list = genome_to_set(genome)
            f = fitness(U_list)
            scored_pop.append((f, genome))
            if f > best_fitness:
                best_fitness = f
                best_genome = genome
        
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # Selection: keep top half
        elite_count = max(5, pop_size // 3)
        next_pop = [g for _, g in scored_pop[:elite_count]]
        
        # Fill rest with offspring
        while len(next_pop) < pop_size:
            if time.time() > deadline - 5:
                break
            # Crossover
            p1 = scored_pop[random.randint(0, elite_count - 1)][1]
            p2 = scored_pop[random.randint(0, elite_count - 1)][1]
            
            # Single-point crossover
            cross_point = random.randint(1, max_val)
            child = p1[:cross_point] + p2[cross_point:]
            
            # Mutation
            for _ in range(random.randint(1, 3)):
                pos = random.randint(1, max_val)
                child[pos] = 1 - child[pos]
            
            next_pop.append(child)
        
        population = next_pop
    
    # Return best set found
    best_U = genome_to_set(best_genome)
    return best_U, fitness(best_U)

def _smooth_surrogate_search(U_seed, deadline):
    """
    Use a smooth surrogate: instead of exact |U-U|/|U+U|, use a soft count
    based on the distribution of sums and differences.
    """
    U_seed = sorted(set(U_seed))
    if not U_seed or 0 not in U_seed:
        U_seed = [0, 1]
    
    best_U = list(U_seed)
    best_score = _score_set(best_U)
    
    # Try local perturbations with a smooth objective
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        U = list(best_U)
        improved = False
        
        # Try adding, removing, or replacing elements
        for move_type in ['add', 'remove', 'replace']:
            if time.time() > deadline - 3:
                break
            
            if move_type == 'add' and len(U) < 25:
                # Try adding elements near existing ones
                candidates = set()
                for u in U:
                    for delta in [-3, -2, -1, 1, 2, 3, 5, 7, 11]:
                        c = u + delta
                        if 0 < c <= 500 and c not in U:
                            candidates.add(c)
                
                random.shuffle(list(candidates))
                for c in list(candidates)[:15]:
                    new_U = sorted(U + [c])
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_U = new_U
                        best_score = s
                        improved = True
                        break
            elif move_type == 'remove' and len(U) > 3:
                removable = [x for x in U if x != 0]
                random.shuffle(removable)
                for c in removable[:10]:
                    new_U = [x for x in U if x != c]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_U = new_U
                        best_score = s
                        improved = True
                        break
            elif move_type == 'replace' and len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                candidates = set()
                for delta in range(-5, 6):
                    c = old_val + delta
                    if 0 < c <= 500 and c not in U and c != old_val:
                        candidates.add(c)
                
                random.shuffle(list(candidates))
                for c in list(candidates)[:10]:
                    new_U = U.copy()
                    new_U[idx] = c
                    new_U.sort()
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_U = new_U
                        best_score = s
                        improved = True
                        break
            
            if improved:
                break
    
    return best_U, best_score

def _final_polish(U, deadline):
    """
    High-precision final polish: exhaustive local search with tight tolerance.
    """
    U = sorted(set(U))
    current_score = _score_set(U)
    best_score = current_score
    best_U = list(U)
    
    max_val_limit = max(U) * 2 + 50
    max_val_limit = min(max_val_limit, 1000)
    
    no_improvement = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 1 and no_improvement < max_no_improvement:
        improved = False
        
        # Try all possible single-element changes
        # Add
        if len(U) < 30:
            candidates = set()
            for u in U:
                for delta in range(-10, 11):
                    c = u + delta
                    if 0 < c <= max_val_limit and c not in U:
                        candidates.add(c)
            # Also try some structured additions
            for u in U:
                for mult in [2, 3]:
                    c = u * mult
                    if 0 < c <= max_val_limit and c not in U:
                        candidates.add(c)
            
            for c in sorted(candidates):
                new_U = sorted(U + [c])
                s = _score_set(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(U)
                    break
        
        # Remove
        if len(U) > 3:
            removable = [x for x in U if x != 0]
            for c in removable:
                new_U = [x for x in U if x != c]
                s = _score_set(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(U)
                    break
        
        # Replace
        if len(U) > 2:
            for i in range(1, len(U)):
                old_val = U[i]
                candidates = set()
                for delta in range(-15, 16):
                    c = old_val + delta
                    if 0 < c <= max_val_limit and c not in U and c != old_val:
                        candidates.add(c)
                
                found = False
                for c in sorted(candidates):
                    new_U = U.copy()
                    new_U[i] = c
                    new_U.sort()
                    s = _score_set(new_U)
                    if s > current_score + 1e-12:
                        U = new_U
                        current_score = s
                        improved = True
                        found = True
                        if current_score > best_score:
                            best_score = current_score
                            best_U = list(U)
                        break
                if found:
                    break
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Initial seeds
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Structured seeds
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 6, 10, 15, 21, 28], # Triangular + 1?
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular
    ]
    
    for cand in structured_candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Cyclic difference set seeds
    for q in [2, 3, 4, 5, 7]:
        if time.time() > deadline - 30:
            break
        diff_set = _cyclic_difference_set_seed(q, deadline)
        s = _score_set(diff_set)
        if s > best_score:
            best_score = s
            best_set = list(diff_set)
    
    # Greedy construction with ratio-maximizing selection
    if time.time() < deadline - 40:
        greedy_set = _greedy_construct(max_elements=25, max_val=5000, deadline=deadline)
        greedy_score = _score_set(greedy_set)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = greedy_set
    
    # Geometric search (inspired by continuous optimization approach)
    if time.time() < deadline - 25:
        geo_set, geo_score = _continuous_geometric_search(deadline)
        if geo_score > best_score and geo_set:
            best_score = geo_score
            best_set = geo_set
    
    # Genetic algorithm refinement
    if time.time() < deadline - 20:
        ga_set, ga_score = _genetic_algorithm(best_set, deadline, pop_size=25, generations=40)
        if ga_score > best_score:
            best_score = ga_score
            best_set = ga_set
    
    # Smooth surrogate search
    if time.time() < deadline - 15:
        smooth_set, smooth_score = _smooth_surrogate_search(best_set, deadline)
        if smooth_score > best_score:
            best_score = smooth_score
            best_set = smooth_set
    
    # Dynamic Local Search with multiple restarts
    seeds = [best_set]
    # Add a few other good structured seeds for diversity
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds and random starts
        if restarts % 3 == 0:
            base = seeds[restarts % len(seeds)]
        else:
            # Random start
            n = random.randint(4, 15)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            base = [0] + elements
            base.sort()
        
        U = list(base)
        
        # Run local search
        local_score, local_set = _local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set
    
    # Final Polish: High-precision local search on the best set
    # Use a tighter limit and more iterations if time permits
    if time.time() < deadline - 2:
        final_score, final_set = _final_polish(best_set, deadline)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    return best_set
