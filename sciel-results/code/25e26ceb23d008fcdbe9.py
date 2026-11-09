"""
Combined approach: CSP-style search with forward checking + beam search refinement +
conflict-aware local search. The core idea is to use the CSP solver for global exploration
and beam search for maintaining diverse partial sequences, then refine with targeted
local search that considers future constraint propagation.
"""
import time
import random
import sys
from collections import defaultdict
from functools import lru_cache

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _fitness(seq):
    """
    Fitness for genetic algorithm: primarily length, with tie-breaking by
    how much "room" remains (smaller last value = more room for extension).
    Returns a tuple (length, -last_value) for comparison.
    """
    if not seq:
        return (0, 0)
    return (len(seq), -seq[-1])

def _genetic_crossover(parent1, parent2, deadline):
    """
    Crossover: take prefix from one parent, try to extend with elements
    from the other parent that are valid and greater than the prefix's last element.
    """
    if len(parent1) < len(parent2):
        p1, p2 = parent1, parent2
    else:
        p1, p2 = parent2, parent1
    
    # Try different crossover points
    best_child = []
    for cp in range(1, min(len(p1), 8) + 1):
        if time.time() > deadline:
            break
        prefix = list(p1[:cp])
        child = list(prefix)
        used_sums = _get_all_sums(prefix)
        last = prefix[-1]
        
        # Try to extend with elements from p2[cp:] that fit
        for v in p2[cp:]:
            if v <= last:
                continue
            tails = _compute_new_sums(prefix, v)
            if not used_sums.intersection(tails):
                child.append(v)
                used_sums.update(tails)
                last = v
                prefix = list(child)
        
        if len(child) > len(best_child):
            best_child = child
    
    # Also try greedy extension from the best crossover prefix
    if best_child:
        extended = list(best_child)
        used_sums = _get_all_sums(extended)
        while time.time() < deadline:
            found = False
            for v in range(extended[-1] + 1, 257):
                tails = _compute_new_sums(extended, v)
                if not used_sums.intersection(tails):
                    extended.append(v)
                    used_sums.update(tails)
                    found = True
                    break
            if not found:
                break
        if len(extended) > len(best_child):
            best_child = extended
    
    return best_child

def _genetic_mutate(seq, deadline, rng):
    """
    Mutation: randomly replace one or two elements near the end with different values,
    then greedily extend.
    """
    if not seq or len(seq) < 2:
        return seq
    
    best_mutant = list(seq)
    n = len(seq)
    
    # Try replacing 1-2 elements near the end
    for num_repl in [1, 2]:
        for _ in range(5):
            if time.time() > deadline:
                break
            positions = rng.sample(range(max(0, n - 4), n), min(num_repl, n))
            mutant = list(seq)
            
            valid = True
            for pos in sorted(positions, reverse=True):
                prev_val = mutant[pos - 1] if pos > 0 else 0
                next_val = mutant[pos + 1] if pos + 1 < n else 257
                lo = prev_val + 1
                hi = next_val - 1
                if lo >= hi:
                    valid = False
                    break
                new_val = rng.randint(lo, hi)
                mutant[pos] = new_val
            
            if not valid or not _check_valid(mutant):
                continue
            
            # Greedily extend
            extended = list(mutant)
            used_sums = _get_all_sums(extended)
            while time.time() < deadline:
                found = False
                for v in range(extended[-1] + 1, 257):
                    tails = _compute_new_sums(extended, v)
                    if not used_sums.intersection(tails):
                        extended.append(v)
                        used_sums.update(tails)
                        found = True
                        break
                if not found:
                    break
            
            if len(extended) > len(best_mutant):
                best_mutant = extended
    
    return best_mutant

def _genetic_algorithm(deadline, pop_size=20, generations=None):
    """
    Genetic algorithm to evolve candidate sequences.
    Fitness = sequence length (primary), -last_value (secondary, more room).
    """
    if generations is None:
        generations = 50
    
    rng = random.Random(42)
    
    # Initialize population with diverse seeds
    population = []
    
    # Seed 1: greedy smallest
    pop_seed = _construct_greedy_smallest()
    population.append(list(pop_seed))
    
    # Seed 2: greedy largest
    pop_seed2 = _construct_greedy_largest()
    population.append(list(pop_seed2))
    
    # Seed 3: powers construction
    pop_seed3 = _construct_powers()
    population.append(list(pop_seed3))
    
    # Add randomized initial sequences
    for _ in range(pop_size - 3):
        if time.time() > deadline:
            break
        # Random valid sequence
        seq = []
        last = 0
        for _ in range(20):
            if time.time() > deadline:
                break
            candidates = []
            for v in range(last + 1, 257):
                if len(seq) == 0:
                    candidates.append(v)
                else:
                    tails = _compute_new_sums(seq, v)
                    all_sums = _get_all_sums(seq)
                    if not all_sums.intersection(tails):
                        candidates.append(v)
            if not candidates:
                break
            # Pick a random candidate, biased toward smaller values
            idx = min(len(candidates) - 1, int(rng.random() ** 2 * len(candidates)))
            v = candidates[idx]
            seq.append(v)
            last = v
        if len(seq) > 0:
            population.append(seq)
    
    # Remove duplicates
    unique_pop = []
    seen = set()
    for s in population:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_pop.append(s)
    population = unique_pop
    
    best_seq = max(population, key=_fitness) if population else []
    best_fit = _fitness(best_seq)
    
    gen = 0
    while gen < generations and time.time() < deadline - 2:
        gen += 1
        new_population = []
        
        # Elitism: keep top 3
        population.sort(key=_fitness, reverse=True)
        for s in population[:3]:
            new_population.append(list(s))
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline - 2:
            # Selection: tournament of 3
            p1, p2 = rng.sample(population, min(2, len(population)))
            
            # Crossover
            child = _genetic_crossover(p1, p2, deadline - 2)
            if not child:
                child = list(p1) if _fitness(p1) >= _fitness(p2) else list(p2)
            
            # Mutation with probability 0.4
            if rng.random() < 0.4:
                mutated = _genetic_mutate(child, deadline - 2, rng)
                if _fitness(mutated) >= _fitness(child):
                    child = mutated
            
            new_population.append(child)
        
        # Update best
        for s in new_population:
            fit = _fitness(s)
            if fit > best_fit:
                best_fit = fit
                best_seq = list(s)
        
        population = new_population
    
    return best_seq

def _compute_new_sums(seq, v):
    """Compute new block sums when appending v to seq."""
    new_sums = [v]
    total = v
    for i in range(len(seq) - 1, -1, -1):
        total += seq[i]
        new_sums.append(total)
    return new_sums

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _construct_greedy_largest():
    """Greedy largest - different bias, may find different optima."""
    values, sums = [], set()
    for x in range(256, 0, -1):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    values.sort()
    return values

def _get_all_sums(seq):
    """Compute all consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _solve_csp(deadline, seed=None):
    """
    CSP-style solver with forward checking and randomized value ordering.
    """
    if seed is not None:
        random.seed(seed)
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, sums, last_val):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        # Branch-and-bound: max possible additional elements
        max_additional = 256 - last_val
        if n + max_additional <= best_len:
            return
        
        # Generate valid candidates with forward checking
        candidates = []
        for v in range(last_val + 1, 257):
            new_sums_list = _compute_new_sums(seq, v)
            valid = True
            for s in new_sums_list:
                if s in sums:
                    valid = False
                    break
            if valid:
                candidates.append((v, new_sums_list))
        
        if not candidates:
            return
        
        # Value ordering: mix of smallest (most room) and random for diversity
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 8:
            ordered = candidates[:4] + random.sample(candidates[4:], min(4, len(candidates) - 4))
        else:
            ordered = candidates
        
        for v, new_sums_list in ordered:
            if time.time() > deadline:
                return
            new_sums = sums.union(new_sums_list)
            dfs(seq + [v], new_sums, v)
    
    dfs([], set(), 0)
    return best_seq

def _beam_search(deadline, beam_width=12):
    """
    Beam search maintaining top-K partial sequences.
    Each beam entry prioritizes length first, then smaller last value for more room.
    """
    best_result = []
    
    # Initial beam with empty sequence
    beam = [(0, 0, [], set())]  # (neg_len, last_val, values, sums)
    
    while beam and time.time() < deadline - 2:
        new_beam = []
        for neg_len, last_val, values, sums in beam:
            # Try extending this sequence
            for x in range(last_val + 1, 257):
                if time.time() > deadline - 2:
                    break
                tails = _compute_new_sums(values, x)
                if not sums.intersection(tails):
                    new_values = values + [x]
                    new_sums = sums | set(tails)
                    new_neg_len = -(len(new_values))
                    new_beam.append((new_neg_len, x, new_values, new_sums))
            
            # Keep current state as candidate
            new_beam.append((neg_len, last_val, values, sums))
        
        if not new_beam:
            break
        
        # Sort and keep top beam_width
        new_beam.sort(key=lambda e: (e[0], e[1]))
        beam = new_beam[:beam_width]
        
        # Track best
        for entry in beam:
            if len(entry[2]) > len(best_result):
                best_result = entry[2]
    
    return best_result

def _solve_csp_multi_restart(deadline, n_restarts=6):
    """
    Multi-restart CSP solver with different random seeds.
    """
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    remaining_time = deadline - time.time()
    if remaining_time <= 1:
        return best_seq
    
    per_restart = remaining_time / max(n_restarts, 1)
    
    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break
        restart_deadline = min(deadline - 2, time.time() + per_restart)
        try:
            result = _solve_csp(restart_deadline, seed=r * 12345 + 678)
            if len(result) > best_len:
                best_seq = result
                best_len = len(result)
        except Exception:
            pass
    
    return best_seq

def _local_search_refine(seq, deadline):
    """
    Local search: try to extend the sequence by finding valid next elements.
    Also try alternative endings to unlock further extensions.
    """
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    # Try to extend greedily
    while time.time() < deadline - 1:
        extended = False
        for v in range(best[-1] + 1, 257):
            candidate = best + [v]
            if _check_valid(candidate):
                best = candidate
                best_len = len(best)
                extended = True
                break
        if not extended:
            break
    
    # Try replacing the last element with a different one to unlock extensions
    for pos in range(max(0, len(best) - 3), len(best)):
        if time.time() > deadline - 1:
            break
        original = best[pos:]
        for v in range((best[pos-1] + 1) if pos > 0 else 1, 257):
            candidate = best[:pos] + [v]
            if not _check_valid(candidate):
                continue
            # Try to extend this candidate
            extended_candidate = list(candidate)
            while time.time() < deadline - 1:
                ext_found = False
                for w in range(extended_candidate[-1] + 1, 257):
                    test = extended_candidate + [w]
                    if _check_valid(test):
                        extended_candidate = test
                        ext_found = True
                        break
                if not ext_found:
                    break
            if len(extended_candidate) > best_len:
                best = extended_candidate
                best_len = len(extended_candidate)
    
    return best

def _conflict_aware_extension(seq, deadline):
    """
    Conflict-aware extension: for each position near the end, try alternative values
    that minimize future conflicts (fewer blocked sums).
    """
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    # For positions near the end, try alternatives and see which allows more extension
    for pos in range(max(0, len(best) - 4), len(best)):
        if time.time() > deadline - 2:
            break
        
        original_val = best[pos]
        prev_val = best[pos-1] if pos > 0 else 0
        
        # Try different values at this position
        for v in range(prev_val + 1, 257):
            if v == original_val:
                continue
            if time.time() > deadline - 2:
                break
            
            candidate = best[:pos] + [v]
            if not _check_valid(candidate):
                continue
            
            # Try to extend from here
            extended = list(candidate)
            while time.time() < deadline - 2:
                ext_found = False
                for w in range(extended[-1] + 1, 257):
                    test = extended + [w]
                    if _check_valid(test):
                        extended = test
                        ext_found = True
                        break
                if not ext_found:
                    break
            
            if len(extended) > best_len:
                best = extended
                best_len = len(extended)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Multiple baselines
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    try:
        greedy_large = _construct_greedy_largest()
        if len(greedy_large) > best_len:
            best = greedy_large
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Multi-restart CSP solver
    try:
        csp_result = _solve_csp_multi_restart(deadline - 15, n_restarts=6)
        if len(csp_result) > best_len:
            best = csp_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Beam search for diverse partial sequences
    try:
        beam_result = _beam_search(deadline - 10, beam_width=12)
        if len(beam_result) > best_len:
            best = beam_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Genetic algorithm to explore non-local regions
    try:
        ga_result = _genetic_algorithm(deadline - 5, pop_size=20, generations=40)
        if len(ga_result) > best_len:
            best = ga_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Conflict-aware extension
    try:
        conflict_result = _conflict_aware_extension(best, deadline - 2)
        if len(conflict_result) > best_len:
            best = conflict_result
            best_len = len(best)
    except Exception:
        pass
    
    # Local search refinement on the best sequence found
    try:
        refined = _local_search_refine(best, deadline)
        if len(refined) > best_len:
            best = refined
            best_len = len(best)
    except Exception:
        pass
    
    # Final verification
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
