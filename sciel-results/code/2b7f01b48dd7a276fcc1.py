"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

# Precompute prime factors for all numbers 1..511
def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

_PRIME_FACTORS = {}
for _i in range(1, 512):
    _PRIME_FACTORS[_i] = _factorize(_i)

# Precompute score contribution for each a
_SCORES = [0.0] + [1.0 / (512 - a) for a in range(1, 512)]

def _is_coprime_to_all_fast(a, chosen_set):
    """Check if a is coprime to all elements in chosen_set using precomputed factors."""
    fa = _PRIME_FACTORS[a]
    for b in chosen_set:
        if fa & _PRIME_FACTORS[b]:
            return False
    return True

def _score(chosen):
    return sum(_SCORES[a] for a in chosen)

def _greedy_from_high():
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if _is_coprime_to_all_fast(a, chosen_set):
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _greedy_with_lookahead():
    """Greedy that looks ahead by depth to avoid blocking high-value future picks."""
    chosen = []
    chosen_set = set()
    candidates = list(range(511, 0, -1))
    
    while candidates:
        best_candidate = None
        best_score = -1
        for i, a in enumerate(candidates):
            if not _is_coprime_to_all_fast(a, chosen_set):
                continue
            # Try adding a and see what we can still get
            new_set = chosen_set | {a}
            # Greedily fill the rest
            temp_chosen = list(chosen) + [a]
            temp_set = set(new_set)
            for b in candidates[i+1:]:
                if _is_coprime_to_all_fast(b, temp_set):
                    temp_chosen.append(b)
                    temp_set.add(b)
            sc = _score(temp_chosen)
            if sc > best_score:
                best_score = sc
                best_candidate = a
        
        if best_candidate is None:
            break
        
        chosen.append(best_candidate)
        chosen_set.add(best_candidate)
        candidates = [a for a in candidates if a != best_candidate]
    
    return chosen

def _local_search_2swap(chosen, deadline):
    """Local search with 1-remove-re-greedy and 2-swap moves."""
    best = list(chosen)
    best_set = set(best)
    best_score = _score(best)
    n = len(best)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Try removing each element and re-greeding
        for i in range(n):
            if time.time() > deadline:
                break
            remaining = [x for j, x in enumerate(best) if j != i]
            remaining_set = set(remaining)
            new_list = list(remaining)
            for a in range(511, 0, -1):
                if a in remaining_set:
                    continue
                if _is_coprime_to_all_fast(a, remaining_set):
                    new_list.append(a)
                    remaining_set.add(a)
            new_score = _score(new_list)
            if new_score > best_score + 1e-12:
                best = new_list
                best_set = set(best)
                best_score = new_score
                n = len(best)
                improved = True
                break
        
        # Try 2-swap: remove two elements, add two different ones, then re-greedy
        if not improved and time.time() < deadline and n >= 2:
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    removed1, removed2 = best[i], best[j]
                    remaining = [x for k, x in enumerate(best) if k != i and k != j]
                    remaining_set = set(remaining)
                    # Try adding pairs of candidates
                    best_swap = None
                    best_swap_score = best_score
                    # Only try a limited set of high-value candidates for speed
                    candidates = [a for a in range(511, max(0, 400), -1) if a not in remaining_set and a != removed1 and a != removed2]
                    for a in candidates:
                        if not _is_coprime_to_all_fast(a, remaining_set):
                            continue
                        set1 = remaining_set | {a}
                        for b in candidates:
                            if b == a or b in set1:
                                continue
                            if not _is_coprime_to_all_fast(b, set1):
                                continue
                            new_set = set1 | {b}
                            new_list = remaining + [a, b]
                            # Re-greedy fill
                            for c in range(511, 0, -1):
                                if c in new_set:
                                    continue
                                if _is_coprime_to_all_fast(c, new_set):
                                    new_list.append(c)
                                    new_set.add(c)
                            sc = _score(new_list)
                            if sc > best_swap_score + 1e-12:
                                best_swap_score = sc
                                best_swap = new_list
                    if best_swap is not None:
                        best = best_swap
                        best_set = set(best)
                        best_score = best_swap_score
                        n = len(best)
                        improved = True
                        break
                if improved:
                    break
    return best

def _simulated_annealing(chosen, deadline):
    """Simulated annealing to escape local optima."""
    current = list(chosen)
    current_set = set(current)
    current_score = _score(current)
    best = list(current)
    best_set = set(best)
    best_score = current_score
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor: remove one random element, add one random compatible element
        if len(current) < 2:
            break
        
        # Remove a random element
        idx = random.randint(0, len(current) - 1)
        removed = current[idx]
        new_set = set(current)
        new_set.remove(removed)
        
        # Find all compatible candidates
        candidates = [a for a in range(1, 512) if a not in new_set and _is_coprime_to_all_fast(a, new_set)]
        if not candidates:
            T *= cooling
            continue
        
        # Pick a random candidate, biased toward high values
        # Use exponential weighting
        weights = [1.0 / (512 - a) for a in candidates]
        total_w = sum(weights)
        r = random.random() * total_w
        cum = 0
        chosen_a = candidates[0]
        for a, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                chosen_a = a
                break
        
        new_list = [x for x in current if x != removed] + [chosen_a]
        new_score = _score(new_list)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current = new_list
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best = list(current)
                best_set = set(best)
                best_score = current_score
        
        T *= cooling
    
    return best

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s kill
    
    # Generate multiple initial solutions
    solutions = []
    
    # Standard greedy
    sol1 = _greedy_from_high()
    solutions.append(sol1)
    
    # Greedy with lookahead
    sol2 = _greedy_with_lookahead()
    solutions.append(sol2)
    
    # Multi-start with random perturbations
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        order = list(range(511, 0, -1))
        # Perturb: swap random pairs
        for _ in range(10):
            i, j = rng.randint(0, 510), rng.randint(0, 510)
            order[i], order[j] = order[j], order[i]
        chosen = []
        chosen_set = set()
        for a in order:
            if _is_coprime_to_all_fast(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        solutions.append(chosen)
    
    # Local search on each solution
    best_overall = None
    best_overall_score = -1
    
    for sol in solutions:
        if time.time() > deadline:
            break
        improved = _local_search_2swap(sol, deadline)
        sc = _score(improved)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = improved
    
    # Simulated annealing on the best solution
    if time.time() < deadline and best_overall is not None:
        sa_result = _simulated_annealing(best_overall, deadline)
        sa_score = _score(sa_result)
        if sa_score > best_overall_score:
            best_overall = sa_result
            best_overall_score = sa_score
    
    return best_overall
# EVOLVE-BLOCK-END
