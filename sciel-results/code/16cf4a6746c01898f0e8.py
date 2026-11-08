import time
import random
import numpy as np
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Vectorized scoring for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def greedy_construct(deadline):
    """Greedy construction: build set element by element, picking best candidate each time."""
    U = [0]
    current_score = score_set_fast(U)
    best_score = current_score
    best_U = list(U)
    
    max_elements = 25
    max_val = 20000
    
    while time.time() < deadline - 30 and len(U) < max_elements:
        # Generate candidate elements to try
        candidates = set()
        M = max(U) if U else 1
        
        # Candidates: multiples of gaps, near existing, geometric extensions
        for a in U:
            for b in U:
                if a != b:
                    candidates.add(a + b)
                    candidates.add(abs(a - b))
                    if a > b and a - b > 0:
                        candidates.add(a + (a - b))
                        candidates.add(b + (a - b))
        
        # Add some random candidates
        for _ in range(50):
            candidates.add(random.randint(1, max_val))
        
        # Add geometric candidates
        if len(U) >= 2:
            ratio_est = U[-1] / U[-2] if U[-2] > 0 else 2
            for mult in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
                candidates.add(int(M * ratio_est * mult))
        
        # Filter valid candidates
        valid_candidates = [c for c in candidates if c > 0 and c not in U and c <= max_val]
        
        if not valid_candidates:
            break
        
        # Score each candidate and pick the best
        best_candidate = None
        best_candidate_score = current_score
        
        # Limit candidates to keep it fast
        random.seed(42)
        if len(valid_candidates) > 200:
            valid_candidates = random.sample(valid_candidates, 200)
        
        for cand in valid_candidates:
            new_U = sorted(U + [cand])
            s = score_set_fast(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = cand
        
        if best_candidate is None:
            # No improvement, try random additions
            improved = False
            for _ in range(20):
                cand = random.randint(1, max_val)
                if cand in U:
                    continue
                new_U = sorted(U + [cand])
                s = score_set_fast(new_U)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = cand
                    improved = True
            if not improved:
                break
        
        U = sorted(U + [best_candidate])
        current_score = best_candidate_score
        if current_score > best_score:
            best_score = current_score
            best_U = list(U)
    
    return best_score, best_U

def evolutionary_search(U, deadline, pop_size=20, generations=100):
    """Evolutionary algorithm with tournament selection and crossover."""
    U = sorted(set(U))
    current_score = score_set_fast(U)
    
    # Initialize population
    population = [list(U)]
    for _ in range(pop_size - 1):
        mutated = list(U)
        for _ in range(random.randint(1, 3)):
            move_type = random.random()
            if move_type < 0.4 and len(mutated) < 30:
                new_val = random.randint(1, max(mutated) + 100 if mutated else 100)
                if new_val not in mutated:
                    mutated.append(new_val)
            elif move_type < 0.6 and len(mutated) > 2:
                idx = random.randint(1, len(mutated) - 1)
                mutated.pop(idx)
            elif len(mutated) > 1:
                idx = random.randint(1, len(mutated) - 1)
                old_val = mutated[idx]
                delta = random.randint(-max(1, old_val // 5), max(1, old_val // 5))
                new_val = old_val + delta
                if new_val > 0 and new_val not in mutated:
                    mutated[idx] = new_val
                    mutated.sort()
        population.append(sorted(set(mutated)))
    
    # Score population
    scores = [score_set_fast(ind) for ind in population]
    best_idx = scores.index(max(scores))
    best_score = scores[best_idx]
    best_U = list(population[best_idx])
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
        
        # Tournament selection
        def select():
            a, b = random.sample(range(len(population)), 2)
            return population[a] if scores[a] >= scores[b] else population[b]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best
        new_population.append(list(population[best_idx]))
        
        for _ in range(pop_size - 1):
            parent1 = select()
            parent2 = select()
            
            # Crossover: take elements from both parents
            if random.random() < 0.5:
                # Uniform crossover
                child = []
                for a, b in zip(sorted(parent1), sorted(parent2)):
                    if random.random() < 0.5:
                        child.append(a)
                    else:
                        child.append(b)
                # Ensure 0 is present
                if 0 not in child:
                    child.append(0)
                child = sorted(set(child))
            else:
                # Take one parent and mutate
                child = list(parent1)
                for _ in range(random.randint(1, 3)):
                    move_type = random.random()
                    if move_type < 0.4 and len(child) < 30:
                        new_val = random.randint(1, max(child) + 50 if child else 50)
                        if new_val not in child:
                            child.append(new_val)
                    elif move_type < 0.6 and len(child) > 2:
                        idx = random.randint(1, len(child) - 1)
                        child.pop(idx)
                    elif len(child) > 1:
                        idx = random.randint(1, len(child) - 1)
                        old_val = child[idx]
                        delta = random.randint(-max(1, old_val // 4), max(1, old_val // 4))
                        new_val = old_val + delta
                        if new_val > 0 and new_val not in child:
                            child[idx] = new_val
                            child.sort()
                child = sorted(set(child))
            
            new_population.append(child)
        
        population = new_population
        scores = [score_set_fast(ind) for ind in population]
        
        # Update best
        gen_best_idx = scores.index(max(scores))
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_U = list(population[gen_best_idx])
            best_idx = gen_best_idx
    
    return best_score, best_U

def local_refine(U, deadline):
    """Local refinement with multiple perturbation scales."""
    U = sorted(set(U))
    current_score = score_set_fast(U)
    best_score = current_score
    best_U = list(U)
    
    scales = [1, 2, 5, 10, 20, 50, 100]
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        n = len(U)
        
        for scale in scales:
            if time.time() > deadline - 3:
                break
            for i in range(1, n):
                if time.time() > deadline - 3:
                    break
                old_val = U[i]
                for delta in range(-scale, scale + 1):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_U = [U[j] if j != i else new_val for j in range(n)]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        best_score = new_score
                        best_U = list(U)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try adding/removing elements
        if not improved and time.time() < deadline - 3:
            # Try adding
            if n < 28:
                M = max(U)
                for offset in range(-10, 11):
                    new_val = M + offset
                    if new_val <= 0 or new_val in U:
                        continue
                    new_U = sorted(U + [new_val])
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        best_score = new_score
                        best_U = list(U)
                        improved = True
                        break
            
            # Try removing
            if not improved and n > 3:
                for i in range(1, n):
                    new_U = [U[j] for j in range(n) if j != i]
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        best_score = new_score
                        best_U = list(U)
                        improved = True
                        break
    
    return best_score, best_U

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Phase 1: Greedy construction (~40s)
    greedy_deadline = start_time + 40
    g_score, g_U = greedy_construct(greedy_deadline)
    
    best_score = g_score
    best_U = list(g_U)
    
    # Phase 2: Evolutionary search from greedy result (~30s)
    evo_deadline = start_time + 75
    if time.time() < evo_deadline - 5:
        e_score, e_U = evolutionary_search(best_U, evo_deadline, pop_size=15, generations=80)
        if e_score > best_score:
            best_score = e_score
            best_U = e_U
    
    # Phase 3: Local refinement (~30s)
    if time.time() < deadline - 3:
        l_score, l_U = local_refine(best_U, deadline)
        if l_score > best_score:
            best_score = l_score
            best_U = l_U
    
    # Ensure 0 is in the set and it's valid
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
