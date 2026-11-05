"""Optimized construction for max 3-AP, no 4-AP subset of [1,256]."""
import time
import random
import sys
from collections import Counter

N = 256
DEADLINE_MARGIN = 10  # seconds before the 90s limit

# Precompute all 4-term APs
FAP = []  # list of (a, b, c, d) tuples
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            FAP.append((a, b, c, e))

FAP_COUNT = len(FAP)

# Precompute bitmask for each 4-AP for fast validity check
FAP_MASKS = []
for (a, b, c, e) in FAP:
    mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
    FAP_MASKS.append(mask)

# Precompute 3-AP contribution: for each element x, which other elements form 3-APs with x
# A 3-AP (a, b, c) with a < b < c and b-a = c-b = d
# For element x, it can be:
#   - first term: x, x+d, x+2d
#   - middle term: x-d, x, x+d  
#   - last term: x-2d, x-d, x
# We'll precompute the count of 3-APs that include a given pair of elements

# For fast 3-AP counting using bitmasks:
# 3-AP (a, b, c) where a + c = 2b, a < b < c
# For a given set S (as bitmask), count pairs (a, c) in S with a < c and (a+c)/2 in S

# Precompute: for each b (middle), list of (a, c) pairs where a + c = 2b, a < b < c
MID_PAIRS = [None] * (N + 1)
for b in range(1, N + 1):
    pairs = []
    for d in range(1, min(b, N - b + 1)):
        a = b - d
        c = b + d
        if 1 <= a and c <= N:
            pairs.append((a, c))
    MID_PAIRS[b] = pairs

def count_3aps(selected):
    """Count distinct 3-term APs with positive difference in selected set."""
    sel = set(selected)
    count = 0
    for b in range(1, N + 1):
        if b not in sel:
            continue
        for (a, c) in MID_PAIRS[b]:
            if a in sel and c in sel:
                count += 1
    return count

def has_4ap(selected):
    """Check if selected set has any nonconstant 4-term AP."""
    sel = set(selected)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return True
    return False

def to_mask(selected):
    """Convert set to bitmask."""
    mask = 0
    for x in selected:
        mask |= (1 << (x - 1))
    return mask

def from_mask(mask):
    """Convert bitmask to sorted list."""
    result = []
    for i in range(1, N + 1):
        if mask & (1 << (i - 1)):
            result.append(i)
    return result

def count_3aps_mask(mask):
    """Count 3-APs using bitmask representation."""
    count = 0
    for b in range(1, N + 1):
        if not (mask & (1 << (b - 1))):
            continue
        for (a, c) in MID_PAIRS[b]:
            if (mask & (1 << (a - 1))) and (mask & (1 << (c - 1))):
                count += 1
    return count

def has_4ap_mask(mask):
    """Check 4-AP validity using bitmask."""
    for fm in FAP_MASKS:
        if (mask & fm) == fm:
            return True
    return False

def greedy_construct():
    """Greedy: add elements that maximize 3-AP contribution while avoiding 4-APs."""
    selected = set()
    for x in range(1, N + 1):
        candidate = selected | {x}
        if not has_4ap(candidate):
            selected.add(x)
    return sorted(selected)

def greedy_reverse_construct():
    """Start with all elements, remove those in 4-APs to maximize remaining 3-APs."""
    selected = set(range(1, N + 1))
    changed = True
    while changed:
        changed = False
        violating = []
        for (a, b, c, e) in FAP:
            if a in selected and b in selected and c in selected and e in selected:
                violating.append((a, b, c, e))
        if not violating:
            break
        elem_4ap_count = Counter()
        for (a, b, c, e) in violating:
            for elem in (a, b, c, e):
                elem_4ap_count[elem] += 1
        worst_elem = max(elem_4ap_count, key=lambda x: elem_4ap_count[x])
        selected.remove(worst_elem)
        changed = True
    return sorted(selected)

def local_search(initial, time_limit=5.0):
    """Local search: try adding/removing elements to improve 3-AP count."""
    start = time.time()
    selected = set(initial)
    current_score = count_3aps(selected)
    
    best_selected = set(selected)
    best_score = current_score
    
    while time.time() - start < time_limit:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if x in selected:
                continue
            candidate = selected | {x}
            if has_4ap(candidate):
                continue
            new_score = count_3aps(candidate)
            if new_score > current_score:
                selected = candidate
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
                break
        if not improved:
            # Try removing elements
            for x in list(selected):
                candidate = selected - {x}
                new_score = count_3aps(candidate)
                if new_score > current_score:
                    selected = candidate
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
                    break
        if not improved:
            # Try swaps
            for x in range(1, N + 1):
                if x in selected:
                    continue
                for y in list(selected):
                    candidate = (selected - {y}) | {x}
                    if has_4ap(candidate):
                        continue
                    new_score = count_3aps(candidate)
                    if new_score > current_score:
                        selected = candidate
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = set(selected)
                        break
                if improved:
                    break
            if not improved:
                break
    
    return sorted(best_selected), best_score

def genetic_algorithm(initials, pop_size=20, generations=50, time_limit=20.0, mutation_rate=0.1):
    """Genetic algorithm to evolve 4-AP-free subsets maximizing 3-AP count."""
    start = time.time()
    rng = random.Random(42)
    
    # Initialize population
    population = []
    for ind in initials:
        population.append(set(ind))
    
    # Pad population with variations if needed
    while len(population) < pop_size:
        base = population[rng.randrange(len(population))]
        mutated = set(base)
        # Apply random mutations
        for _ in range(rng.randint(1, 5)):
            op = rng.random()
            if op < 0.4:
                # Add random element
                x = rng.randint(1, N)
                if x not in mutated:
                    candidate = mutated | {x}
                    if not has_4ap(candidate):
                        mutated = candidate
            elif op < 0.8:
                # Remove random element
                if mutated:
                    x = rng.choice(list(mutated))
                    mutated.discard(x)
            else:
                # Swap
                if mutated:
                    x = rng.choice(list(mutated))
                    y = rng.randint(1, N)
                    if y not in mutated:
                        candidate = (mutated - {x}) | {y}
                        if not has_4ap(candidate):
                            mutated = candidate
        population.append(mutated)
    
    # Evaluate initial population
    scores = [count_3aps(ind) for ind in population]
    
    best_score = max(scores)
    best_individual = set(population[scores.index(best_score)])
    
    for gen in range(generations):
        if time.time() - start > time_limit:
            break
        
        # Sort by fitness (descending)
        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        population = [population[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        # Elitism: keep top individuals
        elite_count = max(2, pop_size // 5)
        new_population = [set(population[i]) for i in range(elite_count)]
        new_scores = [scores[i] for i in range(elite_count)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select(pop, sc, k=3):
                candidates = rng.sample(range(len(pop)), min(k, len(pop)))
                best_c = max(candidates, key=lambda i: sc[i])
                return set(pop[best_c])
            
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            # Crossover: uniform crossover
            child = set()
            for x in range(1, N + 1):
                if rng.random() < 0.5:
                    if x in parent1:
                        child.add(x)
                else:
                    if x in parent2:
                        child.add(x)
            
            # Ensure child is valid (no 4-AP)
            if has_4ap(child):
                # Repair: remove elements from 4-APs
                changed = True
                while changed:
                    changed = False
                    for (a, b, c, e) in FAP:
                        if a in child and b in child and c in child and e in child:
                            # Remove the element with highest 4-AP count
                            counts = Counter()
                            for (a2, b2, c2, e2) in FAP:
                                if a2 in child and b2 in child and c2 in child and e2 in child:
                                    for elem in (a2, b2, c2, e2):
                                        counts[elem] += 1
                            worst = max(counts, key=lambda x: counts[x])
                            child.discard(worst)
                            changed = True
                            break
            
            # Mutation
            if rng.random() < mutation_rate:
                for _ in range(rng.randint(1, 3)):
                    op = rng.random()
                    if op < 0.4:
                        x = rng.randint(1, N)
                        if x not in child:
                            candidate = child | {x}
                            if not has_4ap(candidate):
                                child = candidate
                    elif op < 0.8:
                        if child:
                            x = rng.choice(list(child))
                            child.discard(x)
                    else:
                        if child:
                            x = rng.choice(list(child))
                            y = rng.randint(1, N)
                            if y not in child:
                                candidate = (child - {x}) | {y}
                                if not has_4ap(candidate):
                                    child = candidate
            
            new_population.append(child)
            new_scores.append(count_3aps(child))
        
        population = new_population
        scores = new_scores
        
        # Track best
        gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_individual = set(population[gen_best_idx])
    
    return sorted(best_individual), best_score

def construct():
    start_time = time.time()
    
    # Try multiple strategies
    best_result = []
    best_score = -1
    
    # Strategy 1: Greedy forward
    if time.time() - start_time < DEADLINE_MARGIN:
        r1 = greedy_construct()
        s1 = count_3aps(r1)
        if s1 > best_score:
            best_score = s1
            best_result = r1
        # Local search from this
        if time.time() - start_time < DEADLINE_MARGIN:
            r1_ls, s1_ls = local_search(r1, time_limit=min(3.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if s1_ls > best_score:
                best_score = s1_ls
                best_result = r1_ls
    
    # Strategy 2: Greedy reverse
    if time.time() - start_time < DEADLINE_MARGIN:
        r2 = greedy_reverse_construct()
        s2 = count_3aps(r2)
        if s2 > best_score:
            best_score = s2
            best_result = r2
        # Local search from this
        if time.time() - start_time < DEADLINE_MARGIN:
            r2_ls, s2_ls = local_search(r2, time_limit=min(3.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if s2_ls > best_score:
                best_score = s2_ls
                best_result = r2_ls
    
    # Strategy 3: Randomized greedy with multiple restarts
    if time.time() - start_time < DEADLINE_MARGIN:
        for trial in range(10):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            random.seed(trial)
            order = list(range(1, N + 1))
            random.shuffle(order)
            selected = set()
            for x in order:
                candidate = selected | {x}
                if not has_4ap(candidate):
                    selected.add(x)
            r3 = sorted(selected)
            s3 = count_3aps(r3)
            if s3 > best_score:
                best_score = s3
                best_result = r3
    
    # Strategy 4: Genetic algorithm seeded with best results found so far
    if best_result and time.time() - start_time < DEADLINE_MARGIN:
        elapsed = time.time() - start_time
        remaining = DEADLINE_MARGIN - elapsed
        if remaining > 5:
            # Create diverse initial population
            initials = [best_result]
            # Add some mutated versions
            for _ in range(5):
                mutated = set(best_result)
                for _ in range(random.randint(1, 5)):
                    op = random.random()
                    if op < 0.4:
                        x = random.randint(1, N)
                        if x not in mutated:
                            candidate = mutated | {x}
                            if not has_4ap(candidate):
                                mutated = candidate
                    elif op < 0.8:
                        if mutated:
                            x = random.choice(list(mutated))
                            mutated.discard(x)
                    else:
                        if mutated:
                            x = random.choice(list(mutated))
                            y = random.randint(1, N)
                            if y not in mutated:
                                candidate = (mutated - {x}) | {y}
                                if not has_4ap(candidate):
                                    mutated = candidate
                initials.append(sorted(mutated))
            
            ga_result, ga_score = genetic_algorithm(
                initials, 
                pop_size=15,
                generations=30,
                time_limit=min(remaining - 5, 25.0),
                mutation_rate=0.15
            )
            if ga_score > best_score:
                best_score = ga_score
                best_result = ga_result
    
    # Final local search on best
    if best_result and time.time() - start_time < DEADLINE_MARGIN:
        final_ls, final_s = local_search(best_result, time_limit=min(10.0, DEADLINE_MARGIN - (time.time() - start_time)))
        if final_s > best_score:
            best_score = final_s
            best_result = final_ls
    
    if not best_result:
        # Fallback: just return a small valid set
        best_result = [1]
    
    return best_result

# Verify no 4-APs in the result
def validate(result):
    sel = set(result)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps(result)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
