import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))
    
    # For each element, which 4-APs does it participate in?
    elem_to_4aps = [[] for _ in range(N)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # For each element, count how many 3-APs it participates in (weight)
    elem_3ap_count = [0] * N
    for ap in three_aps:
        for x in ap:
            elem_3ap_count[x] += 1
            
    # Sort indices by weight descending for greedy construction
    weight_order = sorted(range(N), key=lambda x: (-elem_3ap_count[x], x))

    def count_3aps(sel):
        """Count 3-APs in selected set (sel is a set of 0-indexed elements)."""
        sel_set = sel
        cnt = 0
        for ap in three_aps:
            if ap[0] in sel_set and ap[1] in sel_set and ap[2] in sel_set:
                cnt += 1
        return cnt
        
    def count_4ap_violations(sel):
        """Count 4-APs fully contained in sel."""
        sel_set = sel
        cnt = 0
        for ap in four_aps:
            if ap[0] in sel_set and ap[1] in sel_set and ap[2] in sel_set and ap[3] in sel_set:
                cnt += 1
        return cnt

    def greedy_construct():
        """Greedy: add elements in order of decreasing 3-AP weight, skip if creates 4-AP."""
        selected = set()
        for x in weight_order:
            conflict = False
            for ap_idx in elem_to_4aps[x]:
                ap = four_aps[ap_idx]
                others = [y for y in ap if y != x]
                if all(y in selected for y in others):
                    conflict = True
                    break
            if not conflict:
                selected.add(x)
        return selected

    def greedy_repair(selected):
        """Remove elements to eliminate all 4-APs, keeping as many as possible."""
        selected = set(selected)
        changed = True
        while changed:
            changed = False
            for ap in four_aps:
                if all(x in selected for x in ap):
                    worst = min(ap, key=lambda x: elem_3ap_count[x])
                    selected.discard(worst)
                    changed = True
                    break
        return selected

    def local_search(selected, max_iters=100):
        """Hill-climbing: try adds, removes, swaps to improve 3-AP count while maintaining no 4-APs."""
        selected = set(selected)
        current_score = count_3aps(selected)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            improved = False
            
            # Try adding an element
            candidates = [x for x in range(N) if x not in selected]
            rng.shuffle(candidates)
            for x in candidates:
                conflict = False
                for ap_idx in elem_to_4aps[x]:
                    ap = four_aps[ap_idx]
                    others = [y for y in ap if y != x]
                    if all(y in selected for y in others):
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_sel = selected | {x}
                new_score = count_3aps(new_sel)
                if new_score > current_score:
                    selected = new_sel
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing an element
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for x in sel_list:
                new_sel = selected - {x}
                new_score = count_3aps(new_sel)
                if new_score > current_score:
                    selected = new_sel
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for x_remove in sel_list:
                if time.time() - start_time > TIME_LIMIT - 3:
                    return selected, current_score
                candidates = [x for x in range(N) if x not in selected]
                rng.shuffle(candidates)
                for x_add in candidates:
                    new_sel = (selected - {x_remove}) | {x_add}
                    has_4ap = False
                    for ap_idx in elem_to_4aps[x_add]:
                        ap = four_aps[ap_idx]
                        others = [y for y in ap if y != x_add]
                        if all(y in new_sel for y in others):
                            has_4ap = True
                            break
                    if has_4ap:
                        continue
                    
                    new_score = count_3aps(new_sel)
                    if new_score > current_score:
                        selected = new_sel
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected, current_score

    # Phase 1: Initial Solution via Greedy + Local Search
    best_solution = greedy_construct()
    best_score = count_3aps(best_solution)
    best_solution, best_score = local_search(best_solution, max_iters=200)
    
    # Phase 2: Genetic Algorithm
    # Population size
    POP_SIZE = 20
    population = [best_solution]
    
    # Initialize rest of population with repaired random/greedy sets
    for _ in range(POP_SIZE - 1):
        if rng.random() < 0.5:
            # Random set, repaired
            rand_set = set(rng.sample(range(N), rng.randint(50, 150)))
            ind = greedy_repair(rand_set)
        else:
            # Perturbed greedy
            base = greedy_construct()
            # Remove some random elements
            for _ in range(rng.randint(1, 5)):
                if base:
                    base.discard(rng.choice(list(base)))
            # Fill back greedily
            for x in weight_order:
                if x in base:
                    continue
                conflict = False
                for ap_idx in elem_to_4aps[x]:
                    ap = four_aps[ap_idx]
                    others = [y for y in ap if y != x]
                    if all(y in base for y in others):
                        conflict = True
                        break
                if not conflict:
                    base.add(x)
            ind = base
            
        population.append(ind)
        
    # Evaluate initial population
    scores = [count_3aps(ind) for ind in population]
    
    generation = 0
    max_generations = 1000
    
    while generation < max_generations:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        generation += 1
        
        # Sort population by score
        pop_with_scores = sorted(zip(scores, population), key=lambda x: -x[0])
        scores = [s for s, _ in pop_with_scores]
        population = [ind for _, ind in pop_with_scores]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_solution = set(population[0])
            
        # Periodically polish the best solution with local search
        if generation % 10 == 0:
            polished, polished_score = local_search(set(population[0]), max_iters=50)
            if polished_score > best_score:
                best_score = polished_score
                best_solution = polished
            # Replace the top individual with the polished one
            population[0] = polished
            scores[0] = polished_score
            # Re-sort
            pop_with_scores = sorted(zip(scores, population), key=lambda x: -x[0])
            scores = [s for s, _ in pop_with_scores]
            population = [ind for _, ind in pop_with_scores]

        # Create offspring
        new_population = []
        
        # Elitism: keep top 2
        new_population.append(population[0])
        new_population.append(population[1])
        
        for _ in range(POP_SIZE - 2):
            # Tournament selection
            p1_idx = rng.randint(0, 4)
            p2_idx = rng.randint(0, 4)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Uniform Crossover
            child = set()
            for x in range(N):
                if rng.random() < 0.5:
                    if x in p1:
                        child.add(x)
                else:
                    if x in p2:
                        child.add(x)
            
            # Mutation: flip some bits
            for _ in range(rng.randint(1, 5)):
                x = rng.randint(0, N - 1)
                if x in child:
                    child.discard(x)
                else:
                    child.add(x)
            
            # Repair to ensure no 4-APs
            child = greedy_repair(child)
            
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        scores = [count_3aps(ind) for ind in population]
        
        # Quick local search on a few random individuals to help convergence
        if generation % 5 == 0:
            for i in rng.sample(range(POP_SIZE), min(3, POP_SIZE)):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                improved_sol, improved_score = local_search(set(population[i]), max_iters=20)
                if improved_score > scores[i]:
                    population[i] = improved_sol
                    scores[i] = improved_score

    # Final polish
    if time.time() - start_time < TIME_LIMIT - 3:
        best_solution, best_score = local_search(best_solution, max_iters=200)
        
    # Verify no 4-APs
    assert count_4ap_violations(best_solution) == 0, "4-AP violation found!"
    
    return sorted(x + 1 for x in best_solution)
