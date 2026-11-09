"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
import numpy as np

def _precompute_constraints():
    """Precompute all K(2,2,2) constraint patterns and their triple indices."""
    all_triples = list(itertools.combinations(range(16), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # A K(2,2,2) is defined by 3 disjoint pairs (a,a'), (b,b'), (c,c')
    # The 8 transversal triples are all combinations choosing one from each pair
    constraints = []  # list of lists of triple indices
    
    for a in range(16):
        for a_prime in range(a+1, 16):
            for b in range(16):
                if b in (a, a_prime):
                    continue
                for b_prime in range(16):
                    if b_prime in (a, a_prime, b):
                        continue
                    for c in range(16):
                        if c in (a, a_prime, b, b_prime):
                            continue
                        for c_prime in range(16):
                            if c_prime in (a, a_prime, b, b_prime, c):
                                continue
                            
                            # 8 transversal triples
                            transversals = [
                                (a, b, c), (a, b, c_prime),
                                (a, b_prime, c), (a, b_prime, c_prime),
                                (a_prime, b, c), (a_prime, b, c_prime),
                                (a_prime, b_prime, c), (a_prime, b_prime, c_prime)
                            ]
                            
                            idx_list = []
                            valid = True
                            for t in transversals:
                                ts = tuple(sorted(t))
                                if ts not in triple_to_idx:
                                    valid = False
                                    break
                                idx_list.append(triple_to_idx[ts])
                            
                            if valid:
                                constraints.append(idx_list)
    
    return all_triples, triple_to_idx, constraints

def _count_violations(bitmask, constraints):
    """Count how many K(2,2,2) constraints are violated (all 8 triples present)."""
    violations = 0
    for constraint in constraints:
        # Check if all 8 triples in this constraint are present in bitmask
        all_present = True
        for idx in constraint:
            if not (bitmask >> idx) & 1:
                all_present = False
                break
        if all_present:
            violations += 1
    return violations

def _ga_fitness(bitmask, constraints):
    """
    Fitness = number of selected triples - penalty * number of violations.
    We want to maximize this.
    """
    n_selected = bin(bitmask).count('1')
    violations = _count_violations(bitmask, constraints)
    return n_selected - 100 * violations

def _greedy_initial():
    """Generate an initial valid triple set using greedy."""
    all_triples, triple_to_idx, constraints = _greedy_initial._cache
    random.seed(random.randint(0, 2**31 - 1))
    
    # Use a simpler greedy: try to add triples, checking constraints incrementally
    # For speed, use a constraint-count approach
    constraint_counts = [0] * len(constraints)
    triple_in_constraints = defaultdict(list)
    
    for ci, constraint in enumerate(constraints):
        for idx in constraint:
            triple_in_constraints[idx].append(ci)
    
    _greedy_initial._cache_triple_in_constraints = triple_in_constraints
    _greedy_initial._cache_constraint_counts = constraint_counts
    
    all_indices = list(range(len(all_triples)))
    random.shuffle(all_indices)
    
    selected = set()
    for idx in all_indices:
        # Check if adding this triple would complete any constraint
        can_add = True
        for ci in triple_in_constraints.get(idx, []):
            if constraint_counts[ci] == 7:
                can_add = False
                break
        
        if can_add:
            selected.add(idx)
            for ci in triple_in_constraints.get(idx, []):
                constraint_counts[ci] += 1
    
    return selected, all_triples

def _local_search_greedy(selected, all_triples, time_limit=5.0):
    """Improve the triple set using greedy local search."""
    start_time = time.time()
    all_triples, triple_to_idx, constraints = _greedy_initial._cache
    triple_in_constraints = _greedy_initial._cache_triple_in_constraints
    constraint_counts = _greedy_initial._cache_constraint_counts
    
    # Rebuild constraint counts from selected
    for ci in range(len(constraints)):
        constraint_counts[ci] = 0
    for idx in selected:
        for ci in triple_in_constraints.get(idx, []):
            constraint_counts[ci] += 1
    
    all_indices = list(range(len(all_triples)))
    missing = [i for i in all_indices if i not in selected]
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        random.shuffle(missing)
        
        for idx in missing:
            can_add = True
            for ci in triple_in_constraints.get(idx, []):
                if constraint_counts[ci] == 7:
                    can_add = False
                    break
            
            if can_add:
                selected.add(idx)
                for ci in triple_in_constraints.get(idx, []):
                    constraint_counts[ci] += 1
                missing.remove(idx)
                improved = True
                break
    
    # Try 1-remove, 2-add moves
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        selected_list = list(selected)
        random.shuffle(selected_list)
        
        for idx_rem in selected_list[:30]:
            # Remove idx_rem
            selected.remove(idx_rem)
            for ci in triple_in_constraints.get(idx_rem, []):
                constraint_counts[ci] -= 1
            
            # Try to add as many missing triples as possible
            added = []
            remaining_missing = [i for i in missing if i != idx_rem]
            random.shuffle(remaining_missing)
            
            for idx_add in remaining_missing:
                can_add = True
                for ci in triple_in_constraints.get(idx_add, []):
                    if constraint_counts[ci] == 7:
                        can_add = False
                        break
                
                if can_add:
                    selected.add(idx_add)
                    for ci in triple_in_constraints.get(idx_add, []):
                        constraint_counts[ci] += 1
                    added.append(idx_add)
            
            if len(added) > 1:
                # Net gain
                for idx_add in added:
                    missing.remove(idx_add)
                improved = True
                break
            else:
                # Revert
                for idx_add in added:
                    selected.remove(idx_add)
                    for ci in triple_in_constraints.get(idx_add, []):
                        constraint_counts[ci] -= 1
                selected.add(idx_rem)
                for ci in triple_in_constraints.get(idx_rem, []):
                    constraint_counts[ci] += 1
    
    return selected

def _genetic_algorithm(all_triples, constraints, pop_size=50, generations=50, time_limit=30.0):
    """
    Genetic algorithm with bitset-based fitness.
    Each chromosome is a set of selected triple indices.
    """
    start_time = time.time()
    n_triples = len(all_triples)
    triple_in_constraints = _greedy_initial._cache_triple_in_constraints
    
    def make_bitmask(selected):
        bm = 0
        for idx in selected:
            bm |= (1 << idx)
        return bm
    
    def bitmask_to_set(bm):
        s = set()
        for i in range(n_triples):
            if (bm >> i) & 1:
                s.add(i)
        return s
    
    # Initialize population with diverse greedy solutions
    population = []
    for _ in range(pop_size):
        sel, _ = _greedy_initial()
        population.append(sel)
    
    best_sel = max(population, key=len)
    best_size = len(best_sel)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate fitness
        def fitness(sel):
            n_sel = len(sel)
            # Count violations
            cc = [0] * len(constraints)
            for idx in sel:
                for ci in triple_in_constraints.get(idx, []):
                    cc[ci] += 1
            violations = sum(1 for c in cc if c == 8)
            return n_sel - 100 * violations
        
        fitnesses = [fitness(sel) for sel in population]
        
        # Selection: tournament
        def tournament_select(k=3):
            indices = random.sample(range(pop_size), k)
            best_idx = max(indices, key=lambda i: fitnesses[i])
            return population[best_idx]
        
        new_population = []
        
        # Elitism: keep top 2
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        new_population.append(population[sorted_indices[0]])
        new_population.append(population[sorted_indices[1]])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: uniform
            child = set()
            for idx in range(n_triples):
                if random.random() < 0.5:
                    if idx in parent1:
                        child.add(idx)
                else:
                    if idx in parent2:
                        child.add(idx)
            
            # Mutation: flip some bits
            for _ in range(random.randint(1, 5)):
                if random.random() < 0.5:
                    # Add a random triple
                    idx = random.randint(0, n_triples - 1)
                    child.add(idx)
                else:
                    # Remove a random triple from child
                    if child:
                        idx = random.choice(list(child))
                        child.remove(idx)
            
            # Repair: remove violations greedily
            cc = [0] * len(constraints)
            for idx in child:
                for ci in triple_in_constraints.get(idx, []):
                    cc[ci] += 1
            
            # Remove triples from violated constraints
            changed = True
            while changed:
                changed = False
                for ci in range(len(constraints)):
                    if cc[ci] == 8:
                        # Remove one triple from this constraint
                        constraint = constraints[ci]
                        in_child = [idx for idx in constraint if idx in child]
                        if in_child:
                            idx_to_remove = random.choice(in_child)
                            child.remove(idx_to_remove)
                            for ci2 in triple_in_constraints.get(idx_to_remove, []):
                                cc[ci2] -= 1
                            changed = True
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        for sel in population:
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
    
    return best_sel

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute
    all_triples, triple_to_idx, constraints = _precompute_constraints()
    _greedy_initial._cache = (all_triples, triple_to_idx, constraints)
    
    # Initialize cache structures
    triple_in_constraints = defaultdict(list)
    for ci, constraint in enumerate(constraints):
        for idx in constraint:
            triple_in_constraints[idx].append(ci)
    _greedy_initial._cache_triple_in_constraints = triple_in_constraints
    _greedy_initial._cache_constraint_counts = [0] * len(constraints)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple greedy + local search iterations
    while time.time() < deadline - 40:
        g_set, _ = _greedy_initial()
        ls_set = _local_search_greedy(g_set, all_triples, time_limit=3.0)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_set = ls_set
    
    # Phase 2: Genetic algorithm to escape local optima
    if time.time() < deadline - 10:
        ga_time = min(30.0, deadline - 10 - time.time())
        ga_set = _genetic_algorithm(all_triples, constraints, pop_size=30, generations=40, time_limit=ga_time)
        
        if len(ga_set) > best_size:
            best_size = len(ga_set)
            best_set = ga_set
    
    # Final polish: try to add more triples
    if time.time() < deadline - 5:
        final_set = _local_search_greedy(set(best_set), all_triples, time_limit=min(5.0, deadline - 5 - time.time()))
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = final_set
    
    return [list(all_triples[i]) for i in sorted(best_set)]
# EVOLVE-BLOCK-END
