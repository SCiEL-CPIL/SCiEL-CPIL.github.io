"""Rewritten program: Genetic Algorithm and Exhaustive Search for optimal finite sets."""
import math
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = np.array(sorted(set(U)), dtype=int)
        if len(U) == 0 or U[0] != 0:
            return 0.0
        if len(U) == 1:
            return 1.0
        
        # Compute sumset and difference set sizes
        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        if n_sums == 0:
            return 0.0
            
        ratio = n_diffs / n_sums
        if ratio <= 0:
            return 0.0
            
        max_u = U[-1]
        if max_u == 0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
            
        return 1 + math.log(ratio) / denom

    best_set = [0]
    best_score = 0.0
    
    # Phase 1: Exhaustive search for small sets
    # We suspect small sets with small max U yield high scores
    max_val_exhaustive = 15
    max_size_exhaustive = 6
    
    # Precompute all combinations
    for k in range(3, max_size_exhaustive + 1):
        for combo in combinations(range(1, max_val_exhaustive + 1), k):
            U = [0] + list(combo)
            s = score_of(U)
            if s > best_score:
                best_score = s
                best_set = U

    # Phase 2: Genetic Algorithm for larger sets or refined search
    # Use the best found so far as a seed
    random.seed(42)
    
    pop_size = 50
    generations = 100
    max_val_ga = 30
    min_size_ga = 4
    max_size_ga = 12
    
    def create_individual():
        size = random.randint(min_size_ga, max_size_ga)
        vals = sorted(random.sample(range(1, max_val_ga + 1), size - 1))
        return [0] + vals

    def crossover(p1, p2):
        # Simple splice crossover
        size = random.randint(min_size_ga, max_size_ga)
        # Combine elements from both, keep unique, sort, truncate
        combined = set(p1) | set(p2)
        # Remove 0 for selection
        combined.discard(0)
        vals = sorted(list(combined))
        if len(vals) < size - 1:
            # Need more elements
            missing = random.sample([x for x in range(1, max_val_ga + 1) if x not in combined], size - 1 - len(vals))
            vals.extend(missing)
            vals = sorted(set(vals))
        # Take first size-1 elements? Or random sample?
        # Random sample preserves diversity better
        if len(vals) > size - 1:
            vals = random.sample(vals, size - 1)
        return [0] + sorted(vals)

    def mutate(ind):
        # Add, remove, or replace
        op = random.choice(['add', 'remove', 'replace'])
        ind = list(ind)
        if op == 'add' and len(ind) < max_size_ga:
            x = random.randint(1, max_val_ga)
            if x not in ind:
                ind.append(x)
                ind.sort()
        elif op == 'remove' and len(ind) > min_size_ga:
            i = random.randint(1, len(ind) - 1)
            ind.pop(i)
        elif op == 'replace' and len(ind) > 1:
            i = random.randint(1, len(ind) - 1)
            old = ind[i]
            x = random.randint(1, max_val_ga)
            if x != old and x not in ind:
                ind[i] = x
                ind.sort()
        return ind

    # Initialize population
    population = [create_individual() for _ in range(pop_size - 1)]
    population.append(list(best_set))
    
    for gen in range(generations):
        # Evaluate
        scored_pop = [(score_of(ind), ind) for ind in population]
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_set = scored_pop[0][1]
            
        # Selection: Keep top half
        elite = [ind for _, ind in scored_pop[:pop_size // 2]]
        
        # Generate new population
        new_population = list(elite)
        while len(new_population) < pop_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            new_population.append(child)
            
        population = new_population

    return sorted(best_set)
# EVOLVE-BLOCK-END
