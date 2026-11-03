import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def score_set(U):
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    # Initial population
    # We want sets that are "spread out" to have large difference sets
    # but "clustered" in a way that sums overlap to have small sum sets?
    # Actually, for U+U to be small, elements should be close together (geometric progression?).
    # For U-U to be large, elements should be far apart (Golomb ruler?).
    # This is a tension.
    
    # Let's start with a few diverse initial candidates
    def random_initial_set():
        # Random size between 10 and 50
        n = random.randint(10, 50)
        # Random max value between 100 and 10000
        max_val = random.randint(100, 10000)
        # Sample n-1 distinct values from 1..max_val
        # Ensure 0 is included
        if n == 1:
            return [0]
        samples = random.sample(range(1, max_val + 1), n - 1)
        samples.append(0)
        return sorted(samples)

    def geometric_set(r, n):
        # Geometric progression often has small sumsets
        # U = {0, 1, r, r^2, ..., r^(n-1)}
        # Sumset size is roughly n^2/2? Difference set size is roughly n^2/2?
        # Let's see.
        U = [0]
        current = 1
        for _ in range(n - 1):
            U.append(current)
            current *= r
        return U

    def golomb_like_set(n):
        # Approximate Golomb ruler: squares
        # U = {i^2 for i in 0..n-1}
        # Differences are distinct? No, but many.
        # Sums?
        return [i*i for i in range(n)]

    population = []
    # Add some structured starts
    for r in [2, 3, 4, 5, 10]:
        for n in [5, 10, 15, 20]:
            population.append(geometric_set(r, n))
    for n in [10, 20, 30, 50]:
        population.append(golomb_like_set(n))
    
    # Add random starts
    for _ in range(20):
        population.append(random_initial_set())

    # Evaluate initial population
    scored_population = [(score_set(ind), ind) for ind in population]
    scored_population.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_population[0][0]
    best_set = scored_population[0][1]

    # Genetic Algorithm
    pop_size = 30
    # Ensure we have enough individuals
    while len(scored_population) < pop_size:
        ind = random_initial_set()
        scored_population.append((score_set(ind), ind))
        scored_population.sort(key=lambda x: x[0], reverse=True)
    
    # Keep only top pop_size
    scored_population = scored_population[:pop_size]

    mutation_rate = 0.3
    crossover_rate = 0.7

    while time.time() < deadline:
        new_population = []
        
        # Elitism: keep the best 2
        for i in range(2):
            new_population.append(scored_population[i])

        while len(new_population) < pop_size:
            # Selection: Tournament selection
            def select():
                # Pick 3 random individuals, return the best
                candidates = random.sample(scored_population, min(3, len(scored_population)))
                return max(candidates, key=lambda x: x[0])[1]
            
            if random.random() < crossover_rate and len(scored_population) >= 2:
                p1 = select()
                p2 = select()
                
                # Crossover: Uniform crossover on sorted lists
                # This is tricky because lengths differ.
                # Let's try: take max(len(p1), len(p2)) elements.
                # For each position, pick from p1 or p2?
                # Better: Interleave?
                # Simplest: Pick a random subset of indices from the union?
                # Let's try "Blending" or "Subsequence" crossover.
                
                # Strategy: 
                # 1. Sort p1 and p2 (they are already sorted)
                # 2. Create a new set by picking elements from p1 and p2
                #    such that the resulting set has size ~ (len(p1)+len(p2))/2
                #    and maintains some structure.
                
                # Alternative Crossover: 
                # Pick a random split point in the value range?
                # Or just take a random subset of the union?
                
                # Let's try: 
                # Union of p1 and p2, then randomly sample a subset of size N
                union_set = set(p1) | set(p2)
                if not union_set:
                    child = [0]
                else:
                    # Target size
                    target_size = random.randint(min(len(p1), len(p2)), max(len(p1), len(p2)))
                    target_size = min(target_size, len(union_set))
                    if target_size == 0:
                        child = [0]
                    else:
                        # Ensure 0 is in the child
                        if 0 in union_set:
                            child = [0]
                            remaining = target_size - 1
                            others = [x for x in union_set if x != 0]
                            if remaining > 0 and others:
                                child.extend(random.sample(others, min(remaining, len(others))))
                        else:
                            child = random.sample(list(union_set), target_size)
                            child.append(0)
                        child = sorted(set(child))
            else:
                p1 = select()
                child = p1[:]

            # Mutation
            if random.random() < mutation_rate:
                # Type 1: Add a random element
                if random.random() < 0.5:
                    if child:
                        max_val = max(child)
                        # Add a value near the max or random
                        new_val = random.randint(1, max_val * 2)
                        child.append(new_val)
                        child = sorted(set(child))
                # Type 2: Remove a random element (not 0)
                elif random.random() < 0.5:
                    removable = [x for x in child if x != 0]
                    if removable:
                        idx = random.choice(removable)
                        child.remove(idx)
                # Type 3: Shift/Scale
                else:
                    # Scale all non-zero elements by a small factor?
                    # Or add a constant?
                    # Let's try: replace a random element with a nearby one
                    if child:
                        idx = random.choice([x for x in child if x != 0] or [0])
                        if idx != 0:
                            new_val = max(1, idx + random.randint(-10, 10))
                            if new_val not in child:
                                child.remove(idx)
                                child.append(new_val)
                                child = sorted(set(child))
            
            # Ensure 0 is in the set
            if 0 not in child:
                child.append(0)
                child = sorted(set(child))
            
            if not child:
                child = [0]
                
            new_population.append((score_set(child), child))
        
        # Evaluate new population
        scored_population = sorted(new_population, key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored_population[0][0] > best_score:
            best_score = scored_population[0][0]
            best_set = scored_population[0][1]
            
            # Optional: If we find a very good score, we can tighten the search
            # But for now, just continue.

    return best_set
