import time
import random
from math import gcd

def _lcm_range(n):
    result = 1
    for i in range(2, n + 1):
        result = result * i // gcd(result, i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 30
    lcm_val = _lcm_range(n)
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Best solution found so far
    best_S = float('inf')
    best_a = None
    
    # Helper to evaluate sum
    def eval_sum(a):
        return sum(a[i] * weights[i] for i in range(n))
    
    # Helper to update best
    def update_best(a, S):
        nonlocal best_S, best_a
        if S != 0 and abs(S) < best_S:
            best_S = abs(S)
            best_a = a[:]
            
    # Local Search: Hill Climbing with 1-bit flips
    # Returns improved solution and its sum
    def local_search(a, S, max_iters=1000):
        a = a[:]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Randomize order to avoid bias? Or fixed? Fixed is faster but might be biased.
            # Let's use fixed order for speed in the inner loop of GA.
            for idx in range(n):
                old_val = a[idx]
                # Try other 2 values
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[idx]
                    new_S = S + delta
                    if new_S == 0:
                        continue
                    if abs(new_S) < abs(S):
                        a[idx] = new_val
                        S = new_S
                        improved = True
                        break # Restart scan from 0? Or continue?
                        # Restarting scan is safer for steepest descent
                        # But expensive. Let's just break inner and continue outer while loop.
                        # Actually, if we change a[idx], other indices might now allow improvement.
                        # Breaking inner and continuing outer while loop restarts the for loop.
                        break
        return a, S

    # Initialize Population
    pop_size = 40
    population = []
    
    # Seed solution: 1/29 - 1/30
    seed_a = [0] * n
    seed_a[28] = 1
    seed_a[29] = -1
    seed_S = eval_sum(seed_a)
    update_best(seed_a, seed_S)
    population.append((seed_a, seed_S))
    
    # Add some random initial solutions
    for _ in range(pop_size - 1):
        a = [random.choice([-1, 0, 1]) for _ in range(n)]
        S = eval_sum(a)
        if S == 0:
            a[0] = 1
            S = eval_sum(a)
        population.append((a, S))
        
    # Evaluate initial population with local search to ensure they are local optima
    # This is expensive, so we might limit it or do it in the first generation.
    # Let's run local search on the seed and a few others to get good starting points.
    for i in range(min(5, len(population))):
        a, S = population[i]
        a, S = local_search(a, S, max_iters=50)
        population[i] = (a, S)
        update_best(a, S)

    # Genetic Algorithm Loop
    # Parameters
    tournament_size = 5
    mutation_rate = 0.3
    crossover_rate = 0.8
    
    while time.time() < deadline:
        # Selection and Crossover
        offspring = []
        
        # Elitism: Keep best 2
        population.sort(key=lambda x: abs(x[1]) if x[1] != 0 else float('inf'))
        elites = population[:2]
        
        for _ in range(pop_size):
            if random.random() < crossover_rate and len(offspring) < pop_size - 2:
                # Tournament selection
                p1 = random.sample(population, tournament_size)[0]
                p2 = random.sample(population, tournament_size)[0]
                
                a1, S1 = p1
                a2, S2 = p2
                
                # Uniform Crossover
                a_child = [0] * n
                S_child = 0
                for i in range(n):
                    if random.random() < 0.5:
                        a_child[i] = a1[i]
                    else:
                        a_child[i] = a2[i]
                    S_child += a_child[i] * weights[i]
            else:
                # Clone from tournament
                p1 = random.sample(population, tournament_size)[0]
                a_child = p1[0][:]
                S_child = p1[1]
                
            # Mutation
            if random.random() < mutation_rate:
                # Flip 1 or 2 bits
                num_flips = random.choice([1, 2])
                for _ in range(num_flips):
                    idx = random.randint(0, n - 1)
                    old_val = a_child[idx]
                    # Choose new val different from old
                    options = [v for v in (-1, 0, 1) if v != old_val]
                    new_val = random.choice(options)
                    delta = (new_val - old_val) * weights[idx]
                    S_child += delta
                    a_child[idx] = new_val
            
            # Local Search on offspring
            # Limit iterations to keep GA moving
            a_child, S_child = local_search(a_child, S_child, max_iters=20)
            
            if S_child == 0:
                # Force non-zero
                a_child[0] = 1
                S_child = eval_sum(a_child)
            
            offspring.append((a_child, S_child))
            update_best(a_child, S_child)
            
        # Combine elites and offspring
        population = elites + offspring[:pop_size - len(elites)]
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final Polish on the best solution found
    if best_a is not None and time.time() < deadline - 1:
        a = best_a[:]
        S = eval_sum(a)
        # Run a more intensive local search on the global best
        a, S = local_search(a, S, max_iters=2000)
        update_best(a, S)
        
    return best_a
