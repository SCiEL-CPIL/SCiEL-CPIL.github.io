import itertools
import random
import time

def construct():
    """
    Construct a 2-coloring of triples avoiding red K4^(3) and blue K5^(3).
    Uses structured coloring via F_3 pair assignments with genetic algorithm optimization.
    """
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for overhead
    
    # Try n=34 first, then fall back to lower n
    for n in range(34, 8, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Precompute all triples in lexicographic order
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        
        # Precompute all 4-sets and 5-sets for penalty calculation
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        # Map each triple to its index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # For each 4-set, store the 4 triple indices
        four_set_triple_indices = []
        for s in four_sets:
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            four_set_triple_indices.append(ts)
            
        # For each 5-set, store the 10 triple indices
        five_set_triple_indices = []
        for s in five_sets:
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            five_set_triple_indices.append(ts)
        
        # Genetic algorithm to find good pair assignments
        # Each pair (i,j) gets a value in {0, 1, 2}
        # Triple (a,b,c) color = 0 if (v_ab + v_bc + v_ac) % 3 == 0 else 1
        
        def evaluate_pair_assignment(pair_values):
            """
            Given pair values (dict from (i,j) to value in {0,1,2}),
            compute colors and penalty.
            """
            # Compute colors for all triples
            colors = []
            for (a, b, c) in triples:
                v_ab = pair_values.get((a, b), pair_values.get((b, a), 0))
                v_bc = pair_values.get((b, c), pair_values.get((c, b), 0))
                v_ac = pair_values.get((a, c), pair_values.get((c, a), 0))
                s = (v_ab + v_bc + v_ac) % 3
                colors.append(0 if s == 0 else 1)
            
            # Calculate penalty: count red K4^(3) and blue K5^(3)
            penalty = 0
            
            # Check 4-sets for all-red
            red_count_4 = [0] * len(four_sets)
            for t_idx, c in enumerate(colors):
                if c == 0:  # Red
                    # This triple belongs to some 4-sets
                    # We need to know which 4-sets contain this triple
                    # For efficiency, precompute triple_to_fours
                    pass
            
            # More efficient: precompute which 4-sets contain each triple
            # For now, use a simpler approach: iterate through 4-sets
            for i, ts in enumerate(four_set_triple_indices):
                all_red = all(colors[t] == 0 for t in ts)
                if all_red:
                    penalty += 1
            
            # Check 5-sets for all-blue
            for j, ts in enumerate(five_set_triple_indices):
                all_blue = all(colors[t] == 1 for t in ts)
                if all_blue:
                    penalty += 1
            
            return colors, penalty
        
        # Genetic algorithm
        num_pairs = n * (n - 1) // 2
        pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize population
        pop_size = 50
        population = []
        for _ in range(pop_size):
            pair_values = {p: random.randint(0, 2) for p in pair_list}
            colors, penalty = evaluate_pair_assignment(pair_values)
            population.append((penalty, pair_values, colors))
        
        # Sort by penalty
        population.sort(key=lambda x: x[0])
        
        # If we already have a solution, return it
        if population[0][0] == 0:
            return {'n': n, 'colors': population[0][2]}
        
        # Genetic algorithm iterations
        best_penalty = population[0][0]
        best_solution = population[0]
        
        max_gens = 100
        for gen in range(max_gens):
            if time.time() - start_time > time_limit:
                break
            
            # Selection: keep top half
            elite_size = pop_size // 2
            new_population = population[:elite_size].copy()
            
            # Crossover and mutation to fill rest
            while len(new_population) < pop_size:
                # Select parents using tournament selection
                parent1_idx = random.randint(0, elite_size - 1)
                parent2_idx = random.randint(0, elite_size - 1)
                p1 = population[parent1_idx][1]
                p2 = population[parent2_idx][1]
                
                # Crossover: mix pair values
                child = {}
                for p in pair_list:
                    if random.random() < 0.5:
                        child[p] = p1[p]
                    else:
                        child[p] = p2[p]
                
                # Mutation: randomly change some pair values
                for p in pair_list:
                    if random.random() < 0.1:
                        child[p] = random.randint(0, 2)
                
                colors, penalty = evaluate_pair_assignment(child)
                new_population.append((penalty, child, colors))
            
            # Sort new population
            new_population.sort(key=lambda x: x[0])
            population = new_population
            
            # Update best
            if population[0][0] < best_penalty:
                best_penalty = population[0][0]
                best_solution = population[0]
                if best_penalty == 0:
                    break
        
        if best_penalty == 0:
            return {'n': n, 'colors': best_solution[2]}
    
    # Fallback: return a small valid coloring
    # Use a simple deterministic coloring for small n
    n = 9
    colors = []
    for a, b, c in itertools.combinations(range(n), 3):
        # Color based on sum modulo 3
        s = (a + b + c) % 3
        colors.append(0 if s == 0 else 1)
    return {'n': n, 'colors': colors}
