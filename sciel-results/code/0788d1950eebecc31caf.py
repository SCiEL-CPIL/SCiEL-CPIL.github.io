import random
import math

def construct():
    N = 256
    # Generate all 4-APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Pre-compute for each index, which APs it belongs to and what the other indices are
    # We store the indices of the other 3 elements in the AP
    aps_by_index = [[] for _ in range(N)]
    for ap in aps:
        i1, i2, i3, i4 = ap
        aps_by_index[i1].append((i2, i3, i4))
        aps_by_index[i2].append((i1, i3, i4))
        aps_by_index[i3].append((i1, i2, i4))
        aps_by_index[i4].append((i1, i2, i3))

    random.seed(42)
    
    def calculate_score(colors):
        count = 0
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c1 == c2 == c3 == c4:
                count += 1
        return count

    def local_search(colors, max_steps=1000):
        """
        Perform greedy local search (Hill Climbing) on the given coloring.
        Returns the improved coloring and its score.
        """
        current_score = calculate_score(colors)
        improved = True
        steps = 0
        while improved and steps < max_steps:
            improved = False
            steps += 1
            for i in range(N):
                # Try flipping i
                old_color = colors[i]
                new_color = 1 - old_color
                
                # Calculate delta E
                delta_e = 0
                for others in aps_by_index[i]:
                    o1, o2, o3 = others
                    c_o1, c_o2, c_o3 = colors[o1], colors[o2], colors[o3]
                    
                    if c_o1 == 0 and c_o2 == 0 and c_o3 == 0:
                        if old_color == 0:
                            delta_e -= 1
                        else:
                            delta_e += 1
                    elif c_o1 == 1 and c_o2 == 1 and c_o3 == 1:
                        if old_color == 1:
                            delta_e -= 1
                        else:
                            delta_e += 1
                
                if delta_e < 0:
                    colors[i] = new_color
                    current_score += delta_e
                    improved = True
        
        return colors, current_score

    # Genetic Algorithm Parameters
    pop_size = 50
    generations = 50
    tournament_size = 5
    mutation_rate = 0.05
    elitism_count = 5

    # Initialize Population
    population = []
    for _ in range(pop_size):
        colors = [random.randint(0, 1) for _ in range(N)]
        population.append(colors)

    best_overall_score = float('inf')
    best_overall_colors = None

    for gen in range(generations):
        # Evaluate current population
        scored_pop = []
        for colors in population:
            score = calculate_score(colors)
            scored_pop.append((score, colors))
            if score < best_overall_score:
                best_overall_score = score
                best_overall_colors = colors[:]

        # Sort by score
        scored_pop.sort(key=lambda x: x[0])
        
        # Elitism: Keep the best ones
        new_population = []
        for i in range(elitism_count):
            new_population.append(scored_pop[i][1][:])

        # Fill the rest of the population
        while len(new_population) < pop_size:
            # Tournament Selection
            parents = []
            for _ in range(2):
                tournament = random.sample(scored_pop, tournament_size)
                winner = min(tournament, key=lambda x: x[0])
                parents.append(winner[1])
            
            # Crossover (Uniform)
            child = [0] * N
            for i in range(N):
                if random.random() < 0.5:
                    child[i] = parents[0][i]
                else:
                    child[i] = parents[1][i]
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Local Search Refinement
            # Limit steps to keep runtime manageable
            child, _ = local_search(child, max_steps=200)
            
            new_population.append(child)

        population = new_population

    return best_overall_colors
