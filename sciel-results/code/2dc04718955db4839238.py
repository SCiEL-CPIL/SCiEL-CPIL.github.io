import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 32
    MAX_VAL = 512
    
    def fitness(S):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d in S.
        S is a sorted list of distinct integers.
        """
        S_set = set(S)
        diffs = set()
        # Iterate over pairs (a, c) where a < c
        # If (a+c) is even, b = (a+c)/2. If b in S, then d = c-b is a valid difference.
        # Note: d = (c-a)/2.
        for i in range(N):
            a = S[i]
            for j in range(i + 1, N):
                c = S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if b in S_set:
                        d = c - b
                        if d > 0:
                            diffs.add(d)
        return len(diffs)

    def random_individual():
        return tuple(sorted(random.sample(range(1, MAX_VAL + 1), N)))

    def mutate(ind):
        # Try swapping two elements or replacing one
        ind_list = list(ind)
        op = random.random()
        if op < 0.5:
            # Swap two random positions
            i, j = random.sample(range(N), 2)
            ind_list[i], ind_list[j] = ind_list[j], ind_list[i]
        else:
            # Replace one element with a random new one
            i = random.randint(0, N - 1)
            new_val = random.randint(1, MAX_VAL)
            if new_val in ind_list:
                # If collision, pick another or just skip mutation for this element
                # Better: try to find a unique value
                for _ in range(10):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val not in ind_list:
                        break
            ind_list[i] = new_val
        # Ensure distinct and sorted
        # If replacement caused duplicate, we need to fix it
        # Simplest: if duplicate, regenerate that element or pick from remaining
        if len(set(ind_list)) < N:
             # Fix duplicates by picking random unused values
             used = set(ind_list)
             for i in range(N):
                 if ind_list.count(ind_list[i]) > 1:
                     # Find a unique value
                     while True:
                         new_val = random.randint(1, MAX_VAL)
                         if new_val not in used:
                             ind_list[i] = new_val
                             used.add(new_val)
                             break
        return tuple(sorted(ind_list))

    def crossover(p1, p2):
        # Uniform crossover
        child = []
        for i in range(N):
            if random.random() < 0.5:
                child.append(p1[i])
            else:
                child.append(p2[i])
        # Handle duplicates
        if len(set(child)) < N:
            used = set(child)
            for i in range(N):
                if child.count(child[i]) > 1:
                    while True:
                        new_val = random.randint(1, MAX_VAL)
                        if new_val not in used:
                            child[i] = new_val
                            used.add(new_val)
                            break
        return tuple(sorted(child))

    # Initial Population
    pop_size = 100
    population = []
    
    # Seed with a grid construction
    # 4 rows, 8 cols
    # S = { i * M + j + 1 }
    # Try a few M values
    for M in [20, 50, 100, 150]:
        s = set()
        for i in range(4):
            for j in range(8):
                val = i * M + j + 1
                if val <= MAX_VAL:
                    s.add(val)
        if len(s) == N:
            population.append(tuple(sorted(s)))
            
    # Add some randoms
    while len(population) < pop_size:
        population.append(random_individual())

    # Evaluate initial population
    scores = [fitness(ind) for ind in population]
    best_score = max(scores)
    best_ind = population[scores.index(best_score)]
    
    # Genetic Algorithm Loop
    generation = 0
    while time.time() < deadline:
        # Selection: Tournament
        def select():
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if scores[idx1] >= scores[idx2]:
                return population[idx1]
            else:
                return population[idx2]

        new_population = []
        
        # Elitism: keep the best few
        # Sort by score
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        for i in range(5): # Keep top 5
            new_population.append(population[sorted_indices[i]])
            
        while len(new_population) < pop_size:
            p1 = select()
            p2 = select()
            if random.random() < 0.7:
                child = crossover(p1, p2)
            else:
                child = p1
            child = mutate(child)
            new_population.append(child)
            
        population = new_population
        scores = [fitness(ind) for ind in population]
        
        current_best_score = max(scores)
        if current_best_score > best_score:
            best_score = current_best_score
            best_ind = population[scores.index(best_score)]
            
        generation += 1
        # Optional: print progress if needed, but keep quiet for speed

    return list(best_ind)

if __name__ == "__main__":
    print(construct())
