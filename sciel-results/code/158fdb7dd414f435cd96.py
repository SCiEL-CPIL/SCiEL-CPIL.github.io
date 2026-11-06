"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # First, filter out numbers where 2*a is a square (diagonal constraint)
    # If 2*a is a square, a cannot be in the set.
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    # Map valid numbers to indices for bitset operations
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph: which valid numbers conflict with which
    # a and b conflict if a + b is a perfect square (a != b)
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])
    
    # Build adjacency bitsets for fast operations
    adj_bits = []
    for i in range(num_valid):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        adj_bits.append(bits)
    
    # Function to check if a bitset represents a valid independent set
    def is_valid(bits):
        # For each set bit, check if it has any neighbors in the set
        # This is O(N^2) in worst case, but we can optimize
        temp = bits
        while temp:
            v = (temp & -temp).bit_length() - 1
            # Check if v has any neighbors in bits
            if adj_bits[v] & bits:
                return False
            temp &= ~(1 << v)
        return True

    # Function to calculate size of independent set (number of bits)
    def set_size(bits):
        return bin(bits).count('1')
    
    # Repair function: remove conflicting vertices to make a valid independent set
    # We prioritize keeping vertices with lower degree or higher "value"
    def repair(bits):
        if is_valid(bits):
            return bits
        
        # Convert to list of indices
        verts = []
        temp = bits
        while temp:
            v = (temp & -temp).bit_length() - 1
            verts.append(v)
            temp &= ~(1 << v)
        
        # Greedy removal: 
        # While there is a conflict, remove one of the conflicting vertices.
        # Heuristic: remove the one with higher degree in the current subgraph
        current_bits = bits
        changed = True
        while changed:
            changed = False
            # Find a conflict
            temp = current_bits
            conflict_found = False
            while temp:
                v = (temp & -temp).bit_length() - 1
                neighbors_in_set = adj_bits[v] & current_bits
                if neighbors_in_set:
                    # There is a conflict involving v
                    # We need to remove either v or one of its neighbors
                    # Let's pick the one with higher degree in current_bits
                    deg_v = bin(adj_bits[v] & current_bits).count('1')
                    
                    # Find the neighbor with max degree
                    max_deg = -1
                    max_neighbor = -1
                    temp_n = neighbors_in_set
                    while temp_n:
                        u = (temp_n & -temp_n).bit_length() - 1
                        deg_u = bin(adj_bits[u] & current_bits).count('1')
                        if deg_u > max_deg:
                            max_deg = deg_u
                            max_neighbor = u
                        temp_n &= ~(1 << u)
                    
                    # Remove the one with higher degree
                    if deg_v >= max_deg:
                        to_remove = v
                    else:
                        to_remove = max_neighbor
                    
                    current_bits &= ~(1 << to_remove)
                    changed = True
                    break # Restart search for conflicts
                temp &= ~(1 << v)
            
        return current_bits

    # Initial population: random valid independent sets
    def generate_random_individual():
        bits = 0
        # Randomly decide to include each vertex, then repair
        for i in range(num_valid):
            if random.random() < 0.5:
                bits |= (1 << i)
        return repair(bits)
    
    # Greedy individual for initial population
    def generate_greedy_individual():
        remaining = (1 << num_valid) - 1
        selected = 0
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            best_v = -1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                deg = bin(adj_bits[v] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp &= ~(1 << v)
            
            if best_v != -1:
                selected |= (1 << best_v)
                remaining &= ~(1 << best_v)
                remaining &= ~adj_bits[best_v]
        return selected

    # Crossover: Uniform crossover on bitsets, followed by repair
    def crossover(parent1, parent2):
        child = 0
        for i in range(num_valid):
            if random.random() < 0.5:
                if parent1 & (1 << i):
                    child |= (1 << i)
            else:
                if parent2 & (1 << i):
                    child |= (1 << i)
        return repair(child)
    
    # Mutation: Flip a bit, followed by repair
    def mutate(individual, rate=0.1):
        bits = individual
        if random.random() < rate:
            # Flip a random bit
            i = random.randint(0, num_valid - 1)
            bits ^= (1 << i)
        return repair(bits)
    
    # Fitness: Size of the independent set
    def fitness(bits):
        return set_size(bits)
    
    # Genetic Algorithm
    pop_size = 100
    population = []
    
    # Initialize population
    # Include some greedy solutions
    for _ in range(10):
        population.append(generate_greedy_individual())
    for _ in range(pop_size - 10):
        population.append(generate_random_individual())
    
    best_individual = max(population, key=fitness)
    best_size = fitness(best_individual)
    
    generation = 0
    while time.time() < deadline:
        # Selection: Tournament selection
        next_population = []
        
        # Elitism: Keep the best few
        population.sort(key=fitness, reverse=True)
        next_population.extend(population[:5])
        
        while len(next_population) < pop_size:
            # Select two parents
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            child = mutate(child)
            
            next_population.append(child)
        
        population = next_population
        
        # Update best
        current_best = max(population, key=fitness)
        if fitness(current_best) > best_size:
            best_size = fitness(current_best)
            best_individual = current_best
            
        generation += 1
        
        # Check time limit
        if time.time() > deadline - 5:
            break

    # Convert best_individual bitset to list of numbers
    result = []
    temp = best_individual
    while temp:
        v = (temp & -temp).bit_length() - 1
        result.append(valid_numbers[v])
        temp &= ~(1 << v)
    
    if not result:
        # Fallback to a simple greedy if GA failed completely
        result = [1] # 1 is always valid? 1+1=2 not square. 1+3=4 square. So 1 is valid alone.
        
    return sorted(result)
# EVOLVE-BLOCK-END
