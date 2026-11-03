import random
import time
import sys
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Pre-compute neighbors as numpy arrays for fast operations
    # neighbors[i] = array of indices within distance <= radius of i
    # For radius=1: 1 + n*(q-1) = 15 for q=3,n=7
    neigh_list = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                pow_q = q ** (n - 1 - i)
                new_idx = idx + (new_val - w[i]) * pow_q
                neigh.append(new_idx)
        neigh_list.append(np.array(neigh, dtype=np.int32))
    
    # Coverage matrix: cov_matrix[i, j] = 1 if j is within radius of i
    # This allows fast coverage computation
    cov_matrix = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        cov_matrix[i, neigh_list[i]] = 1
    
    # Initial solution: greedy set cover
    uncovered = np.ones(total_words, dtype=np.int8)
    code_set = set()
    
    while uncovered.sum() > 0:
        # Find word covering most uncovered
        gains = cov_matrix @ uncovered
        best_idx = np.argmax(gains)
        code_set.add(best_idx)
        uncovered[cov_matrix[best_idx] == 1] = 0
    
    code_list = list(code_set)
    
    # Compute initial coverage
    coverage = np.zeros(total_words, dtype=np.int32)
    for idx in code_list:
        coverage[cov_matrix[idx] == 1] += 1
    
    # Genetic Algorithm with simulated annealing
    best_code = code_list[:]
    best_size = len(code_list)
    
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for overhead
    
    # Population-based search
    pop_size = 50
    population = [code_list[:]]
    for _ in range(pop_size - 1):
        # Create variants via random mutations
        variant = code_list[:]
        # Remove some random codewords
        for _ in range(random.randint(0, 3)):
            if variant:
                variant.pop(random.randint(0, len(variant) - 1))
        # Add some random codewords
        for _ in range(random.randint(0, 3)):
            variant.append(random.randint(0, total_words - 1))
        # Deduplicate
        variant = list(set(variant))
        population.append(variant)
    
    # Simulated annealing parameters
    temperature = 10.0
    cooling_rate = 0.9995
    min_temp = 0.1
    
    generation = 0
    
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Evaluate all solutions in population
        valid_solutions = []
        for sol in population:
            # Check if valid and compute size
            if not sol:
                continue
            
            # Compute coverage
            cov = np.zeros(total_words, dtype=np.int32)
            for idx in sol:
                cov[cov_matrix[idx] == 1] += 1
            
            if cov.min() >= 1:
                valid_solutions.append((len(sol), sol))
        
        if not valid_solutions:
            # All invalid, restart with greedy
            code_list = best_code[:]
            population = [code_list[:]] * pop_size
            continue
        
        # Sort by size (ascending)
        valid_solutions.sort(key=lambda x: x[0])
        
        # Update best
        if valid_solutions[0][0] < best_size:
            best_size = valid_solutions[0][0]
            best_code = valid_solutions[0][1][:]
        
        # Select parents: keep top half, generate new half
        top_half = [sol for _, sol in valid_solutions[:pop_size//2]]
        
        new_population = top_half[:]
        
        while len(new_population) < pop_size:
            # Crossover: combine two parents
            if len(top_half) >= 2:
                p1 = top_half[random.randint(0, len(top_half) - 1)]
                p2 = top_half[random.randint(0, len(top_half) - 1)]
                # Uniform crossover
                child = []
                for i in range(max(len(p1), len(p2))):
                    if i < len(p1) and (i >= len(p2) or random.random() < 0.5):
                        child.append(p1[i])
                    else:
                        child.append(p2[i])
                child = list(set(child))
            else:
                child = top_half[0][:] if top_half else best_code[:]
            
            # Mutation: add/remove/replace
            if child:
                if random.random() < 0.3:
                    # Remove a random codeword
                    child.pop(random.randint(0, len(child) - 1))
                if random.random() < 0.3:
                    # Add a random codeword
                    child.append(random.randint(0, total_words - 1))
                if random.random() < 0.2 and child:
                    # Replace a random codeword
                    idx = random.randint(0, len(child) - 1)
                    child[idx] = random.randint(0, total_words - 1)
            
            child = list(set(child))
            new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
    
    # Final local search on best solution
    code_list = best_code[:]
    coverage = np.zeros(total_words, dtype=np.int32)
    for idx in code_list:
        coverage[cov_matrix[idx] == 1] += 1
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Try removals
        random.shuffle(code_list)
        for i in range(len(code_list)):
            idx = code_list[i]
            for nb in neigh_list[idx]:
                coverage[nb] -= 1
            
            if coverage.min() >= 1:
                del code_list[i]
                improved = True
                break
            else:
                for nb in neigh_list[idx]:
                    coverage[nb] += 1
        
        # Try swaps
        if not improved:
            for i in range(len(code_list)):
                c = code_list[i]
                # Try a few random swaps
                for _ in range(5):
                    w = random.randint(0, total_words - 1)
                    if w == c:
                        continue
                    
                    for nb in neigh_list[c]:
                        coverage[nb] -= 1
                    for nb in neigh_list[w]:
                        coverage[nb] += 1
                    
                    if coverage.min() >= 1:
                        code_list[i] = w
                        improved = True
                        break
                    else:
                        for nb in neigh_list[w]:
                            coverage[nb] -= 1
                        for nb in neigh_list[c]:
                            coverage[nb] += 1
                
                if improved:
                    break
    
    # Convert to result
    result = [idx_to_word(idx) for idx in code_list]
    return result
