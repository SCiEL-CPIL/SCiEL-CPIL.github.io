import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 110s, hard kill at 120s
    deadline = start_time + 110.0
    
    # Parameters specific to the task instance usually, but keep generic interface
    # Task is q=3, n=7, radius=1.
    total_words = q ** n
    
    # 1. Precompute neighbors (balls) for each word
    # For radius=1, ball size is 1 + n*(q-1)
    # We represent words as integers 0..total_words-1
    # Precompute the list of indices covered by each integer
    
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute balls
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Coverage matrix M: M[i, j] = 1 if j is in balls[i]
    # This allows fast coverage checks: M[codeword] gives the mask of covered words
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    # 2. Initial Solution: Greedy Set Cover
    # This gives a valid starting point, usually around 180-190 words.
    covered = np.zeros(total_words, dtype=bool)
    initial_codewords = []
    
    while True:
        if time.time() > deadline - 30: # Reserve 30s for GA
            break
        uncovered_mask = ~covered
        if not np.any(uncovered_mask):
            break
        scores = M.dot(uncovered_mask.astype(np.uint8))
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        initial_codewords.append(best_idx)
        covered[balls[best_idx]] = True
        
    # Ensure we have a valid solution. If greedy failed due to time, fallback to random valid?
    # Greedy on 2187 items is very fast (<0.1s usually).
    
    # If we didn't finish greedy, we might have an invalid solution.
    # Let's check validity.
    def is_valid(codewords):
        if len(codewords) == 0:
            return total_words == 0
        # Sum of rows for codewords should cover all 1s
        # M[codewords].sum(axis=0) should be all >= 1
        # Using numpy slicing: M[codewords, :].sum(axis=0)
        # This creates a view/copy. For 100x2187, it's fine.
        coverage_counts = M[codewords, :].sum(axis=0)
        return np.all(coverage_counts >= 1)

    if not is_valid(initial_codewords):
        # Fallback: Randomly add until valid (slow but safe)
        # Or just extend greedy
        while not is_valid(initial_codewords) and time.time() < deadline - 10:
             uncovered = np.where(M[initial_codewords, :].sum(axis=0) == 0)[0]
             if len(uncovered) == 0: break
             # Pick one uncovered, find a codeword that covers it
             # Actually, just pick random codeword? No, pick one that covers an uncovered.
             # Greedy step for the remaining
             uncovered_mask = ~covered
             scores = M.dot(uncovered_mask.astype(np.uint8))
             best_idx = np.argmax(scores)
             if scores[best_idx] > 0:
                 initial_codewords.append(best_idx)
                 covered[balls[best_idx]] = True
             else:
                 # Should not happen if radius>=1 and space connected
                 break

    # 3. Genetic Algorithm
    # State: list of codewords (integers)
    # Fitness: -len(codewords) (minimize size). Validity is hard constraint.
    
    best_codewords = initial_codewords[:]
    best_size = len(best_codewords)
    
    # Population size
    pop_size = 50
    # Initialize population with best solution and random perturbations
    population = [best_codewords[:]]
    for _ in range(pop_size - 1):
        # Random perturbation of best
        mut = best_codewords[:]
        # Randomly remove some
        for _ in range(np.random.randint(1, 5)):
            if mut:
                idx = np.random.randint(len(mut))
                mut.pop(idx)
        # Randomly add some
        for _ in range(np.random.randint(1, 5)):
            new_c = np.random.randint(total_words)
            if new_c not in mut:
                mut.append(new_c)
        # Ensure validity? No, let GA handle it, but we need valid individuals for comparison.
        # Strategy: Repair step.
        # If invalid, greedily add until valid.
        if not is_valid(mut):
             covered_local = np.zeros(total_words, dtype=bool)
             for c in mut:
                 covered_local[balls[c]] = True
             while not np.all(covered_local) and time.time() < deadline - 5:
                 uncovered_mask = ~covered_local
                 scores = M.dot(uncovered_mask.astype(np.uint8))
                 best_idx = np.argmax(scores)
                 if scores[best_idx] == 0: break
                 mut.append(best_idx)
                 covered_local[balls[best_idx]] = True
        population.append(mut)
        
    # GA Loop
    while time.time() < deadline - 5:
        # 1. Select parents (Tournament)
        # 2. Crossover + Mutation
        # 3. Evaluate
        # 4. Replace worst
        
        # Simple approach: Generate a few candidates per generation
        for _ in range(5): # 5 candidates per loop
            if time.time() > deadline - 5: break
            
            # Pick a parent
            p1_idx = np.random.randint(len(population))
            parent = population[p1_idx]
            
            child = parent[:]
            
            # Mutation strategies:
            # 1. Remove one
            # 2. Swap one
            # 3. Add one
            op = np.random.randint(3)
            
            if op == 0 and len(child) > 1:
                # Remove
                idx = np.random.randint(len(child))
                child.pop(idx)
            elif op == 1:
                # Swap
                if child:
                    idx = np.random.randint(len(child))
                    new_c = np.random.randint(total_words)
                    child[idx] = new_c
            else:
                # Add
                new_c = np.random.randint(total_words)
                if new_c not in child:
                    child.append(new_c)
            
            # Repair if invalid
            if not is_valid(child):
                covered_local = np.zeros(total_words, dtype=bool)
                for c in child:
                    covered_local[balls[c]] = True
                while not np.all(covered_local) and time.time() < deadline - 2:
                    uncovered_mask = ~covered_local
                    scores = M.dot(uncovered_mask.astype(np.uint8))
                    best_idx = np.argmax(scores)
                    if scores[best_idx] == 0: break
                    child.append(best_idx)
                    covered_local[balls[best_idx]] = True
            
            # If still invalid, discard
            if not is_valid(child):
                continue
                
            # Prune redundant codewords (Local Search step within GA)
            # Try to remove any redundant codeword
            changed = True
            while changed:
                changed = False
                cover_count = M[child, :].sum(axis=0)
                for i in range(len(child)):
                    c = child[i]
                    # Check if min cover count in balls[c] is > 1
                    # If yes, removing c keeps everything covered
                    if np.min(cover_count[balls[c]]) > 1:
                        child.pop(i)
                        changed = True
                        break
            
            # Evaluate
            if len(child) < best_size:
                best_size = len(child)
                best_codewords = child[:]
                # Inject best into population
                population[0] = best_codewords[:]
            else:
                # Replace a random member if it's "better" or just random replacement?
                # Let's replace the worst in population (largest size)
                worst_idx = np.argmax([len(p) for p in population])
                if len(child) < len(population[worst_idx]):
                    population[worst_idx] = child
                elif np.random.rand() < 0.1:
                    population[worst_idx] = child

    # Final Pruning on best
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        cover_count = M[best_codewords, :].sum(axis=0)
        for i in range(len(best_codewords)):
            c = best_codewords[i]
            if np.min(cover_count[balls[c]]) > 1:
                best_codewords.pop(i)
                changed = True
                break

    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
