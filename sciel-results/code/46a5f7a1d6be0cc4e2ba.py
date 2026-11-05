import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    # Handle the specific case efficiently, fallback to greedy for others
    if q != 3 or n != 7 or radius != 1:
        # Simple greedy fallback
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = [set() for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].add(word_to_idx[new_word])
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = len(neighbors[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            uncovered -= neighbors[best_c]
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)  # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors as bitsets (integers) for fast operations
    # Each codeword covers itself + 14 neighbors = 15 words
    neighbors = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                new_word = tuple(new_word)
                j = word_to_idx[new_word]
                mask |= (1 << j)
        neighbors[i] = mask

    # Helper: Check if a set of codewords (list of indices) covers all
    def is_valid(code):
        if not code:
            return N == 0
        covered = 0
        for c in code:
            covered |= neighbors[c]
        # Check if covered is all 1s (2^N - 1)
        # But N=2187, so 2^2187 is huge. We can't use integer bitsets for N>63 easily in Python without big ints, 
        # which are slow. Let's use numpy bitsets or just boolean arrays.
        # Actually, Python ints are arbitrary precision. 1 << 2187 is fine.
        # But checking if covered == (1<<N)-1 is O(N/word_size) which is fast.
        return covered == (1 << N) - 1

    # Helper: Get uncovered mask
    def get_uncovered_mask(code):
        covered = 0
        for c in code:
            covered |= neighbors[c]
        full_mask = (1 << N) - 1
        return full_mask ^ covered

    # Helper: Greedy repair from uncovered mask
    def greedy_repair(uncovered_mask):
        solution = []
        remaining = uncovered_mask
        while remaining:
            best_c = -1
            best_gain = -1
            # Scan all N candidates
            # To speed up, we can pre-sort or use heuristics, but N=2187 is small enough
            for c in range(N):
                # Gain is popcount(neighbors[c] & remaining)
                gain = (neighbors[c] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            remaining &= ~neighbors[best_c]
        return solution

    # Helper: Prune redundant codewords
    def prune(code):
        if not code:
            return []
        code_list = list(code)
        # Use a boolean array for cover counts? Or just bitset union?
        # For pruning, we can iterate and check if removing a codeword breaks validity.
        # is_valid is O(|code| * N/word_size) which is fast.
        # But we can be smarter: 
        # Compute cover count for each word?
        # Let's stick to the bitset approach for simplicity and speed.
        # If we remove c, new_covered = covered ^ (neighbors[c] & ~(covered ^ neighbors[c]))? No.
        # It's easier to just recompute or use a cover count array.
        
        # Let's use cover count array for O(N) prune per step, or O(|code|*N) total.
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            # Extract bits from mask? That's slow.
            # Let's just use the boolean array approach for cover count, it's O(|code|*15)
            # Wait, neighbors are 15 words. 
            # So cover_count update is O(|code|*15) which is very fast.
            pass
        
        # Actually, let's just use the bitset validity check. It's robust.
        # Prune by removing redundant codewords.
        # A codeword c is redundant if all its neighbors are covered by others.
        # i.e., (covered_without_c) includes neighbors[c].
        # covered_without_c = covered ^ (neighbors[c] & ~covered)? No.
        # Let's just do: 
        # current_covered = 0
        # for c in code_list: current_covered |= neighbors[c]
        # Then for each i, check if (current_covered ^ neighbors[code_list[i]]) still covers everything?
        # No, that's not right because neighbors[code_list[i]] might overlap.
        
        # Simple iterative removal:
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                temp_code = code_list[:i] + code_list[i+1:]
                if is_valid(temp_code):
                    code_list.pop(i)
                    changed = True
                    break
        
        return code_list

    # Initial Solution: Greedy
    initial_uncovered = (1 << N) - 1
    current_code = greedy_repair(initial_uncovered)
    current_code = prune(current_code)
    best_code = list(current_code)
    best_size = len(best_code)

    start_time = time.time()
    timeout = 115.0

    random.seed(42)

    # Genetic Algorithm Parameters
    pop_size = 50
    population = [list(current_code) for _ in range(pop_size)]
    
    # Diversify population
    for i in range(1, pop_size):
        code = list(population[0])
        k = random.randint(1, 5)
        for _ in range(k):
            if code:
                idx = random.randint(0, len(code)-1)
                code.pop(idx)
                # Repair
                uncovered = get_uncovered_mask(code)
                if uncovered:
                    repair = greedy_repair(uncovered)
                    code.extend(repair)
                code = prune(code)
        population[i] = code

    # Main Loop
    while time.time() - start_time < timeout:
        new_population = []
        
        for _ in range(pop_size):
            # Selection: Tournament of 3
            participants = random.sample(population, 3)
            parent = min(participants, key=lambda x: len(x))
            
            # Crossover: Uniform Set Crossover
            # Let's pick a random second parent
            parent2_idx = random.randint(0, len(population)-1)
            parent2 = population[parent2_idx]
            
            # Child is a random mix of parent and parent2
            # Since they are sets (lists with unique elements ideally), we can merge and sample?
            # Or just take parent and mutate.
            # Let's do: Child = set(parent) union set(parent2), then random subset?
            # That might lose coverage.
            # Let's stick to mutation-based evolution on the selected parent, 
            # plus occasionally crossing over.
            
            if random.random() < 0.5:
                # Crossover
                set1 = set(parent)
                set2 = set(parent2)
                # Randomly assign each element to come from parent or parent2?
                # Since we want to explore, let's take union and then randomly drop some?
                # Dangerous. Let's just take parent and add some from parent2.
                child = list(parent)
                # Add a few random elements from parent2 that are not in child
                for c in random.sample(parent2, min(3, len(parent2))):
                    if c not in child:
                        child.append(c)
            else:
                child = list(parent)

            # Mutation
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                mutation_type = random.choice(['remove', 'add', 'swap'])
                
                if mutation_type == 'remove' and child:
                    idx = random.randint(0, len(child)-1)
                    child.pop(idx)
                    # Repair if invalid
                    uncovered = get_uncovered_mask(child)
                    if uncovered:
                        repair = greedy_repair(uncovered)
                        child.extend(repair)
                        
                elif mutation_type == 'add':
                    c = random.randint(0, N-1)
                    if c not in child:
                        child.append(c)
                        
                elif mutation_type == 'swap' and child:
                    idx = random.randint(0, len(child)-1)
                    new_c = random.randint(0, N-1)
                    if new_c not in child:
                        child[idx] = new_c
                        # Repair if invalid
                        uncovered = get_uncovered_mask(child)
                        if uncovered:
                            repair = greedy_repair(uncovered)
                            child.extend(repair)

            # Local Search (Prune)
            child = prune(child)
            
            # Ensure validity
            uncovered = get_uncovered_mask(child)
            if uncovered:
                repair = greedy_repair(uncovered)
                child.extend(repair)
                child = prune(child)
                
            new_population.append(child)
            
            # Update best
            if len(child) < best_size:
                best_size = len(child)
                best_code = list(child)
                
        # Elitism: Keep the best individuals
        new_population.sort(key=lambda x: len(x))
        # Keep top 50% or so?
        # Let's just replace population with new ones, but ensure best is kept.
        if len(best_code) < len(new_population[0]):
            new_population[0] = list(best_code)
            
        population = new_population

    return [list(all_words[i]) for i in best_code]
