import itertools
import random
import time
import math
import sys

def construct(q, n, radius):
    # Assume q=3, n=6, radius=1 based on task description
    # But handle generally if possible.
    
    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1_sa()
    
    # Fallback generic greedy
    return _construct_generic(q, n, radius)

def _construct_ternary_6_1_sa():
    # Precompute neighbors for q=3, n=6
    # Total words: 729
    # Neighbor list: for each word, list of indices within distance <= 1
    
    def to_int(word):
        val = 0
        for digit in word:
            val = val * 3 + digit
        return val

    def to_word(val):
        word = []
        for _ in range(6):
            word.append(val % 3)
            val //= 3
        return word[::-1]

    all_words = list(range(729))
    neighbors = [[] for _ in range(729)]
    
    for w_idx in all_words:
        w = to_word(w_idx)
        nbs = [w_idx]
        for i in range(6):
            for new_val in range(3):
                if new_val == w[i]:
                    continue
                new_word = w.copy()
                new_word[i] = new_val
                nbs.append(to_int(new_word))
        neighbors[w_idx] = nbs

    # Initial Code Generation (Size 73)
    # Reuse the logic from the previous program to get a valid 73-code quickly
    def get_initial_code():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                initial_code.append(h + list(tail))
        
        code_set = set(to_int(w) for w in initial_code)
        
        coverage = [0] * 729
        def add_codeword(c):
            for nb in neighbors[c]:
                coverage[nb] += 1
        def remove_codeword(c):
            for nb in neighbors[c]:
                coverage[nb] -= 1

        for c in code_set:
            add_codeword(c)
        
        current_code = list(code_set)
        
        # Greedy deletion to minimize size
        # We want to reach 73. The seed is 81.
        # We perform greedy deletion until no more can be removed.
        improved = True
        while improved:
            improved = False
            for c in list(current_code):
                remove_codeword(c)
                if min(coverage) > 0:
                    current_code.remove(c)
                    improved = True
                else:
                    add_codeword(c)
                    
        return current_code, coverage

    current_code, coverage = get_initial_code()
    
    # State for SA
    x = [0] * 729
    for c in current_code:
        x[c] = 1
        
    current_size = len(current_code)
    uncovered = 0
    for c in range(729):
        if coverage[c] == 0:
            uncovered += 1
            
    M = 1000  # Penalty for uncovered words
    current_energy = current_size + M * uncovered
    
    best_code = set(current_code)
    best_size = current_size
    best_uncovered = uncovered
    
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Simulated Annealing Parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    steps = 0
    while time.time() < deadline and T > T_min:
        # Pick a random word to flip
        i = random.randint(0, 728)
        
        if x[i] == 0:
            # Add codeword i
            delta_size = 1
            # Check change in uncovered
            # If we add i, it covers neighbors[i].
            # Uncovered decreases by the number of neighbors of i that were uncovered.
            # But we need to know which were uncovered.
            # We don't have a set of uncovered indices readily available for fast query.
            # We can compute delta_uncovered by checking coverage of neighbors.
            
            # Optimization: Just compute new energy?
            # No, we need delta.
            # Let's compute delta_uncovered.
            # A word j is uncovered if coverage[j] == 0.
            # If we add i, coverage[j] increases for j in neighbors[i].
            # If coverage[j] was 0, it becomes 1, so uncovered count decreases by 1.
            # If coverage[j] was > 0, it stays > 0, so uncovered count doesn't change.
            
            delta_unc = 0
            for nb in neighbors[i]:
                if coverage[nb] == 0:
                    delta_unc -= 1
            
            # Apply move
            for nb in neighbors[i]:
                coverage[nb] += 1
            x[i] = 1
            
            new_size = current_size + 1
            new_unc = uncovered + delta_unc
            new_energy = new_size + M * new_unc
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_size = new_size
                uncovered = new_unc
                current_energy = new_energy
                
                if uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code = set(c for c in range(729) if x[c] == 1)
            else:
                # Reject move
                for nb in neighbors[i]:
                    coverage[nb] -= 1
                x[i] = 0
        else:
            # Remove codeword i
            # Check if any neighbor becomes uncovered
            # If coverage[nb] == 1, it becomes 0, so uncovered increases by 1.
            # If coverage[nb] > 1, it stays > 0.
            
            delta_unc = 0
            for nb in neighbors[i]:
                if coverage[nb] == 1:
                    delta_unc += 1
            
            # Apply move
            for nb in neighbors[i]:
                coverage[nb] -= 1
            x[i] = 0
            
            new_size = current_size - 1
            new_unc = uncovered + delta_unc
            new_energy = new_size + M * new_unc
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_size = new_size
                uncovered = new_unc
                current_energy = new_energy
                
                if uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code = set(c for c in range(729) if x[c] == 1)
            else:
                # Reject move
                for nb in neighbors[i]:
                    coverage[nb] += 1
                x[i] = 1
                
        T *= cooling_rate
        steps += 1
        
        # If we found 72, we can stop early or keep trying for 71?
        # 71 is lower bound. Unlikely to find 71.
        # If best_size <= 72, we can break to save time for polish?
        # Or just let it run to maximize chance of 72 if we are at 73.
        # If we are at 72, we are done (best known is 73, 72 is a breakthrough).
        if best_size <= 72:
             # We found a better code than the known best. 
             # We can break and polish.
             break

    # Final Polish: Greedy deletion on the best code found
    # Ensure we start from the best code state
    x_final = [0] * 729
    for c in best_code:
        x_final[c] = 1
        
    coverage_final = [0] * 729
    for c in best_code:
        for nb in neighbors[c]:
            coverage_final[nb] += 1
            
    current_code_final = list(best_code)
    
    # Greedy deletion
    improved = True
    while improved:
        improved = False
        for c in list(current_code_final):
            # Try removing c
            for nb in neighbors[c]:
                coverage_final[nb] -= 1
            if min(coverage_final) > 0:
                x_final[c] = 0
                current_code_final.remove(c)
                improved = True
            else:
                for nb in neighbors[c]:
                    coverage_final[nb] += 1
                
    # Convert back to list of words
    return [to_word(c) for c in current_code_final]

def _construct_generic(q, n, radius):
    # Simple greedy for fallback
    # This is not optimized for the specific task but ensures correctness for other params
    total = q ** n
    words = [list(map(int, itertools.product(range(q), repeat=n))[i]) for i in range(total)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total
    code = []
    
    remaining = set(range(total))
    
    while remaining:
        # Find word that covers most remaining
        best_w = None
        best_count = -1
        for w_idx in remaining:
            w = words[w_idx]
            count = 0
            for r_idx in remaining:
                if dist(w, words[r_idx]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        code.append(words[best_w])
        # Update remaining
        w = words[best_w]
        new_remaining = set()
        for r_idx in remaining:
            if dist(w, words[r_idx]) > radius:
                new_remaining.add(r_idx)
        remaining = new_remaining
        
    return code
