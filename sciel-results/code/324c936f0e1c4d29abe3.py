import random
import numpy as np
import time

def construct(q, n, radius):
    # Generate all words
    all_words = list(map(list, np.ndindex(*([q]*n))))
    total_words = len(all_words)
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (words within distance <= radius)
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(neigh)
        
    def get_initial_code():
        # Randomized Greedy
        counts = np.zeros(total_words, dtype=int)
        codewords = []
        
        while True:
            uncovered = [i for i in range(total_words) if counts[i] == 0]
            if not uncovered:
                break
            
            # Pick a random uncovered word
            target = random.choice(uncovered)
            candidates = neighbors[target]
            
            best_score = -1
            best_cands = []
            for c in candidates:
                score = 0
                for u in neighbors[c]:
                    if counts[u] == 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
                    
            chosen = random.choice(best_cands)
            codewords.append(chosen)
            for u in neighbors[chosen]:
                counts[u] += 1
                
        return codewords

    def simulate_annealing(initial_code, time_limit=105):
        current_code = initial_code.copy()
        counts = np.zeros(total_words, dtype=int)
        for c in current_code:
            for u in neighbors[c]:
                counts[u] += 1
                
        def is_valid():
            return np.all(counts > 0)
            
        current_energy = len(current_code)
        best_code = current_code.copy()
        best_energy = current_energy
        
        T = 1.0
        T_min = 0.05
        cooling_rate = 0.9995
        start_time = time.time()
        
        while T > T_min and (time.time() - start_time) < time_limit:
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove
                if len(current_code) > 0:
                    idx = random.randint(0, len(current_code) - 1)
                    c = current_code[idx]
                    
                    # Tentative remove
                    for u in neighbors[c]:
                        counts[u] -= 1
                        
                    if np.all(counts > 0):
                        # Valid
                        current_code.pop(idx)
                        new_energy = len(current_code)
                        delta_E = new_energy - current_energy
                        # Delta E is -1
                        # Always accept if valid (improvement)
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = current_code.copy()
                    else:
                        # Revert
                        for u in neighbors[c]:
                            counts[u] += 1
                            
            elif move_type < 0.6:
                # Add
                c = random.randint(0, total_words - 1)
                if c not in current_code:
                    current_code.append(c)
                    for u in neighbors[c]:
                        counts[u] += 1
                        
                    new_energy = len(current_code)
                    delta_E = new_energy - current_energy
                    # Delta E is +1
                    if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = current_code.copy()
                    else:
                        # Revert
                        current_code.pop()
                        for u in neighbors[c]:
                            counts[u] -= 1
            else:
                # Swap
                if len(current_code) > 0:
                    idx = random.randint(0, len(current_code) - 1)
                    old_c = current_code[idx]
                    new_c = random.randint(0, total_words - 1)
                    
                    if old_c != new_c and new_c not in current_code:
                        # Tentative swap
                        for u in neighbors[old_c]:
                            counts[u] -= 1
                        for u in neighbors[new_c]:
                            counts[u] += 1
                            
                        if np.all(counts > 0):
                            # Valid swap
                            current_code[idx] = new_c
                            # Energy unchanged
                            # Accept to explore
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_code = current_code.copy()
                        else:
                            # Revert
                            for u in neighbors[old_c]:
                                counts[u] += 1
                            for u in neighbors[new_c]:
                                counts[u] -= 1
                                
            T *= cooling_rate
            
        return best_code

    # Run multiple restarts if time permits?
    # For now, one long run.
    initial = get_initial_code()
    best = simulate_annealing(initial)
    
    return [all_words[i] for i in best]
