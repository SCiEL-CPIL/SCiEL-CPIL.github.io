import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    
    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: greedy for generality
        total = q ** n
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = q ** i
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)
        
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(len(balls)):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break
        
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17
    
    # Precompute balls as lists of integers
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)
    
    # Precompute inverse: for each y, list of x such that y in balls[x]
    # Due to symmetry, inv[y] = balls[y]
    
    # Build inverse index: for each word y, which codewords cover it
    # inv[x] = list of y such that x is in balls[y] = balls[x] by symmetry
    # So we can use balls directly as the inverse
    
    # Fast greedy using precomputed inverse
    def fast_greedy(noise_scale=0.0):
        """Greedy set cover with optional noise to break ties."""
        covered = np.zeros(total, dtype=bool)
        counts = np.zeros(total, dtype=np.int32)
        # Initial counts: all balls have size 17
        counts[:] = ball_size
        if noise_scale > 0:
            noise = np.random.uniform(0, 1, size=total)
            score = counts.astype(float) + noise * noise_scale
        else:
            score = counts.astype(float)
        
        selected = []
        uncovered_count = total
        
        while uncovered_count > 0:
            if time.time() > deadline - 15:
                break
            # Find the codeword covering the most uncovered points
            max_idx = int(np.argmax(score))
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            
            # Update: for each y in balls[max_idx], if not covered, mark covered
            # and decrement counts for all x that cover y
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    uncovered_count -= 1
                    # Decrement counts for all codewords that cover y
                    # inv[y] = balls[y] by symmetry
                    for x in balls[y]:
                        counts[x] -= 1
                        score[x] -= 1.0
            
            if uncovered_count == 0:
                break
        
        return selected
    
    # Local search: remove redundant codewords
    def local_search_remove(selected_list):
        """Remove redundant codewords from the solution."""
        current_set = set(selected_list)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            for y in balls[c]:
                cover_count[y] += 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_set:
                    continue
                # Check if c is redundant: all words in balls[c] are covered by others
                redundant = True
                for y in balls[c]:
                    if cover_count[y] == 1:
                        redundant = False
                        break
                if redundant:
                    for y in balls[c]:
                        cover_count[y] -= 1
                    current_set.remove(c)
                    improved = True
        
        return list(current_set)
    
    # Simulated Annealing to escape local minima
    def simulated_annealing(initial_solution, time_budget=30.0):
        """SA with penalty-based energy to escape local minima."""
        sa_start = time.time()
        sa_deadline = sa_start + time_budget
        
        current_set = set(initial_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            for y in balls[c]:
                cover_count[y] += 1
        
        # Energy = number of codewords + penalty * number of uncovered words
        def get_energy(cset, cc):
            uncovered = np.sum(cc == 0)
            penalty = 10.0
            return len(cset) + penalty * uncovered
        
        best_set = set(current_set)
        best_energy = get_energy(current_set, cover_count)
        
        T = 5.0  # initial temperature
        cooling = 0.9995
        
        # Precompute: for each codeword, its ball as a numpy array for speed
        ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
        
        while time.time() < sa_deadline:
            # Try a random move: add or remove a codeword
            if random.random() < 0.4 and len(current_set) > 500:
                # Remove a random codeword
                c = random.choice(list(current_set))
                delta_energy = 1.0  # removing one codeword
                # Check how many words become uncovered
                newly_uncovered = 0
                for y in balls[c]:
                    if cover_count[y] == 1:
                        newly_uncovered += 1
                delta_energy += 10.0 * newly_uncovered
                
                if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(T, 1e-10)):
                    # Accept
                    for y in balls[c]:
                        cover_count[y] -= 1
                    current_set.remove(c)
                    energy = get_energy(current_set, cover_count)
                    if energy < best_energy:
                        best_energy = energy
                        best_set = set(current_set)
            elif random.random() < 0.6:
                # Add a random codeword not in current set
                # Pick a random uncovered word and add a codeword that covers it
                uncovered_words = np.where(cover_count == 0)[0]
                if len(uncovered_words) > 0:
                    target_y = int(random.choice(uncovered_words))
                    # Pick a random codeword that covers target_y
                    candidates = [x for x in balls[target_y] if x not in current_set]
                    if candidates:
                        c = random.choice(candidates)
                        # Check how many new words get covered
                        newly_covered = 0
                        for y in balls[c]:
                            if cover_count[y] == 0:
                                newly_covered += 1
                        delta_energy = 1.0 - 10.0 * newly_covered
                        
                        if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(T, 1e-10)):
                            for y in balls[c]:
                                cover_count[y] += 1
                            current_set.add(c)
                            energy = get_energy(current_set, cover_count)
                            if energy < best_energy:
                                best_energy = energy
                                best_set = set(current_set)
            else:
                # Swap: remove one, add another
                if len(current_set) > 500:
                    c_remove = random.choice(list(current_set))
                    # Find a codeword to add that covers some of the words c_remove covers
                    target_words = balls[c_remove]
                    candidates = []
                    for y in target_words:
                        for x in balls[y]:
                            if x not in current_set:
                                candidates.append(x)
                                break
                    if candidates:
                        c_add = random.choice(candidates)
                        # Compute delta energy
                        # Remove c_remove: words that lose coverage
                        lose_coverage = 0
                        for y in balls[c_remove]:
                            if cover_count[y] == 1:
                                lose_coverage += 1
                        # Add c_add: words that gain coverage
                        gain_coverage = 0
                        for y in balls[c_add]:
                            if cover_count[y] == 0:
                                gain_coverage += 1
                        delta_energy = -1.0 + 1.0 + 10.0 * lose_coverage - 10.0 * gain_coverage
                        
                        if delta_energy <= 0 or random.random() < np.exp(-delta_energy / max(T, 1e-10)):
                            # Execute swap
                            for y in balls[c_remove]:
                                cover_count[y] -= 1
                            current_set.remove(c_remove)
                            for y in balls[c_add]:
                                cover_count[y] += 1
                            current_set.add(c_add)
                            energy = get_energy(current_set, cover_count)
                            if energy < best_energy:
                                best_energy = energy
                                best_set = set(current_set)
            
            T *= cooling
        
        return list(best_set)
    
    # Main algorithm
    best_size = total
    best_solution = None
    
    # 1. Initial greedy
    greedy_selected = fast_greedy()
    if len(greedy_selected) < best_size:
        best_size = len(greedy_selected)
        best_solution = greedy_selected[:]
    
    # 2. Local search to remove redundant
    if best_solution is not None:
        refined = local_search_remove(best_solution)
        if len(refined) < best_size:
            best_size = len(refined)
            best_solution = refined
    
    # 3. Simulated Annealing
    if best_solution is not None and time.time() < deadline - 20:
        sa_time = min(40.0, deadline - time.time() - 10)
        if sa_time > 5:
            sa_result = simulated_annealing(best_solution, time_budget=sa_time)
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_solution = sa_result
            # Local search after SA
            if best_solution is not None:
                refined = local_search_remove(best_solution)
                if len(refined) < best_size:
                    best_size = len(refined)
                    best_solution = refined
    
    # 4. Try additional greedy restarts with noise if time remains
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            noisy_greedy = fast_greedy(noise_scale=0.5)
            if len(noisy_greedy) < best_size:
                best_size = len(noisy_greedy)
                best_solution = noisy_greedy[:]
            # Local search
            if best_solution is not None:
                refined = local_search_remove(best_solution)
                if len(refined) < best_size:
                    best_size = len(refined)
                    best_solution = refined
    
    # 5. Final local search pass
    if best_solution is not None and time.time() < deadline - 5:
        refined = local_search_remove(best_solution)
        if len(refined) < best_size:
            best_size = len(refined)
            best_solution = refined
    
    # Convert best_solution to list of digit lists
    if best_solution is None:
        best_solution = greedy_selected
    
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)
    
    return result
