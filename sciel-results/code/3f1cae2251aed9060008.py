import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    N = q ** n  # Total number of words
    
    # Precompute all words as tuples and map to indices
    words = []
    word_to_idx = {}
    for i in range(N):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w = tuple(reversed(w))
        words.append(w)
        word_to_idx[w] = i
    
    # Precompute balls using vectorized approach
    # For each word, find all words within Hamming distance <= radius
    balls = [None] * N
    
    # Precompute digit arrays for fast distance computation
    digits_array = np.array(words, dtype=np.int32)  # shape (N, n)
    
    for i in range(N):
        # Compute Hamming distances from word i to all other words
        diff = digits_array != digits_array[i]  # shape (N, n)
        dists = np.sum(diff, axis=1)  # shape (N,)
        ball_i = np.where(dists <= radius)[0]
        balls[i] = ball_i
    
    # Convert balls to lists for easier manipulation
    balls = [list(b) for b in balls]
    
    # Precompute covered_by: for each word w, which codewords cover it
    # covered_by[w] = list of codeword indices c such that w is in balls[c]
    covered_by = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            covered_by[w].append(c)
    
    # Initial solution: Tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_codewords.append(h + list(tail))
    
    current_set = set()
    for cw in init_codewords:
        idx = word_to_idx[tuple(cw)]
        current_set.add(idx)
    
    # Initialize cover counts
    cover_count = np.zeros(N, dtype=np.int32)
    for idx in current_set:
        cover_count[balls[idx]] += 1
    
    uncovered_count = int(np.sum(cover_count == 0))
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Helper: greedy redundancy removal
    def greedy_remove(codeword_set, cc, time_limit=None):
        changed = True
        while changed:
            changed = False
            items = list(codeword_set)
            random.shuffle(items)
            for idx in items:
                if idx not in codeword_set:
                    continue
                if time_limit is not None and time.time() > time_limit:
                    return codeword_set, cc, False
                ball_arr = np.array(balls[idx])
                if np.all(cc[ball_arr] > 1):
                    cc[balls[idx]] -= 1
                    codeword_set.remove(idx)
                    changed = True
                    if len(codeword_set) < best_size:
                        best_size = len(codeword_set)
                        best_set = set(codeword_set)
        return codeword_set, cc, True
    
    # Step 1: Greedy Redundancy Removal
    current_set, cover_count, _ = greedy_remove(current_set, cover_count, deadline)
    
    if uncovered_count == 0 and len(current_set) < best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Step 2: Simulated Annealing with restarts
    T = 1.0
    cooling_rate = 0.998
    min_T = 0.01
    
    def get_energy(size, uncov):
        return uncov * 10000 + size
    
    current_energy = get_energy(len(current_set), uncovered_count)
    
    # Precompute for faster SA moves
    all_indices = np.arange(N)
    
    step_count = 0
    while time.time() < deadline:
        # Reset if too cold
        if T < min_T:
            # Perturbation: add some random words to escape local optima
            if len(current_set) < N:
                candidates = [i for i in all_indices if i not in current_set]
                if candidates:
                    # Add 2-5 random words
                    num_add = min(random.randint(2, 5), len(candidates))
                    for _ in range(num_add):
                        idx_to_add = random.choice(candidates)
                        cover_count[balls[idx_to_add]] += 1
                        current_set.add(idx_to_add)
                        candidates.remove(idx_to_add)
                    uncovered_count = int(np.sum(cover_count == 0))
                    current_energy = get_energy(len(current_set), uncovered_count)
            
            T = 1.0
            # Do a few greedy removals after perturbation
            for _ in range(3):
                items = list(current_set)
                random.shuffle(items)
                for idx in items:
                    if idx not in current_set:
                        continue
                    ball_arr = np.array(balls[idx])
                    if np.all(cover_count[ball_arr] > 1):
                        cover_count[balls[idx]] -= 1
                        current_set.remove(idx)
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break
        
        # Pick a move
        move_type = random.randint(0, 2)
        
        if move_type == 0:  # Remove
            if len(current_set) == 0:
                continue
            idx_to_remove = random.choice(list(current_set))
            
            # Calculate delta
            ball_arr = np.array(balls[idx_to_remove])
            delta_uncov = int(np.sum(cover_count[ball_arr] == 1))
            
            delta_size = -1
            delta_energy = get_energy(len(current_set) + delta_size, uncovered_count + delta_uncov) - current_energy
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                cover_count[balls[idx_to_remove]] -= 1
                current_set.remove(idx_to_remove)
                uncovered_count += delta_uncov
                current_energy += delta_energy
                
                if uncovered_count == 0 and len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    
        elif move_type == 1:  # Add - target uncovered words
            if uncovered_count > 0:
                # Find an uncovered word and add a codeword that covers it
                uncovered_words = np.where(cover_count == 0)[0]
                if len(uncovered_words) > 0:
                    target_word = random.choice(uncovered_words)
                    # Use covered_by to find candidates efficiently
                    candidates = covered_by[target_word]
                    valid_candidates = [c for c in candidates if c not in current_set]
                    if valid_candidates:
                        idx_to_add = random.choice(valid_candidates)
                    else:
                        idx_to_add = random.randint(0, N - 1)
                        while idx_to_add in current_set:
                            idx_to_add = random.randint(0, N - 1)
                else:
                    idx_to_add = random.randint(0, N - 1)
                    while idx_to_add in current_set:
                        idx_to_add = random.randint(0, N - 1)
            else:
                idx_to_add = random.randint(0, N - 1)
                while idx_to_add in current_set:
                    idx_to_add = random.randint(0, N - 1)
            
            ball_arr = np.array(balls[idx_to_add])
            delta_uncov = -int(np.sum(cover_count[ball_arr] == 0))
            
            delta_size = 1
            delta_energy = get_energy(len(current_set) + delta_size, uncovered_count + delta_uncov) - current_energy
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                cover_count[balls[idx_to_add]] += 1
                current_set.add(idx_to_add)
                uncovered_count += delta_uncov
                current_energy += delta_energy
                
                if uncovered_count == 0 and len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    
        else:  # Swap
            if len(current_set) == 0:
                continue
            idx_to_remove = random.choice(list(current_set))
            
            # Choose add target based on uncovered words
            if uncovered_count > 0:
                uncovered_words = np.where(cover_count == 0)[0]
                if len(uncovered_words) > 0:
                    target_word = random.choice(uncovered_words)
                    # Use covered_by to find candidates efficiently
                    candidates = covered_by[target_word]
                    valid_candidates = [c for c in candidates if c not in current_set and c != idx_to_remove]
                    if valid_candidates:
                        idx_to_add = random.choice(valid_candidates)
                    else:
                        idx_to_add = random.randint(0, N - 1)
                        while idx_to_add in current_set or idx_to_add == idx_to_remove:
                            idx_to_add = random.randint(0, N - 1)
                else:
                    idx_to_add = random.randint(0, N - 1)
                    while idx_to_add in current_set or idx_to_add == idx_to_remove:
                        idx_to_add = random.randint(0, N - 1)
            else:
                idx_to_add = random.randint(0, N - 1)
                while idx_to_add in current_set or idx_to_add == idx_to_remove:
                    idx_to_add = random.randint(0, N - 1)
            
            # Calculate delta for swap
            # Remove idx_to_remove, add idx_to_add
            ball_remove = np.array(balls[idx_to_remove])
            ball_add = np.array(balls[idx_to_add])
            
            # For efficiency, compute delta_uncov directly
            # Words that become uncovered: cover_count[j] == 1, j in ball_remove, j not in ball_add
            # Words that become covered: cover_count[j] == 0, j in ball_add, j not in ball_remove
            
            ball_remove_set = set(ball_remove)
            ball_add_set = set(ball_add)
            
            # Words becoming uncovered
            become_uncov = 0
            for j in ball_remove:
                if cover_count[j] == 1 and j not in ball_add_set:
                    become_uncov += 1
            
            # Words becoming covered
            become_cov = 0
            for j in ball_add:
                if cover_count[j] == 0 and j not in ball_remove_set:
                    become_cov += 1
            
            delta_uncov = become_uncov - become_cov
            
            delta_size = 0  # Swap keeps size same
            delta_energy = get_energy(len(current_set) + delta_size, uncovered_count + delta_uncov) - current_energy
            
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                cover_count[balls[idx_to_remove]] -= 1
                cover_count[balls[idx_to_add]] += 1
                current_set.remove(idx_to_remove)
                current_set.add(idx_to_add)
                uncovered_count += delta_uncov
                current_energy += delta_energy
                
                if uncovered_count == 0 and len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        T *= cooling_rate
        step_count += 1
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final greedy removal pass
    if time.time() < deadline - 5:
        best_cc = np.zeros(N, dtype=np.int32)
        for cidx in best_set:
            best_cc[balls[cidx]] += 1
        best_set, best_cc, _ = greedy_remove(best_set, best_cc, deadline)
    
    # Convert best_set indices back to lists of digits
    result = []
    for idx in best_set:
        result.append(list(words[idx]))
    
    return result
