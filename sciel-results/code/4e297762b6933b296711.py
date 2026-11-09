import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0  # Safety margin before 90s kill
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount for all 18-bit values
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute compatibility matrix using numpy
    # compat[i, j] = True if hamming distance between i and j >= MIN_DIST
    # We only need this for a reduced candidate pool to save memory/time
    # Use a subset of candidates: all words with weight >= 3 (since d>=6 means
    # no two words can have weight < 3 if they're to be mutually compatible with 0)
    # Actually, let's use all words but compute compatibility on-the-fly with numpy

    # Build candidate list: all words from 1 to 2^N-1 with weight >= 3
    # (words with weight < 3 cannot be at distance >= 6 from 0)
    # But we also need words at distance >= 6 from each other
    # Let's work with a manageable pool

    # Strategy: Use simulated annealing on a reduced candidate set
    # First, find a good initial solution with greedy
    # Then use SA to improve

    def greedy_construct(seed=None):
        """Greedy construction with random ordering"""
        if seed is not None:
            random.seed(seed)
        rng = random.Random(seed)
        
        # All valid candidates (weight >= 3 to be compatible with 0)
        candidates = [c for c in range(1, 1 << N) if popcount[c] >= MIN_DIST]
        rng.shuffle(candidates)
        
        words = [0]
        word_set = {0}
        
        for c in candidates:
            if time.time() > deadline - 0.5:
                break
            # Check compatibility with all existing words
            compatible = True
            for w in words:
                if popcount[c ^ w] < MIN_DIST:
                    compatible = False
                    break
            if compatible:
                words.append(c)
                word_set.add(c)
        
        return words

    # Get an initial solution
    best_words = greedy_construct(seed=42)
    best_len = len(best_words)

    # Simulated annealing
    def simulated_annealing(initial_words, max_iterations=None):
        """
        SA that tries to maximize code size by:
        1. Swap: replace one word with another to free up space
        2. Add: try to add a new word
        3. Remove: occasionally remove a word to explore different regions
        """
        if max_iterations is None:
            max_iterations = 500000
        
        current_words = list(initial_words)
        current_set = set(current_words)
        current_size = len(current_words)
        
        best_words_sa = list(current_words)
        best_size = current_size
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iterations))
        
        # Precompute for fast compatibility checking
        # We'll use a dictionary to track which words are in the set
        # and check compatibility by iterating over the set
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 1.0:
                break
            
            T = T_start * (cooling_rate ** iteration)
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.6:
                # Move 1: Try to add a random word
                # Pick a random candidate not in current set
                candidates_pool = [c for c in range(1, 1 << N) 
                                   if popcount[c] >= 3 and c not in current_set]
                if not candidates_pool:
                    continue
                c = random.choice(candidates_pool)
                
                # Check if c is compatible
                compatible = True
                for w in current_words:
                    if popcount[c ^ w] < MIN_DIST:
                        compatible = False
                        break
                
                if compatible:
                    # Accept (always better)
                    current_words.append(c)
                    current_set.add(c)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_words_sa = list(current_words)
                else:
                    # Reject (no change)
                    pass
            
            elif move_type < 0.9:
                # Move 2: Swap - remove a random word and add a different one
                if len(current_words) <= 1:
                    continue
                idx = random.randint(0, len(current_words) - 1)
                removed_word = current_words[idx]
                
                # Find a word to add that's compatible with all except removed_word
                candidates_pool = [c for c in range(1, 1 << N)
                                   if popcount[c] >= 3 and c not in current_set]
                if not candidates_pool:
                    continue
                
                # Try a few random candidates
                added = False
                for _ in range(10):
                    c = random.choice(candidates_pool)
                    compatible = True
                    for w in current_words:
                        if w == removed_word:
                            continue
                        if popcount[c ^ w] < MIN_DIST:
                            compatible = False
                            break
                    if compatible:
                        # Perform swap
                        current_words[idx] = c
                        current_set.discard(removed_word)
                        current_set.add(c)
                        added = True
                        break
                
                if added:
                    # Size unchanged, but we may have found a better configuration
                    # for future additions
                    pass
            
            else:
                # Move 3: Remove a random word (to escape local optima)
                if len(current_words) > 1:
                    idx = random.randint(0, len(current_words) - 1)
                    removed_word = current_words.pop(idx)
                    current_set.discard(removed_word)
                    current_size -= 1
            
            # Update best
            if current_size > best_size:
                best_size = current_size
                best_words_sa = list(current_words)
        
        return best_words_sa

    # Run multiple SA rounds with different initial solutions
    num_sa_rounds = 10
    for sa_round in range(num_sa_rounds):
        if time.time() > deadline - 2.0:
            break
        
        # Get a fresh initial solution
        initial = greedy_construct(seed=100 + sa_round)
        if len(initial) <= best_len:
            continue
        
        # Run SA
        result = simulated_annealing(initial, max_iterations=100000)
        if len(result) > best_len:
            best_len = len(result)
            best_words = result
    
    # Final greedy improvement pass
    if time.time() < deadline - 1.0:
        improved = list(best_words)
        current_set = set(improved)
        
        # Try all words systematically
        for c in range(1, 1 << N):
            if time.time() > deadline - 0.5:
                break
            if c in current_set:
                continue
            if popcount[c] < MIN_DIST:
                continue
            compatible = True
            for w in improved:
                if popcount[c ^ w] < MIN_DIST:
                    compatible = False
                    break
            if compatible:
                improved.append(c)
                current_set.add(c)
        
        if len(improved) > best_len:
            best_words = improved
            best_len = len(best_words)

    # Ensure we return a non-empty list
    if not best_words:
        return [0]

    return best_words
