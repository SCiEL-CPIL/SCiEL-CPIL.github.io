"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Helper to check if a candidate word is valid against existing code
    def is_valid(candidate, code_words):
        for w in code_words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True

    # Helper to add a word if valid
    def try_add(candidate, code_words):
        if is_valid(candidate, code_words):
            code_words.append(candidate)
            return True
        return False

    best_code = []
    best_size = 0
    
    # Strategy: Multiple restarts with Greedy + Local Search
    # We will run as many restarts as time allows.
    
    while time.time() < deadline:
        # 1. Initialize with a small random seed or simple base
        # Using a random start is often better for exploring different basins
        # But we can also try to start with a few random words to ensure validity
        current_code = []
        
        # Add a random start word
        start_word = random.getrandbits(N)
        current_code.append(start_word)
        
        # Phase 1: Greedy Random Addition
        # Try to fill the code with random words
        # We limit the number of attempts per restart to avoid getting stuck in a bad local optimum too long
        # But we also need to be efficient.
        attempts = 0
        max_attempts = 5000 # Fixed budget per restart to ensure we can do many restarts
        
        while attempts < max_attempts and len(current_code) < 300:
            candidate = random.getrandbits(N)
            if try_add(candidate, current_code):
                attempts = 0 # Reset counter on success? Or keep going?
                # Keeping a sliding window of failures is better
                # Let's just count total attempts
            else:
                attempts += 1
            
            # Time check inside tight loop
            if time.time() > deadline - 5.0:
                break
        
        # Phase 2: Local Search (Swap Optimization)
        # Try to replace a word in the code with a new random word
        # This allows "shuffling" to find a better packing
        swap_attempts = 0
        max_swap_attempts = 2000
        
        while swap_attempts < max_swap_attempts and time.time() < deadline - 5.0:
            # Pick a random index to swap
            idx = random.randint(0, len(current_code) - 1)
            old_word = current_code[idx]
            
            # Generate a new candidate
            # Perturb the old word? Or random?
            # Random is safer for escaping local minima, but perturb might be faster.
            # Let's try a mix: 50% random, 50% perturb
            if random.random() < 0.5:
                candidate = random.getrandbits(N)
            else:
                # Flip 1-3 bits
                flips = random.randint(1, 3)
                mask = 0
                for _ in range(flips):
                    pos = random.randint(0, N - 1)
                    mask |= (1 << pos)
                candidate = old_word ^ mask
            
            # Check if candidate is valid against ALL OTHERS
            # Note: We removed old_word from consideration for distance check against itself
            valid = True
            for j, w in enumerate(current_code):
                if j == idx:
                    continue
                if (candidate ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                current_code[idx] = candidate
                swap_attempts = 0 # Reset on success
            else:
                swap_attempts += 1
        
        # Phase 3: Greedy Addition again (maybe the swap opened up space)
        greedy_attempts = 0
        while greedy_attempts < 1000 and time.time() < deadline - 5.0:
            candidate = random.getrandbits(N)
            if try_add(candidate, current_code):
                greedy_attempts = 0
            else:
                greedy_attempts += 1
        
        # Update best
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
        # If we hit a very high score, we might stop early to save time?
        # No, keep trying until deadline.
        
        # Safety break if time is critically low
        if time.time() > deadline - 2.0:
            break

    return best_code
# EVOLVE-BLOCK-END
