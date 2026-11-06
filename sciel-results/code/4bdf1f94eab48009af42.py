import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin
    
    N = 17
    MIN_DIST = 6
    
    # Helper to convert int to numpy bit array
    def int_to_bits(w):
        return np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
    
    # Helper to check if a candidate bit array is valid against a set of bit arrays
    # bits: (M, N)
    # cand: (N,)
    def is_valid_vec(cand, bits):
        if bits.shape[0] == 0:
            return True
        # Compute hamming distances
        dists = np.sum(bits ^ cand, axis=1)
        return np.min(dists) >= MIN_DIST

    # Greedy Initialization
    def greedy_init(seed_val=None):
        if seed_val is not None:
            random.seed(seed_val)
        
        words = [0]
        bits = np.array([int_to_bits(0)], dtype=np.uint8)
        
        attempts = 0
        max_attempts = 50000  # Enough to fill up the code
        while len(words) < 340 and attempts < max_attempts:
            # Generate candidate
            # Bias towards words with weight ~ N/2? 
            # Random is fine, but we can try to be smarter.
            cand = random.getrandbits(N)
            cand_bits = int_to_bits(cand)
            
            if is_valid_vec(cand_bits, bits):
                words.append(cand)
                bits = np.vstack([bits, cand_bits])
                attempts = 0  # Reset counter on success
            else:
                attempts += 1
        
        return words, bits

    # Find best initial code via multiple restarts
    best_words = []
    best_bits = None
    best_size = 0
    
    for s in range(10): # 10 restarts
        if time.time() > deadline - 10: break
        w, b = greedy_init(seed_val=42 + s)
        if len(w) > best_size:
            best_size = len(w)
            best_words = w
            best_bits = b
            if best_size >= 250: # Good enough start
                break

    current_words = best_words[:]
    current_bits = best_bits.copy()
    
    # Simulated Annealing / Kicked Local Search
    # We maintain a valid set of words.
    # Moves:
    # 1. Try to add a word.
    # 2. If stuck, kick a word (change it to another valid position relative to the rest).
    
    temperature = 1.0
    cooling = 0.9999
    stuck_counter = 0
    max_stuck = 1000
    kick_prob = 0.5
    
    while time.time() < deadline:
        
        # Phase 1: Try to add
        add_success = False
        # Try a few candidates
        for _ in range(50):
            cand = random.getrandbits(N)
            cand_bits = int_to_bits(cand)
            if is_valid_vec(cand_bits, current_bits):
                current_words.append(cand)
                current_bits = np.vstack([current_bits, cand_bits])
                add_success = True
                stuck_counter = 0
                break
        
        if add_success:
            continue
            
        # Phase 2: Stuck. Kick a word.
        stuck_counter += 1
        
        # If we are stuck for a long time, increase kick aggressiveness or change strategy?
        # For now, just do random kicks.
        
        # Pick a random word to kick
        idx = random.randint(0, len(current_words) - 1)
        old_word = current_words[idx]
        
        # Generate candidate for kick
        # Option 1: Random
        # Option 2: Perturbation of old_word
        if random.random() < 0.7:
            # Perturb
            flips = random.randint(1, 3)
            mask = 0
            for _ in range(flips):
                pos = random.randint(0, N - 1)
                mask |= (1 << pos)
            cand = old_word ^ mask
        else:
            cand = random.getrandbits(N)
            
        cand_bits = int_to_bits(cand)
        
        # Check validity against all other words
        # We need to check cand against current_bits without the row idx
        # Efficient check:
        other_bits = np.concatenate([current_bits[:idx], current_bits[idx+1:]]) if len(current_bits) > 1 else np.empty((0, N), dtype=np.uint8)
        
        if len(other_bits) == 0:
            valid_kick = True
        else:
            dists = np.sum(other_bits ^ cand_bits, axis=1)
            valid_kick = np.min(dists) >= MIN_DIST
            
        if valid_kick:
            # Accept the kick
            current_words[idx] = cand
            current_bits[idx] = cand_bits
            stuck_counter = 0 # Reset stuck counter as we changed state
        else:
            # Kick failed, do nothing
            pass
            
        # Cooling (optional, might not be needed if kicks are always valid)
        # temperature *= cooling
        # if temperature < 0.01: temperature = 0.01
        
    return current_words

# EVOLVE-BLOCK-END
