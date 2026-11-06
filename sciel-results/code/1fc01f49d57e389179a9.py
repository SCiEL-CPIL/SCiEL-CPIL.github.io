"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

MINIMUM_DISTANCE = 6
N = 18
TOTAL_SPACE = 1 << N  # 262144

def _seed_words():
    # First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # Every affine Boolean function on F_2^4 is a word: 32 words, distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _greedy_construct(deadline):
    """
    Greedy construction with multiple random restarts to escape local optima.
    Uses NumPy for vectorized distance checks.
    """
    start_time = time.time()
    
    # Precompute popcount for all 2**18 values? 
    # Memory: 262144 * 1 byte = 256 KB. Very fast to build.
    # This allows O(1) distance lookup: popcount[x ^ y].
    # However, we can also just use np.bit_count which is hardware accelerated.
    # Let's stick to np.bit_count for simplicity and low memory overhead during vectorization.
    
    best_code = []
    
    # We will try several restarts with different random orders.
    # Since the space is 2^18, a full scan takes time.
    # Let's estimate:
    # NumPy vectorized check: (M,) array XOR scalar -> (M,) array -> bit_count -> min.
    # If M=1000, this is very fast (~microseconds).
    # Scanning 262,144 candidates.
    # 262k * 1us = 0.26s per full scan? Maybe more due to Python loop overhead for the outer loop.
    # The outer loop is Python: `for candidate in order:`.
    # 262k iterations in Python takes ~0.05-0.1s just for the loop.
    # The inner NumPy ops add up.
    # Let's assume ~1-2 seconds per full scan.
    # We have 90s. So we can do ~40-90 full scans?
    # But we don't need to scan ALL candidates if we find a good packing.
    # However, greedy is order-dependent. Randomizing the order helps.
    
    # Strategy:
    # 1. Start with seed words (Reed-Muller).
    # 2. Generate a random permutation of the remaining candidates.
    # 3. Iterate through this permutation.
    # 4. If a candidate is valid, add it.
    # 5. Stop if time is up.
    # 6. Keep the best code found across restarts.
    
    # To make it faster, we can skip candidates that are "too close" to existing ones quickly?
    # The NumPy check does this in bulk.
    
    # Let's optimize the inner loop:
    # Instead of creating a new np.array for the code every time we add a word,
    # we can keep a list and only convert to np.array when we need to check?
    # No, we need to check for EVERY candidate.
    # Converting list to np.array every iteration is O(M).
    # If we do it 262k times, and M grows to 1000, that's 262k * 1000 = 262M ops just for conversion?
    # That might be slow.
    
    # Better: Keep code as a list. For each candidate, compute distances to all existing words.
    # If M is small, Python loop is fine.
    # If M is large, NumPy is better.
    
    # Let's use a hybrid approach:
    # Keep code as a list.
    # For each candidate, if len(code) < threshold (e.g., 100), use Python loop with early exit.
    # Else, convert to np.array (cached?) and use vectorized check.
    
    # Actually, converting to np.array is cheap if we don't do it too often.
    # But we do it for every candidate.
    
    # Alternative:
    # Maintain a numpy array `code_np` of fixed size (e.g., 5000) and a counter `m`.
    # When adding, write to `code_np[m]` and increment `m`.
    # Check: `dists = np.bit_count(code_np[:m] ^ candidate)`.
    # This avoids reallocation and conversion.
    
    max_code_size = 5000  # A(18,6) is likely < 5000. A(18,6) bounds?
    # A(18,6) upper bound is around 2^18 / 2^12? No.
    # Sphere packing bound: 2^18 / (1 + 18 + 153 + 816 + 3060 + 8568) ~ 262144 / 12616 ~ 20.
    # Wait, that's for perfect codes.
    # Hamming bound for d=6:
    # Volume of ball radius 2 (since d=6, error correction t=2):
    # V = C(18,0) + C(18,1) + C(18,2) = 1 + 18 + 153 = 172.
    # Bound: 2^18 / 172 ~ 1524.
    # So M <= 1524.
    # So max_code_size = 2000 is safe.
    
    code_np = np.zeros(max_code_size, dtype=np.uint32)
    
    def _restart(seed_val):
        nonlocal best_code
        rng = random.Random(seed_val)
        
        # Initialize code with seed words
        seed_list = _seed_words()
        m = 0
        for w in seed_list:
            code_np[m] = w
            m += 1
        
        # Create list of all other candidates
        # This is memory heavy? 262k ints is fine.
        candidates = list(range(TOTAL_SPACE))
        # Remove seed words? Or just check them again?
        # Checking them again is cheap if they are rejected.
        # But we can filter them out for speed.
        seed_set = set(seed_list)
        candidates = [c for c in candidates if c not in seed_set]
        
        # Shuffle
        rng.shuffle(candidates)
        
        # Greedy scan
        for candidate in candidates:
            if time.time() > deadline:
                return
            
            # Check distance to existing code
            if m == 0:
                code_np[m] = candidate
                m += 1
                continue
                
            # Vectorized check
            dists = np.bit_count(code_np[:m] ^ candidate)
            if dists.min() >= MINIMUM_DISTANCE:
                code_np[m] = candidate
                m += 1
        
        # Convert to list
        current_code = code_np[:m].tolist()
        if len(current_code) > len(best_code):
            best_code = current_code
    
    # Perform restarts
    # How many restarts can we do?
    # Each restart scans 262k candidates.
    # Let's run until deadline.
    
    restart_num = 0
    while time.time() < deadline - 2.0:  # Leave 2s safety margin
        _restart(restart_num)
        restart_num += 1
        
        # Optional: If we found a very large code, we can stop early?
        # Or just keep trying.
        # The best code so far is stored in best_code.
        
    return best_code

def construct():
    # 90s wall-clock limit.
    # Use 85s for safety.
    deadline = time.time() + 85.0
    
    # Start with a simple greedy on sorted order to get a baseline quickly?
    # Or just go straight to random restarts.
    # Random restarts are more powerful.
    
    result = _greedy_construct(deadline)
    
    # Ensure we return a non-empty list of distinct integers
    if not result:
        # Fallback to seed
        result = _seed_words()
        
    return result
# EVOLVE-BLOCK-END
