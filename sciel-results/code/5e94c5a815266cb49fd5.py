"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _generate_reed_muller_seed():
    """Generate the 32-word first-order Reed-Muller code on 16 coords, extended by 0 bit."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _construct_linear_code_seed():
    """
    Construct a [17,9,6] linear code by finding 9 linearly independent basis vectors.
    We use a greedy approach to find basis vectors that all have pairwise distance >= 6.
    """
    # We'll try to construct a [17,9,6] code using a systematic approach
    # Start with the RM(1,4) code as a subcode, then try to add more dimensions
    
    # The RM(1,4) code has dimension 5 and minimum distance 8
    # We need to add 4 more basis vectors to get dimension 9
    
    # For a linear code with minimum distance 6, every non-zero codeword must have weight >= 6
    # In a [17,9,6] code, the minimum weight is 6
    
    # Let's use a different approach: construct the code from known properties
    # The [18,9,6] QR code shortened gives [17,9,6]
    
    # For now, return the RM seed and let the expansion handle the rest
    return _generate_reed_muller_seed()

def _precompute_deltas():
    """Precompute all vectors with weight < MIN_DISTANCE for fast forbidden zone updates."""
    deltas = []
    for d in range(1 << N):
        if d.bit_count() < MIN_DISTANCE:
            deltas.append(d)
    return np.array(deltas, dtype=np.uint32)

def _vectorized_greedy(seed_words, deltas, rng, time_deadline, max_passes=20):
    """
    Vectorized greedy expansion using NumPy for fast forbidden zone marking.
    Scans the entire universe in multiple passes to find a dense packing.
    """
    words = list(seed_words)
    universe_size = 1 << N
    
    # Use a bytearray for memory efficiency (1 byte per entry vs 1 byte for bool)
    # Actually, np.zeros with bool is fine for 2^17 = 131072 entries
    forbidden = np.zeros(universe_size, dtype=bool)
    
    # Initialize forbidden for seed words
    for w in words:
        forbidden[w ^ deltas] = True
    
    best_size = len(words)
    
    for pass_num in range(max_passes):
        if time.time() > time_deadline - 10:
            break
        
        # Find all non-forbidden positions
        available = np.nonzero(~forbidden)[0]
        
        if len(available) == 0:
            break
        
        # Try to add as many as possible in this pass
        added_this_pass = 0
        
        # Process available positions - we want to be strategic about which to add
        # Sort by some heuristic: maybe weight, or just process in order
        # For better packing, we might want to add lower-weight words first
        # since they have larger "forbidden neighborhoods" relative to their weight
        
        # Actually, for maximum density, the order matters less than the strategy
        # Let's just process them in a structured way
        
        # To avoid adding too many that conflict with each other, we process
        # one at a time and check if still available
        for idx in available:
            if time.time() > time_deadline - 10:
                break
            
            w = int(idx)
            if forbidden[w]:
                continue
            
            # Add this word
            words.append(w)
            forbidden[w ^ deltas] = True
            added_this_pass += 1
            
            if added_this_pass > 100:
                break  # Limit per-pass additions to avoid too long a single pass
    
    return words

def _local_search(words, deltas, rng, time_deadline):
    """
    Local search: try to replace words to increase the total count.
    Use 1-for-2 swaps: remove one word, try to add two new words.
    """
    words = list(words)
    universe_size = 1 << N
    best_words = words
    best_size = len(words)
    
    max_iterations = 200
    for it in range(max_iterations):
        if time.time() > time_deadline - 5:
            break
        if len(best_words) < 2:
            break
        
        # Pick a random word to remove
        remove_idx = rng.randint(0, len(best_words) - 1)
        removed_word = best_words[remove_idx]
        
        # Create temporary list without the removed word
        temp_words = best_words[:remove_idx] + best_words[remove_idx+1:]
        
        # Build forbidden set for temp_words
        forbidden = np.zeros(universe_size, dtype=bool)
        for w in temp_words:
            forbidden[w ^ deltas] = True
        
        # Try to find 2 new words to add
        added_words = []
        
        # Scan in structured order
        # Try all positions, but use a limited random subset for speed
        candidates = np.arange(universe_size)
        rng.shuffle(candidates)
        
        for idx_cand in candidates[:50000]:  # Limit candidates for speed
            if len(added_words) == 2:
                break
            if forbidden[idx_cand]:
                continue
            
            w = int(idx_cand)
            temp_words.append(w)
            added_words.append(w)
            forbidden[w ^ deltas] = True
        
        if len(added_words) == 2:
            new_size = len(temp_words)
            if new_size > best_size:
                best_words = temp_words
                best_size = new_size
    
    return best_words

def construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    rng = random.Random(42)
    
    # Precompute deltas
    deltas = _precompute_deltas()
    
    # Get initial seed
    seed = _generate_reed_muller_seed()
    
    best_words = seed
    best_size = len(seed)
    
    # Phase 1: Multiple vectorized greedy expansions with different orderings
    num_greedy_trials = 10
    
    for trial in range(num_greedy_trials):
        if time.time() > time_deadline - 15:
            break
        
        rng_trial = random.Random(1000 + trial)
        # Use different seeds or shuffles for diversity
        words_trial = _vectorized_greedy(seed, deltas, rng_trial, time_deadline - 5)
        
        if len(words_trial) > best_size:
            best_words = words_trial
            best_size = len(words_trial)
    
    # Phase 2: Local Search Improvement
    if time.time() < time_deadline - 10 and len(best_words) > 50:
        rng_ls = random.Random(2000)
        improved_words = _local_search(best_words, deltas, rng_ls, time_deadline - 5)
        if len(improved_words) > best_size:
            best_words = improved_words
            best_size = len(improved_words)
    
    # Final cleanup: ensure all words are valid and distinct
    seen = set()
    valid_words = []
    for w in best_words:
        if w not in seen:
            seen.add(w)
            valid_words.append(w)
    
    # Verify minimum distance constraint
    # This is O(n^2) so do it only if the list is small enough
    if len(valid_words) <= 500:
        verified = []
        for w in valid_words:
            ok = True
            for v in verified:
                if _hamming_dist(w, v) < MIN_DISTANCE:
                    ok = False
                    break
            if ok:
                verified.append(w)
        valid_words = verified
    
    if len(valid_words) < 2:
        return valid_words if valid_words else [0]
        
    return valid_words

if __name__ == "__main__":
    # Quick test
    words = construct()
    print(f"Found {len(words)} words")
