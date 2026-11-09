"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

def _reed_muller_base():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _greedy_build(candidates, min_dist):
    """Greedy: pick first candidate, then repeatedly add the candidate with min distance >= min_dist.
    Returns the list of selected words."""
    if not candidates:
        return []
    selected = [candidates[0]]
    remaining = list(candidates[1:])
    while remaining:
        best_idx = -1
        best_min_dist = -1
        for i, w in enumerate(remaining):
            # Check distance to all selected
            ok = True
            for s in selected:
                if (w ^ s).bit_count() < min_dist:
                    ok = False
                    break
            if ok:
                # We want to add it; for greedy we just take the first valid one
                # But to be smarter, we could pick the one that leaves the most room.
                # Simple greedy: take first valid.
                best_idx = i
                break
        if best_idx == -1:
            break
        selected.append(remaining.pop(best_idx))
    return selected

def _greedy_with_ordering(min_dist, deadline, seed=0):
    """
    Randomized greedy: 
    1. Start with 0.
    2. Build a list of candidates sorted by some random-ish order.
    3. Greedily add if distance >= min_dist.
    """
    random.seed(seed)
    # Start with 0
    selected = [0]
    
    # Generate all possible words? 2^18 = 262144. That's manageable.
    # But checking all pairs is expensive.
    # Let's try a different greedy: 
    # Start with 0.
    # Candidates are all other 2^18-1 words.
    # Sort candidates by weight? Or random?
    
    # Actually, a known good heuristic for packing is to start with 0,
    # then add words of low weight first? Or high weight?
    # Let's try: sort candidates by weight, then by value.
    # Or just random order with multiple restarts.
    
    # Let's try a structured greedy:
    # 1. Add 0.
    # 2. For weight w from 1 to N:
    #      For each word of weight w (in some order):
    #           If dist to all selected >= min_dist, add it.
    
    # Precompute words by weight
    # 2^18 is 262144. 
    # We can iterate through all words and group by weight.
    
    words_by_weight = [[] for _ in range(N+1)]
    for w in range(1, 2**N):
        wt = w.bit_count()
        words_by_weight[wt].append(w)
    
    # Shuffle within each weight class to break symmetry
    for wt in range(N+1):
        random.shuffle(words_by_weight[wt])
    
    selected = [0]
    start_time = time.time()
    
    for wt in range(1, N+1):
        if time.time() > deadline:
            break
        for w in words_by_weight[wt]:
            if time.time() > deadline:
                break
            ok = True
            for s in selected:
                if (w ^ s).bit_count() < min_dist:
                    ok = False
                    break
            if ok:
                selected.append(w)
    
    return selected

def _local_improve(selected, min_dist, deadline):
    """
    Try to replace a codeword with another to allow adding more?
    Or just try to add more using a more sophisticated check?
    For now, let's stick to greedy with multiple random restarts.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Reed-Muller base (32 words)
    base_words = _reed_muller_base()
    best_code = list(base_words)
    best_len = len(best_code)
    
    # Try greedy construction with different random seeds
    # Each greedy run takes some time. We need to be careful.
    # 2^18 = 262144 candidates.
    # Checking each candidate against ~500 selected words: 500 * 262144 = 130M ops.
    # In Python, this is too slow for one run.
    # We need a faster way to check distances.
    
    # Optimization: Precompute distances? No, 2^18 * 2^18 is too big.
    # Use NumPy? 
    # Let's use a bitset approach or just be very careful with Python loops.
    
    # Alternative: Use the fact that we only need to check against selected words.
    # If selected is small (< 100), it's fast.
    # If selected grows to 500, it's slow.
    
    # Let's try a different strategy: 
    # Greedy with early termination.
    # And use multiple shorter runs.
    
    # Actually, let's just run a few greedy passes.
    # Pass 1: Weight order 1..18
    # Pass 2: Weight order 18..1
    # Pass 3: Random order
    
    # To speed up, we can use a "forbidden" set.
    # If a word is too close to any selected, it's forbidden.
    # But we still have to check.
    
    # Let's try to optimize the inner loop.
    # Instead of checking all selected words, we can stop early.
    
    # Let's try one greedy run with weight ascending.
    code = _greedy_with_ordering(MIN_DISTANCE, deadline, seed=42)
    if len(code) > best_len:
        best_code = code
        best_len = len(code)
    
    # Try weight descending
    # Modify _greedy_with_ordering to accept order?
    # Or just write a second version.
    
    def _greedy_desc(min_dist, deadline, seed=0):
        random.seed(seed)
        words_by_weight = [[] for _ in range(N+1)]
        for w in range(1, 2**N):
            wt = w.bit_count()
            words_by_weight[wt].append(w)
        for wt in range(N+1):
            random.shuffle(words_by_weight[wt])
        
        selected = [0]
        for wt in range(N, 0, -1):
            if time.time() > deadline:
                break
            for w in words_by_weight[wt]:
                if time.time() > deadline:
                    break
                ok = True
                for s in selected:
                    if (w ^ s).bit_count() < min_dist:
                        ok = False
                        break
                if ok:
                    selected.append(w)
        return selected
    
    code2 = _greedy_desc(MIN_DISTANCE, deadline, seed=123)
    if len(code2) > best_len:
        best_code = code2
        best_len = len(code2)
    
    # Try random order
    def _greedy_random(min_dist, deadline, seed=0):
        random.seed(seed)
        candidates = list(range(1, 2**N))
        random.shuffle(candidates)
        selected = [0]
        for w in candidates:
            if time.time() > deadline:
                break
            ok = True
            for s in selected:
                if (w ^ s).bit_count() < min_dist:
                    ok = False
                    break
            if ok:
                selected.append(w)
        return selected
    
    code3 = _greedy_random(MIN_DISTANCE, deadline, seed=7)
    if len(code3) > best_len:
        best_code = code3
        best_len = len(code3)
        
    return best_code
# EVOLVE-BLOCK-END
