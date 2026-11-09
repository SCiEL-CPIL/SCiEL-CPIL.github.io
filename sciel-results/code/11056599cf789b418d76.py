import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

# Precompute popcount for all 2^18 values
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(2**N)], dtype=np.uint8)

def _check_distance(code, w):
    """Check if w has distance >= MIN_DISTANCE from all words in code."""
    if not code:
        return True
    diffs = np.array(code, dtype=np.int32) ^ w
    dists = _POPCOUNT_TABLE[diffs]
    return np.min(dists) >= MIN_DISTANCE

def _greedy_numpy(order, deadline, code=None):
    """
    Greedy construction using NumPy vectorized distance checks.
    order: list of candidates to try.
    code: optional starting code.
    """
    if code is None:
        selected = []
    else:
        selected = list(code)
    
    selected_np = np.array(selected, dtype=np.int32) if selected else np.array([], dtype=np.int32)
    
    start_time = time.time()
    
    for w in order:
        if time.time() > deadline:
            break
            
        if w in selected:
            continue
            
        # Check distance to all selected codewords
        if len(selected) == 0:
            selected.append(w)
            selected_np = np.array([w], dtype=np.int32)
            continue
        
        diffs = selected_np ^ w
        dists = _POPCOUNT_TABLE[diffs]
        
        if np.min(dists) >= MIN_DISTANCE:
            selected.append(w)
            selected_np = np.append(selected_np, w)
            
    return selected

def _local_improve(code, deadline):
    """Try to add more words to the code using local search."""
    if not code:
        return code
    
    improved = list(code)
    improved_set = set(improved)
    improved_np = np.array(improved, dtype=np.int32)
    
    start_time = time.time()
    
    # Try adding words in random order
    random.seed(42)
    candidates = list(range(2**N))
    random.shuffle(candidates)
    
    added = True
    while added and time.time() < deadline:
        added = False
        for w in candidates:
            if time.time() > deadline:
                break
            if w in improved_set:
                continue
            
            diffs = improved_np ^ w
            dists = _POPCOUNT_TABLE[diffs]
            
            if np.min(dists) >= MIN_DISTANCE:
                improved.append(w)
                improved_set.add(w)
                improved_np = np.append(improved_np, w)
                added = True
    
    return improved

def _reed_muller_code():
    """Generate the Reed-Muller code RM(1,5) extended to length 18 (size 32)."""
    # RM(1,5) has 2^(1+5) = 64 words of length 32
    # We need length 18, so we'll use a different construction
    # Actually, let's just use a simple greedy to get a starting code
    return []

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_code = []
    best_len = 0
    
    # Strategy 1: Natural order greedy
    if time.time() < deadline - 10:
        order_natural = list(range(2**N))
        code = _greedy_numpy(order_natural, deadline - 10)
        if len(code) > best_len:
            best_len = len(code)
            best_code = code
    
    # Strategy 2: Reverse order greedy
    if time.time() < deadline - 10:
        order_reverse = list(range(2**N - 1, -1, -1))
        code = _greedy_numpy(order_reverse, deadline - 10)
        if len(code) > best_len:
            best_len = len(code)
            best_code = code
    
    # Strategy 3: Random orders
    for seed in [1, 2, 3, 4, 5]:
        if time.time() < deadline - 10:
            random.seed(seed)
            order_random = list(range(2**N))
            random.shuffle(order_random)
            code = _greedy_numpy(order_random, deadline - 10)
            if len(code) > best_len:
                best_len = len(code)
                best_code = code
    
    # Local improvement on the best code found
    if best_code and time.time() < deadline - 5:
        improved = _local_improve(best_code, deadline - 2)
        if len(improved) > best_len:
            best_code = improved
            best_len = len(improved)
    
    # Try more random restarts with local improvement if time permits
    for seed in range(10, 30):
        if time.time() > deadline - 15:
            break
        
        random.seed(seed)
        # Start with a small random code
        initial_code = []
        random_order = list(range(2**N))
        random.shuffle(random_order)
        
        # Quick greedy to get initial code
        for w in random_order[:1000]:
            if _check_distance(initial_code, w):
                initial_code.append(w)
            if len(initial_code) >= 10:
                break
        
        if not initial_code:
            continue
        
        # Local improve
        improved = _local_improve(initial_code, deadline - 10)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_code = improved
    
    return best_code
