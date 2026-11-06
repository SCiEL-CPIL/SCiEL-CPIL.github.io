import time
import random
import numpy as np

def _compute_peak(A, n=128):
    """Compute the peak multiplicity for target range [n, 2n]."""
    A = set(A)
    peak = 0
    for t in range(n, 2*n + 1):
        count = 0
        for a in A:
            b = t - a
            if b in A and a <= b:
                count += 1
        if count > peak:
            peak = count
    return peak

def _is_valid(A, n=128):
    """Check if every integer in [n, 2n] can be written as sum of two elements of A."""
    A = set(A)
    for t in range(n, 2*n + 1):
        found = False
        for a in A:
            b = t - a
            if b in A and a <= b:
                found = True
                break
        if not found:
            return False
    return True

def _multiplicities(A, n=128):
    """Compute multiplicity for each target in [n, 2n]."""
    A = set(A)
    mults = np.zeros(2*n + 1, dtype=int)
    for a in A:
        for b in A:
            if a <= b:
                s = a + b
                if n <= s <= 2*n:
                    mults[s] += 1
    return mults

def construct():
    n = 128
    deadline = time.time() + 80  # 80s safety margin
    
    # Initial set: try a structured approach
    # Start with a set that covers the range
    # A good initial guess: elements that are "spread out"
    
    # Try different initial strategies
    best_A = None
    best_peak = float('inf')
    
    # Strategy 1: Random initial set with enough elements
    random.seed(42)
    
    # We need at least enough elements to cover [128, 256]
    # Minimum size: we need pairs to cover 129 targets
    # With k elements, we have at most k*(k+1)/2 pairs
    # So k*(k+1)/2 >= 129 => k >= 16
    
    # Try various sizes
    for size in [20, 24, 28, 32, 36, 40]:
        if time.time() > deadline - 5:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Generate random set of given size
            A = set(random.sample(range(0, n+1), size))
            
            # Check validity
            if not _is_valid(A, n):
                continue
            
            peak = _compute_peak(A, n)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A)
    
    # Local search from best initial solution
    if best_A is None:
        # Fallback to a valid but suboptimal solution
        best_A = set(range(64, 129))
        best_peak = _compute_peak(best_A, n)
    
    current_A = set(best_A)
    current_peak = best_peak
    
    # Local search: try adding/removing elements
    all_elements = set(range(0, n+1))
    
    while time.time() < deadline - 10:
        # Try a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add' and len(current_A) < n+1:
            # Try adding a random element
            candidates = list(all_elements - current_A)
            if not candidates:
                continue
            new_elem = random.choice(candidates)
            new_A = current_A | {new_elem}
            if _is_valid(new_A, n):
                peak = _compute_peak(new_A, n)
                if peak < current_peak:
                    current_A = new_A
                    current_peak = peak
                    best_A = set(current_A)
                    best_peak = current_peak
                elif peak == current_peak and random.random() < 0.3:
                    # Accept equal peak with some probability to escape local optima
                    current_A = new_A
        
        elif move_type == 'remove' and len(current_A) > 16:
            # Try removing a random element
            candidates = list(current_A)
            if not candidates:
                continue
            old_elem = random.choice(candidates)
            new_A = current_A - {old_elem}
            if _is_valid(new_A, n):
                peak = _compute_peak(new_A, n)
                if peak < current_peak:
                    current_A = new_A
                    current_peak = peak
                    best_A = set(current_A)
                    best_peak = current_peak
        
        elif move_type == 'swap' and len(current_A) > 0:
            # Try swapping: remove one, add another
            candidates_remove = list(current_A)
            candidates_add = list(all_elements - current_A)
            if not candidates_remove or not candidates_add:
                continue
            old_elem = random.choice(candidates_remove)
            new_elem = random.choice(candidates_add)
            new_A = (current_A - {old_elem}) | {new_elem}
            if _is_valid(new_A, n):
                peak = _compute_peak(new_A, n)
                if peak < current_peak:
                    current_A = new_A
                    current_peak = peak
                    best_A = set(current_A)
                    best_peak = current_peak
                elif peak == current_peak and random.random() < 0.2:
                    current_A = new_A
    
    return list(best_A)
