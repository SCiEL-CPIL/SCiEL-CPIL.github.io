"""Optimized subset-sum construction via local search with superincreasing relaxation."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 12
MAX_VAL = 1048576

def _all_subset_sums(arr):
    """Compute all 2^N subset sums using iterative DP."""
    sums = [0] * (1 << N)
    for i in range(N):
        bit = 1 << i
        ai = arr[i]
        for mask in range(bit):
            sums[bit | mask] = sums[mask] + ai
    return sums

def _is_valid(arr):
    """Check all subset sums are distinct."""
    sums = _all_subset_sums(arr)
    return len(set(sums)) == (1 << N)

def _collision_count(arr):
    """Count number of duplicate subset sum pairs (0 if valid)."""
    sums = _all_subset_sums(arr)
    seen = set()
    coll = 0
    for s in sums:
        if s in seen:
            coll += 1
        else:
            seen.add(s)
    return coll

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Start with known valid superincreasing sequence
    best = [1 << i for i in range(N)]
    best_max = max(best)
    
    random.seed(12345)
    
    # Phase 1: Try structured candidates with geometric growth ratios
    # For distinct subset sums, a sufficient condition is superincreasing,
    # but we can relax. Try ratios between 1.5 and 2.0
    ratios_to_try = [1.6, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95]
    for ratio in ratios_to_try:
        if time.time() > deadline:
            break
        # Build sequence: a[0]=1, a[i] = ceil(ratio * a[i-1]) but ensure > sum(prev)
        # Actually, for validity we need a[i] > sum(a[0..i-1]) for superincreasing,
        # but let's just try and check
        cand = [1]
        for i in range(1, N):
            # Try to keep it as small as possible while maintaining some growth
            next_val = max(cand[-1] + 1, int(ratio * cand[-1]) + 1)
            cand.append(next_val)
        
        if len(cand) == N and max(cand) <= MAX_VAL:
            if _is_valid(cand):
                if max(cand) < best_max:
                    best = cand[:]
                    best_max = max(cand)
    
    # Phase 2: Local search from best found
    # Try to reduce individual elements while maintaining validity
    current = best[:]
    current_max = max(current)
    
    # Sort for consistent processing
    current.sort()
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random element to try reducing
        idx = random.randint(1, N - 1)  # Don't touch first element (keep it at 1)
        old_val = current[idx]
        
        if old_val <= 1:
            continue
        
        # Try reducing by a random amount
        max_decrease = old_val - 1
        if random.random() < 0.7:
            # Small reduction
            decrease = random.randint(1, min(max_decrease, 10))
        else:
            # Larger reduction
            decrease = random.randint(1, max_decrease)
        
        new_val = old_val - decrease
        if new_val <= 0:
            continue
        
        current[idx] = new_val
        
        # Quick check: if the sequence is no longer sorted, skip
        # (we want to maintain sorted order for the superincreasing-like property)
        if idx > 0 and current[idx] <= current[idx - 1]:
            current[idx] = old_val
            continue
        if idx < N - 1 and current[idx] >= current[idx + 1]:
            current[idx] = old_val
            continue
        
        # Check validity
        if _is_valid(current):
            new_max = max(current)
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best = current[:]
                    best_max = current_max
        else:
            current[idx] = old_val
    
    # Phase 3: Try a different construction - use the minimal superincreasing
    # but with a[0] = 1 and each subsequent a[i] = sum(a[0..i-1]) + 1
    # This is exactly powers of 2, but let's verify and also try a[0] = 1,
    # a[i] = sum(a[0..i-1]) + delta for small delta
    # Actually this is just powers of 2. Let's try a different base.
    
    # Try: a[0]=1, a[i] = 2*a[i-1] - 1 (gives 1, 1, 1... no)
    # Try: a[i] = a[i-1] + a[i-2] + 1 (Fibonacci-like)
    fib_cand = [1, 2]
    for i in range(2, N):
        fib_cand.append(fib_cand[-1] + fib_cand[-2] + 1)
    if len(fib_cand) == N and max(fib_cand) <= MAX_VAL and _is_valid(fib_cand):
        if max(fib_cand) < best_max:
            best = fib_cand[:]
            best_max = max(fib_cand)
    
    # Phase 4: Final polish - try to reduce the largest element specifically
    current = best[:]
    current.sort()
    
    # Try reducing the last element
    for reduce_attempt in range(50000):
        if time.time() > deadline:
            break
        
        idx = N - 1  # Focus on the largest element
        old_val = current[idx]
        
        if old_val <= 1:
            break
        
        decrease = random.randint(1, min(old_val - 1, 100))
        new_val = old_val - decrease
        
        if new_val <= current[idx - 1] if idx > 0 else new_val <= 0:
            continue
        
        current[idx] = new_val
        
        if _is_valid(current):
            new_max = max(current)
            if new_max < max(best):
                best = current[:]
            current[idx] = new_val
        else:
            current[idx] = old_val
    
    # Ensure we return a valid result
    if not _is_valid(best):
        # Fallback to powers of 2
        best = [1 << i for i in range(N)]
    
    return best
# EVOLVE-BLOCK-END
