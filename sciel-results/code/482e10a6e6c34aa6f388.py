import time
import itertools
import random

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    if n == 0:
        return 0
    
    def is_sidon(subset):
        """Check if subset has all pair sums (with replacement) unique."""
        m = len(subset)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Try subsets in decreasing size order
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [arr[i] for i in combo]
            if is_sidon(subset):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Baseline
    baseline = list(range(12))
    baseline_score = _max_sidon_subset_size(baseline)
    
    best_candidate = baseline
    best_score = baseline_score
    
    if best_score <= 3:
        return best_candidate

    # Targeted search for sets with small span to maximize collisions
    # Minimum span for 12 distinct ints is 11.
    # Try spans 11 to 25.
    
    random.seed(42)
    
    # First, try all APs with small diff (already covered by span 11)
    # Then random sets in small windows
    
    for span in range(11, 31):
        if time.time() > deadline:
            break
            
        # Number of possible windows
        max_start = 128 - span
        
        # Sample a few random windows
        # If span is small, the number of combinations C(span+1, 12) is manageable for small span
        # For span 11, C(12, 12) = 1. Only AP.
        # For span 12, C(13, 12) = 13.
        # For span 13, C(14, 12) = 91.
        # For span 14, C(15, 12) = 455.
        # For span 15, C(16, 12) = 1820.
        # For span 16, C(17, 12) = 6188.
        
        # We can iterate through all subsets for small spans!
        
        if span <= 15:
            # Iterate all subsets of size 12 from range(span+1)
            # But we also need to consider the offset (start).
            # However, the Sidon property is translation invariant.
            # So we just need to find a set of 12 numbers in [0, span] that has no Sidon 4-subset.
            # If we find one, we can place it anywhere in [0, 128].
            
            # Generate all 12-subsets of {0, ..., span}
            # This is C(span+1, 12)
            base_set = list(range(span + 1))
            
            for combo in itertools.combinations(base_set, 12):
                if time.time() > deadline:
                    break
                cand = list(combo)
                # Check if it's valid (distinct, in range) - yes by construction
                score = _max_sidon_subset_size(cand)
                if score < best_score:
                    best_score = score
                    best_candidate = cand
                    if best_score == 3:
                        return best_candidate
        else:
            # For larger spans, use random sampling
            num_samples = 100
            for _ in range(num_samples):
                if time.time() > deadline:
                    break
                # Pick a random window
                start = random.randint(0, max_start)
                # Pick 12 random distinct numbers in [start, start+span]
                cand = random.sample(range(start, start + span + 1), 12)
                score = _max_sidon_subset_size(cand)
                if score < best_score:
                    best_score = score
                    best_candidate = cand
                    if best_score == 3:
                        return best_candidate

    return best_candidate
