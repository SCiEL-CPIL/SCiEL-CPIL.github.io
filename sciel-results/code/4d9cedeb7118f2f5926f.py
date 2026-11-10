"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    def is_sidon(subset):
        """Check if subset has all pair sums unique (including diagonals)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(arr):
        """Find the largest Sidon subset size in arr using backtracking."""
        n = len(arr)
        best_size = 0
        current_subset = []
        arr_sorted = sorted(arr)
        
        def backtrack(idx, current):
            nonlocal best_size
            if len(current) > best_size:
                best_size = len(current)
            
            # Pruning
            if len(current) + (n - idx) <= best_size:
                return

            for i in range(idx, n):
                new_elem = arr_sorted[i]
                # Check if adding new_elem to current keeps it Sidon
                # We need to check sums of new_elem with all elements in current,
                # and the diagonal new_elem + new_elem.
                # These sums must not be in the set of sums of 'current'.
                
                # To avoid rebuilding the sum set every time, we can pass it, 
                # but for N=12, rebuilding is fine.
                # However, we can be smarter:
                # Check collisions with existing sums.
                
                # Let's just use the simple is_sidon for the new candidate.
                # It's O(k^2). k is at most 12.
                # Number of calls is bounded by number of valid prefixes.
                
                new_subset = current + [new_elem]
                if is_sidon(new_subset):
                    backtrack(i + 1, new_subset)

        backtrack(0, [])
        return best_size

    def score_candidate(arr):
        """Lower is better: the max Sidon subset size."""
        return max_sidon_size(arr)

    # Initial candidates
    candidates = [
        [0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25],
        [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20],
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        # Try a set with more collisions: pairs summing to 10
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # AP, bad
        # Try modular structure
        [0, 1, 4, 6, 9, 11, 20, 21, 24, 26, 29, 31]
    ]
    
    best_candidate = None
    best_score = 12
    
    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        s = score_candidate(cand)
        if s < best_score:
            best_score = s
            best_candidate = cand[:]
    
    if best_candidate is None:
        best_candidate = list(range(12))
        best_score = 12
        
    random.seed(42)
    # Local search with steepest descent
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        cand = best_candidate[:]
        improved = False
        
        # Try all possible swaps for a random index
        idx = random.randint(0, 11)
        old_val = cand[idx]
        
        # Generate a few random new values
        for _ in range(10):
            new_val = random.randint(0, 128)
            if new_val == old_val or new_val in cand:
                continue
            
            cand[idx] = new_val
            s = score_candidate(cand)
            
            if s < best_score:
                best_score = s
                best_candidate = cand[:]
                improved = True
                break
            else:
                cand[idx] = old_val
        
        if not improved and iteration % 100 == 0:
            # Restart occasionally to escape local minima
            # Pick a random candidate from initial list or generate new
            if random.random() < 0.1:
                cand = [random.randint(0, 128) for _ in range(12)]
                # Ensure distinct
                while len(set(cand)) < 12:
                    cand[random.randint(0, 11)] = random.randint(0, 128)
                s = score_candidate(cand)
                if s < best_score:
                    best_score = s
                    best_candidate = cand[:]
    
    return best_candidate
# EVOLVE-BLOCK-END
