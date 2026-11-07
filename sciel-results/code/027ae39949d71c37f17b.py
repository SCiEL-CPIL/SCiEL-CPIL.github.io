import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_set = []
    best_size = 0
    
    def check_and_add(sums, v):
        """
        Try to add v to the state represented by sums.
        sums is a list of sets, sums[k] contains sums of subsets of size k.
        Returns new_sums if valid, else None.
        """
        new_sums = [s.copy() for s in sums]
        max_k = len(sums) - 1
        
        # We add v, so new cardinalities go up to max_k + 1
        # For each new cardinality k (from 1 to max_k + 1):
        # new sums are formed by taking sums from k-1 and adding v
        for k in range(1, max_k + 2):
            if k - 1 >= len(sums):
                # This shouldn't happen if k <= max_k + 1
                continue
            
            # Get the source sums
            source_sums = sums[k-1]
            new_part = set(x + v for x in source_sums)
            
            # Check for collisions with all other cardinalities
            # We need to check against the ORIGINAL sums for j != k
            # Because new_sums for j != k hasn't been updated with v yet in a way that interacts?
            # Actually, new_part_k is disjoint from new_part_j (j!=k) because source_sums are disjoint.
            # So we only check new_part_k against original sums[j] for j != k.
            
            for j in range(len(sums)):
                if j == k:
                    continue
                if new_part & sums[j]:
                    return None
            
            # If no collision, update new_sums
            if k < len(new_sums):
                new_sums[k] = new_sums[k].union(new_part)
            else:
                new_sums.append(new_part)
                
        return new_sums

    def run_greedy(order):
        sums = [{0}]
        current_set = []
        for v in order:
            new_sums = check_and_add(sums, v)
            if new_sums is not None:
                current_set.append(v)
                sums = new_sums
        return current_set

    candidates = list(range(1, 257))
    
    # 1. Greedy Increasing
    s1 = run_greedy(candidates)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # 2. Greedy Decreasing
    s2 = run_greedy(sorted(candidates, reverse=True))
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)

    # 3. Randomized Restart
    # Estimate cost of one run.
    # If it's < 0.1s, we can do ~800 restarts.
    # Let's do 100 restarts first and see time.
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 5.0: # Leave 5s for final polish
            break
        random.shuffle(candidates)
        s_rand = run_greedy(candidates)
        if len(s_rand) > best_size:
            best_set = s_rand
            best_size = len(s_rand)
            
    # 4. Local Search / Completion on best_set
    # Convert to set for fast lookup
    current_best = set(best_set)
    
    # Reconstruct sums for best_set
    sums = [{0}]
    for v in sorted(current_best):
        sums = check_and_add(sums, v)
        
    # Try to add any number from 1..256 not in current_best
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates to avoid bias
        candidates_to_try = [v for v in range(1, 257) if v not in current_best]
        random.shuffle(candidates_to_try)
        
        for v in candidates_to_try:
            new_sums = check_and_add(sums, v)
            if new_sums is not None:
                current_best.add(v)
                sums = new_sums
                improved = True
                if len(current_best) > best_size:
                    best_set = list(current_best)
                    best_size = len(best_set)
                break # Restart loop as sums changed

    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
