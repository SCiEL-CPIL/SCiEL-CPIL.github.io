import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    def get_counts(S):
        """Calculate the sum representation counts for a set S."""
        counts = np.zeros(2001, dtype=np.int8)  # int8 is enough (max count is small)
        n = len(S)
        if n == 0:
            return counts
        # O(N^2) loop, fast for N ~ 200
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                counts[a + b] += 1
        return counts

    def greedy_fill(S_init, counts_init, candidates):
        """
        Greedily add elements from candidates to S_init.
        S_init: list of ints
        counts_init: numpy array of counts
        candidates: list of ints to try adding
        Returns: (S_final, counts_final)
        """
        S = list(S_init)
        counts = counts_init.copy()
        S_arr = np.array(S, dtype=np.int32) if S else np.array([], dtype=np.int32)
        
        for x in candidates:
            # Calculate new sums
            if len(S) > 0:
                new_sums = np.concatenate((x + S_arr, np.array([2 * x], dtype=np.int32)))
            else:
                new_sums = np.array([2 * x], dtype=np.int32)
            
            # Check if adding x violates the constraint
            # We need counts[new_sums] to be < 2 for all elements
            if np.all(counts[new_sums] < 2):
                # Valid
                counts[new_sums] += 1
                S.append(x)
                S_arr = np.append(S_arr, x)
        
        return S, counts

    # 1. Initial Baseline: Run a few random greedy constructions
    best_set = []
    best_size = 0
    
    # Run 10 random greedy restarts to find a good initial set
    for i in range(10):
        if time.time() - start_time > time_limit * 0.1:
            break
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        S, _ = greedy_fill([], np.zeros(2001, dtype=np.int8), candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 2. Kick and Regreedy Phase
    # We will repeatedly remove k random elements from the best set and try to refill it.
    # This allows escaping local optima of the greedy algorithm.
    
    while time.time() - start_time < time_limit:
        # Choose a kick size
        # Smaller kicks (2-5) for fine-tuning, larger (10-20) for larger jumps
        k = random.choice([2, 3, 4, 5, 6, 8, 10, 15, 20])
        
        # Ensure we don't kick more than we have
        if k > len(best_set):
            k = len(best_set)
        
        # Create a copy of the best set
        S_current = list(best_set)
        
        # Remove k random elements
        if k > 0:
            indices_to_remove = random.sample(range(len(S_current)), k)
            # Remove from end to start to keep indices valid
            for idx in sorted(indices_to_remove, reverse=True):
                S_current.pop(idx)
        
        # Recalculate counts for the reduced set
        counts_current = get_counts(S_current)
        
        # Prepare candidates: all numbers not in S_current
        S_set = set(S_current)
        candidates = [x for x in range(1, 1001) if x not in S_set]
        random.shuffle(candidates)
        
        # Greedily refill
        S_new, _ = greedy_fill(S_current, counts_current, candidates)
        
        # Update best if improved
        if len(S_new) > best_size:
            best_size = len(S_new)
            best_set = S_new
            
        # Occasionally try a pure random greedy restart
        if random.random() < 0.02:
            candidates_rand = list(range(1, 1001))
            random.shuffle(candidates_rand)
            S_rand, _ = greedy_fill([], np.zeros(2001, dtype=np.int8), candidates_rand)
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_set = S_rand

    return best_set
