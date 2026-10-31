import math
import random
import time
import numpy as np

def score_set(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Use sets for exactness
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        a = U[i]
        for j in range(n):
            b = U[j]
            sums.add(a + b)
            diffs.add(a - b)
            
    size_sum = len(sums)
    size_diff = len(diffs)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
        
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Helper to check if a list is a valid Golomb ruler (distinct differences)
    # and return the span
    def is_golomb(ruler):
        diffs = set()
        n = len(ruler)
        for i in range(n):
            for j in range(i+1, n):
                d = ruler[j] - ruler[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Helper to get the score of a ruler, assuming it's a valid set
    # We reuse score_set but maybe optimize if needed. 
    # For n < 100, score_set is fast enough.
    
    # Search for Golomb rulers of increasing size
    # We use a simple local search (shift/swap) to minimize span
    # Initial guess: greedy construction
    
    def greedy_golomb(n):
        ruler = [0]
        current_max = 0
        for _ in range(n-1):
            candidate = current_max + 1
            while True:
                test_ruler = ruler + [candidate]
                if is_golomb(test_ruler):
                    ruler = test_ruler
                    current_max = candidate
                    break
                candidate += 1
        return ruler

    def local_search_golomb(ruler, max_iters=10000):
        """
        Improves a Golomb ruler by minimizing span using local moves.
        Moves: shift an element (except 0) by +/- 1, or swap two elements.
        Must maintain Golomb property.
        """
        current_ruler = ruler[:]
        current_span = current_ruler[-1]
        best_ruler = current_ruler[:]
        best_span = current_span
        
        n = len(current_ruler)
        
        # Precompute differences for fast check? 
        # For n up to 50, checking is O(n^2). 
        # We can just re-check.
        
        for i in range(max_iters):
            # Check time
            if time.time() > deadline:
                break
            
            # Random move
            # 1. Shift a random element (index > 0) by +/- 1
            # 2. Swap two random elements (indices > 0)
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Shift
                idx = random.randint(1, n-1)
                delta = random.choice([-1, 1])
                new_val = current_ruler[idx] + delta
                if new_val < 0 or new_val >= current_ruler[-1]:
                    continue
                # Ensure no collision with other elements
                if new_val in current_ruler:
                    continue
                
                test_ruler = current_ruler[:]
                test_ruler[idx] = new_val
                test_ruler.sort()
                
                if is_golomb(test_ruler):
                    new_span = test_ruler[-1]
                    if new_span < current_span:
                        current_ruler = test_ruler
                        current_span = new_span
                        if current_span < best_span:
                            best_span = current_span
                            best_ruler = current_ruler[:]
            else:
                # Swap
                i_idx = random.randint(1, n-1)
                j_idx = random.randint(1, n-1)
                if i_idx == j_idx:
                    continue
                
                test_ruler = current_ruler[:]
                test_ruler[i_idx], test_ruler[j_idx] = test_ruler[j_idx], test_ruler[i_idx]
                test_ruler.sort()
                
                if is_golomb(test_ruler):
                    new_span = test_ruler[-1]
                    if new_span < current_span:
                        current_ruler = test_ruler
                        current_span = new_span
                        if current_span < best_span:
                            best_span = current_span
                            best_ruler = current_ruler[:]
                            
        return best_ruler

    # Try sizes from 3 up to a limit
    # We can go higher if time permits
    max_n = 60 
    for n in range(3, max_n + 1):
        if time.time() > deadline:
            break
            
        # Get initial guess
        initial_ruler = greedy_golomb(n)
        
        # Run local search
        # We can run multiple restarts for larger n if time permits
        restarts = 5 if n > 10 else 10
        
        best_ruler_for_n = initial_ruler
        best_span_for_n = initial_ruler[-1]
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            if r == 0:
                start_ruler = initial_ruler
            else:
                # Perturb the best found so far
                start_ruler = best_ruler_for_n[:]
                # Randomly swap a couple of elements to escape local minima
                for _ in range(3):
                    i_idx = random.randint(1, len(start_ruler)-1)
                    j_idx = random.randint(1, len(start_ruler)-1)
                    start_ruler[i_idx], start_ruler[j_idx] = start_ruler[j_idx], start_ruler[i_idx]
                    start_ruler.sort()
                    if not is_golomb(start_ruler):
                         # Revert if invalid
                         start_ruler = best_ruler_for_n[:]
                         break
            
            candidate = local_search_golomb(start_ruler, max_iters=5000)
            
            if candidate[-1] < best_span_for_n:
                best_span_for_n = candidate[-1]
                best_ruler_for_n = candidate
        
        # Evaluate score of the best ruler for this n
        # Note: The score depends on |U+U| and |U-U|, not just span.
        # But minimizing span for fixed n generally helps the denominator.
        # Also, Golomb rulers have maximal |U-U|.
        # So this is a good proxy.
        
        current_score = score_set(best_ruler_for_n)
        
        if current_score > best_score:
            best_score = current_score
            best_set = best_ruler_for_n
            
    return best_set
