import random
import time
import numpy as np

def construct():
    n = 128
    target_start = 128
    target_end = 256
    timeout = 55.0
    start_time = time.time()
    
    best_peak = float('inf')
    best_A = set()

    def get_counts(A_set):
        counts = [0] * (2 * n + 1)
        A_list = list(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                counts[a + b] += 1
        return counts

    def get_peak(counts):
        return max(counts[target_start:target_end + 1])

    def is_valid(counts):
        return all(counts[s] > 0 for s in range(target_start, target_end + 1))

    def add_element(A_set, counts, x):
        for y in A_set:
            counts[x + y] += 1
        counts[2 * x] += 1
        A_set.add(x)

    def remove_element(A_set, counts, x):
        for y in A_set:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        A_set.discard(x)

    def repair_and_optimize(A_init, counts_init):
        A = set(A_init)
        counts = counts_init[:]
        
        # Phase 1: Ensure coverage
        # We add elements to cover missing sums
        max_iterations = 200
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            
            missing = [s for s in range(target_start, target_end + 1) if counts[s] == 0]
            if not missing:
                break
            
            # Pick the most "isolated" missing sum? Or just the first?
            # Let's pick the one with fewest possible pairs from existing A?
            # Actually, we want to add an element that covers the most missing sums.
            
            best_element = None
            best_score = float('inf') # We want to minimize peak increase
            
            # Candidates: elements not in A
            # To be efficient, only consider elements that can form a sum in missing
            # If s is missing, a+b=s. a or b must be new.
            # Let's just iterate through all x in [0, 128] not in A
            
            candidates = [x for x in range(n + 1) if x not in A]
            if not candidates:
                break
                
            # Randomly sample candidates to speed up? Or iterate all if small?
            # 129 is small. Iterate all.
            
            for x in candidates:
                # Calculate new counts if we add x
                new_counts = counts[:]
                # Apply x
                for y in A:
                    new_counts[x + y] += 1
                new_counts[2 * x] += 1
                
                # Check how many missing sums are covered
                covered = sum(1 for s in missing if new_counts[s] > 0)
                if covered == 0:
                    continue
                
                new_peak = get_peak(new_counts)
                # Score: primarily peak, secondarily number covered (prefer more coverage)
                # If peak is same, prefer more coverage.
                score = new_peak * 1000 - covered
                
                if score < best_score:
                    best_score = score
                    best_element = x
                    best_new_counts = new_counts
            
            if best_element is not None:
                add_element(A, counts, best_element)
            else:
                # No single element covers any missing sum?
                # This implies we need to add two elements?
                # Or we are stuck. Break.
                break
        
        if not is_valid(counts):
            return None, None, None

        # Phase 2: Local Search Optimization
        current_peak = get_peak(counts)
        no_improve = 0
        
        while time.time() - start_time < timeout:
            improved = False
            
            # 1. Try removing elements
            A_list = list(A)
            random.shuffle(A_list)
            for a in A_list:
                if time.time() - start_time > timeout:
                    break
                remove_element(A, counts, a)
                if is_valid(counts):
                    new_peak = get_peak(counts)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        improved = True
                        no_improve = 0
                        continue
                    elif new_peak == current_peak:
                        # Keep if smaller set
                        improved = True
                        no_improve = 0
                        continue
                    else:
                        add_element(A, counts, a)
                else:
                    add_element(A, counts, a)
            
            # 2. Try swapping elements
            if not improved:
                A_list = list(A)
                random.shuffle(A_list)
                for a in A_list:
                    if time.time() - start_time > timeout:
                        break
                    
                    # Try swapping a with b
                    # Sample candidates for b
                    candidates = [x for x in range(n + 1) if x not in A]
                    if not candidates:
                        break
                    # Pick a few random candidates
                    for b in random.sample(candidates, min(20, len(candidates))):
                        if time.time() - start_time > timeout:
                            break
                        
                        remove_element(A, counts, a)
                        add_element(A, counts, b)
                        
                        if is_valid(counts):
                            new_peak = get_peak(counts)
                            if new_peak < current_peak:
                                current_peak = new_peak
                                improved = True
                                no_improve = 0
                                break # Restart swap loop
                            else:
                                # Revert
                                remove_element(A, counts, b)
                                add_element(A, counts, a)
                        else:
                            # Revert
                            remove_element(A, counts, b)
                            add_element(A, counts, a)
                    
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve > 20:
                    # Perturbation
                    # Add a random element, then try to remove others
                    candidates = [x for x in range(n + 1) if x not in A]
                    if candidates:
                        b = random.choice(candidates)
                        add_element(A, counts, b)
                        
                        # Try to remove elements to regain validity/peak
                        A_list = list(A)
                        random.shuffle(A_list)
                        for a in A_list:
                            if time.time() - start_time > timeout:
                                break
                            remove_element(A, counts, a)
                            if is_valid(counts):
                                new_peak = get_peak(counts)
                                if new_peak <= current_peak:
                                    current_peak = new_peak
                                    no_improve = 0
                                    break
                                else:
                                    add_element(A, counts, a)
                            else:
                                add_element(A, counts, a)
                        
                        # If perturbation didn't help and we are still worse or same, 
                        # we might want to revert to pre-perturbation state? 
                        # For simplicity, we keep the state and continue.
                        no_improve = 0
                    else:
                        break
            else:
                no_improve = 0

        return A, counts, current_peak

    # Run multiple restarts with different initial sizes
    sizes = [20, 25, 30, 35, 40, 50]
    num_restarts = 10
    
    for size in sizes:
        for _ in range(num_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Random initial set
            A_init = set(random.sample(range(n + 1), size))
            counts_init = get_counts(A_init)
            
            A, counts, peak = repair_and_optimize(A_init, counts_init)
            
            if A is not None and peak < best_peak:
                best_peak = peak
                best_A = A

    # If no valid set found (should not happen), fallback
    if not best_A:
        # Fallback to a known valid set
        A_fallback = set(range(64, 129))
        best_A = A_fallback

    return list(best_A)
