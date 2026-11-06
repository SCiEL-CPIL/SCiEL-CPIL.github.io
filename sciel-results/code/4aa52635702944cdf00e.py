import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    min_target = 128
    max_target = 256
    targets = list(range(min_target, max_target + 1))
    
    def get_counts(A):
        # A is a set of ints
        # Generate all unordered pairs (a, b) with a <= b
        # Using list comprehension for speed
        A_list = list(A)
        n = len(A_list)
        sums = []
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if min_target <= s <= max_target:
                    sums.append(s)
        counts = Counter(sums)
        # Convert to list for fast access
        count_list = [0] * (max_target + 1)
        for s, c in counts.items():
            count_list[s] = c
        return count_list

    def evaluate(A):
        count_list = get_counts(A)
        # Check coverage
        for t in targets:
            if count_list[t] == 0:
                return None, None, None
        peak = max(count_list[t] for t in targets)
        return A, peak, count_list

    best_A = None
    best_peak = float('inf')
    
    # Search for optimal set size
    # Based on analysis, k around 16-20 is likely
    for k in range(16, 26):
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Ensure we don't pick duplicates
            A = set(random.sample(range(0, 129), k))
            
            current_A, current_peak, current_counts = evaluate(A)
            
            if current_A is None:
                continue
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = current_A
                
            if best_peak == 1:
                return list(best_A)
                
            # Local Search
            # We try to reduce the peak
            for step in range(500):
                if time.time() > deadline:
                    break
                
                if current_peak == 1:
                    return list(current_A)
                
                # Pick a random element to remove
                a_out = random.choice(list(current_A))
                
                # Pick a random element to add from the complement
                # Optimization: Pre-compute complement? No, just filter.
                # range(0, 129) has 129 elements. A has ~20.
                # Filtering is fast.
                candidates = [x for x in range(0, 129) if x not in current_A]
                if not candidates:
                    break
                a_in = random.choice(candidates)
                
                new_A = set(current_A)
                new_A.remove(a_out)
                new_A.add(a_in)
                
                new_A_list, new_peak, new_counts = evaluate(new_A)
                
                if new_A_list is None:
                    # Coverage broken, reject
                    continue
                
                # Acceptance criteria
                if new_peak < current_peak:
                    current_A = new_A_list
                    current_peak = new_peak
                    current_counts = new_counts
                    if current_peak == 1:
                        return list(current_A)
                elif new_peak == current_peak:
                    # Accept to explore landscape
                    if random.random() < 0.3:
                        current_A = new_A_list
                        current_counts = new_counts
                else:
                    # Reject worse peak
                    pass

    # If we didn't find peak 1, return the best found
    if best_A is not None:
        return list(best_A)
    
    # Fallback: Return a valid set even if not optimal
    # The seed range(64, 129) is valid
    return list(range(64, 129))
