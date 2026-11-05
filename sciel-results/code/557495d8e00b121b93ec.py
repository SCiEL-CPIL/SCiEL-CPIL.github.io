import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)  # 47 denominators
    
    # Start with a known good solution
    best_indices = [0, 1, 5, 41]  # {2,3,7,43}, deficit = 1/1806
    best_deficit = Fraction(1, 1806)
    
    # Meet-in-the-middle: split into two halves
    # For each half, enumerate all subset sums (as Fractions)
    # Then combine to find best total < 1
    
    mid = n // 2  # 23
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    # Enumerate all subset sums for left half
    # 2^23 = 8.4M subsets - might be too many, let's be smarter
    # Actually, let's use a different approach: greedy + local search
    
    # Approach: Start with greedy, then try to improve via local search
    # Greedy: add denominators in order of largest reciprocal that keeps sum < 1
    current = []
    current_sum = Fraction(0, 1)
    
    # Try all starting points and greedy completions
    for first_idx in range(n):
        if time.time() > deadline:
            break
        trial = [first_idx]
        trial_sum = Fraction(1, denominators[first_idx])
        if trial_sum >= 1:
            continue
        # Greedily add more
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if j in trial:
                    continue
                new_sum = trial_sum + Fraction(1, denominators[j])
                if new_sum < 1:
                    trial.append(j)
                    trial_sum = new_sum
                    changed = True
                    break
        # Evaluate
        deficit = 1 - trial_sum
        if deficit > 0 and deficit < best_deficit:
            best_deficit = deficit
            best_indices = trial[:]
    
    # Local search: try removing one and adding others
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(best_indices)
            current_sum = sum(Fraction(1, denominators[i]) for i in best_indices)
            
            # Try removing one element
            for i in list(current_set):
                new_set = current_set - {i}
                new_sum = current_sum - Fraction(1, denominators[i])
                # Try adding any other element
                for j in range(n):
                    if j in new_set:
                        continue
                    candidate_sum = new_sum + Fraction(1, denominators[j])
                    if candidate_sum < 1:
                        candidate_deficit = 1 - candidate_sum
                        if candidate_deficit < best_deficit:
                            best_deficit = candidate_deficit
                            best_indices = list(new_set | {j})
                            current_set = set(best_indices)
                            current_sum = candidate_sum
                            improved = True
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            # Try adding one element
            for j in range(n):
                if j in current_set:
                    continue
                candidate_sum = current_sum + Fraction(1, denominators[j])
                if candidate_sum < 1:
                    candidate_deficit = 1 - candidate_sum
                    if candidate_deficit < best_deficit:
                        best_deficit = candidate_deficit
                        best_indices = list(current_set | {j})
                        improved = True
                        break
                if time.time() > deadline:
                    break
    
    # Verify
    total = sum(Fraction(1, denominators[i]) for i in best_indices)
    if total < 1:
        return best_indices
    else:
        return [0]
