import time
import random
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (90s limit)
    
    # Numbers of interest: 2 to 256
    N = 256
    numbers = list(range(2, N + 1))
    num_to_idx = {x: i for i, x in enumerate(numbers)}
    n = len(numbers)
    
    # Precompute conflicts
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            if lcm(a, b) <= 256:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    def is_compatible(idx, current_set, skip_idx=None):
        """Check if numbers[idx] is compatible with all indices in current_set."""
        for s in current_set:
            if s == skip_idx:
                continue
            if s in conflicts[idx]:
                return False
        return True
        
    def get_score(current_set):
        return sum(1.0 / numbers[i] for i in current_set)
        
    def improve(initial_set, max_time=None):
        """Local search to improve the set with multiple move types."""
        if max_time is None:
            max_time = deadline
        current_set = set(initial_set)
        current_score = get_score(current_set)
        
        improved = True
        while improved and (time.time() < max_time):
            improved = False
            
            # 1. Try to add a number (smallest first for max weight)
            candidates = [i for i in range(n) if i not in current_set]
            candidates.sort(key=lambda i: numbers[i])
            
            added = False
            for idx in candidates:
                if is_compatible(idx, current_set):
                    current_set.add(idx)
                    current_score += 1.0 / numbers[idx]
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # 2. Try 1-swap: replace x with y where y < x
            swapped = False
            for x_idx in list(current_set):
                x_val = numbers[x_idx]
                best_y = -1
                best_gain = 0.0
                
                for y_val in range(2, x_val):
                    if y_val not in num_to_idx:
                        continue
                    y_idx = num_to_idx[y_val]
                    if y_idx in current_set:
                        continue
                    
                    if is_compatible(y_idx, current_set, skip_idx=x_idx):
                        gain = (1.0 / y_val) - (1.0 / x_val)
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y_idx
                
                if best_y >= 0:
                    current_set.remove(x_idx)
                    current_set.add(best_y)
                    current_score += best_gain
                    improved = True
                    swapped = True
                    break
                
            if swapped:
                continue
            
            # 3. Try 2-swap: remove x1, x2; add y1, y2 where net gain > 0
            # Only try a limited number of 2-swaps for time efficiency
            if time.time() < max_time - 5.0:
                two_swapped = False
                elements = list(current_set)
                # Try removing pairs of larger elements
                for i in range(len(elements)):
                    if two_swapped:
                        break
                    for j in range(i + 1, len(elements)):
                        if two_swapped or time.time() >= max_time - 2.0:
                            break
                        x1_idx = elements[i]
                        x2_idx = elements[j]
                        x1_val = numbers[x1_idx]
                        x2_val = numbers[x2_idx]
                        
                        # Find best pair to add
                        best_pair = None
                        best_pair_gain = 0.0
                        
                        # Try all pairs y1 < y2 not in current_set
                        # This is expensive, so limit to small values
                        for y1_val in range(2, max(x1_val, x2_val)):
                            if y1_val in num_to_idx and y1_val not in [numbers[k] for k in current_set]:
                                y1_idx = num_to_idx[y1_val]
                                if is_compatible(y1_idx, current_set, skip_idx=x1_idx) and \
                                   is_compatible(y1_idx, current_set, skip_idx=x2_idx):
                                    for y2_val in range(y1_val + 1, max(x1_val, x2_val)):
                                        if y2_val in num_to_idx and y2_val not in [numbers[k] for k in current_set]:
                                            y2_idx = num_to_idx[y2_val]
                                            if is_compatible(y2_idx, current_set, skip_idx=x1_idx) and \
                                               is_compatible(y2_idx, current_set, skip_idx=x2_idx) and \
                                               is_compatible(y1_idx, {y2_idx} | (current_set - {x1_idx, x2_idx})):
                                                gain = (1.0/y1_val + 1.0/y2_val) - (1.0/x1_val + 1.0/x2_val)
                                                if gain > best_pair_gain:
                                                    best_pair_gain = gain
                                                    best_pair = (y1_idx, y2_idx)
                        
                        if best_pair is not None:
                            current_set.remove(x1_idx)
                            current_set.remove(x2_idx)
                            current_set.add(best_pair[0])
                            current_set.add(best_pair[1])
                            current_score += best_pair_gain
                            improved = True
                            two_swapped = True
                            break
            
        return current_set, current_score

    # Generate diverse initial solutions
    initial_sets = []
    
    # 1. Greedy from 2
    seed_set = []
    for x in range(2, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in seed_set):
            seed_set.append(idx)
    initial_sets.append(seed_set)
    
    # 2. Greedy from 3
    g3 = []
    for x in range(3, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in g3):
            g3.append(idx)
    initial_sets.append(g3)
    
    # 3. Greedy from 4
    g4 = []
    for x in range(4, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in g4):
            g4.append(idx)
    initial_sets.append(g4)
    
    # 4. Greedy from 5
    g5 = []
    for x in range(5, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in g5):
            g5.append(idx)
    initial_sets.append(g5)
    
    # 5-14. Random restarts with different strategies
    random.seed(42)
    for trial in range(10):
        rand_set = []
        # Strategy: shuffle small numbers, pick compatible ones
        smalls = list(range(2, 60))
        random.shuffle(smalls)
        for x in smalls:
            idx = num_to_idx[x]
            if all(idx not in conflicts[seed_idx] for seed_idx in rand_set):
                rand_set.append(idx)
        # Fill with remaining
        for x in range(2, 257):
            if any(numbers[k] == x for k in rand_set):
                continue
            idx = num_to_idx[x]
            if all(idx not in conflicts[seed_idx] for seed_idx in rand_set):
                rand_set.append(idx)
        initial_sets.append(rand_set)
    
    # 15-20. Random restarts biased toward very small numbers
    for trial in range(6):
        rand_set = []
        very_smalls = list(range(2, 20))
        random.shuffle(very_smalls)
        for x in very_smalls:
            idx = num_to_idx[x]
            if all(idx not in conflicts[seed_idx] for seed_idx in rand_set):
                rand_set.append(idx)
        for x in range(2, 257):
            if any(numbers[k] == x for k in rand_set):
                continue
            idx = num_to_idx[x]
            if all(idx not in conflicts[seed_idx] for seed_idx in rand_set):
                rand_set.append(idx)
        initial_sets.append(rand_set)
    
    best_set = None
    best_score = -1.0
    
    # Allocate time per initial solution
    time_per_solution = (deadline - start_time - 2.0) / len(initial_sets)
    
    for s in initial_sets:
        if time.time() > deadline - 1.0:
            break
        sol_deadline = min(deadline - 1.0, time.time() + time_per_solution)
        improved_set, score = improve(s, max_time=sol_deadline)
        if score > best_score:
            best_score = score
            best_set = improved_set
    
    if best_set is None:
        return []
        
    # Final polish: one more round of improvement on the best solution
    if time.time() < deadline - 2.0:
        polished_set, polished_score = improve(best_set, max_time=deadline - 1.0)
        if polished_score > best_score:
            best_set = polished_set
            best_score = polished_score
            
    # Convert indices back to values
    return [numbers[i] for i in best_set]

if __name__ == "__main__":
    res = construct()
    print(len(res), sum(1/x for x in res))
