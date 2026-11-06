import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    n = 256
    
    # Perfect squares up to 2*256 = 512
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Start with the strong seed: numbers not divisible by 3
    # This gives ~171 elements and scores 1.0
    seed_set = set(a for a in range(1, n + 1) if a % 3 != 0)
    
    # Verify the seed is valid
    def is_valid(s):
        s_list = sorted(s)
        for idx, a in enumerate(s_list):
            if (2 * a) in squares:
                return False
            for b in s_list[idx:]:
                if (a + b) in squares:
                    return False
        return True
    
    if not is_valid(seed_set):
        # Fallback: build from scratch with greedy
        candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
        conflict = {a: set() for a in candidates}
        for idx, a in enumerate(candidates):
            for b in candidates[idx+1:]:
                if a + b in squares:
                    conflict[a].add(b)
                    conflict[b].add(a)
        
        best_set = set()
        for _ in range(50):
            order = candidates[:]
            random.shuffle(order)
            selected = set()
            for a in order:
                if not (conflict[a] & selected):
                    selected.add(a)
            if len(selected) > len(best_set):
                best_set = selected
        seed_set = best_set
    
    current_set = set(seed_set)
    current_size = len(current_set)
    
    # Build conflict structure for all numbers 1..256
    all_nums = list(range(1, n + 1))
    conflict = {a: set() for a in all_nums}
    for idx, a in enumerate(all_nums):
        for b in all_nums[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Local search: try to add elements that were excluded (multiples of 3)
    excluded = [a for a in all_nums if a not in current_set]
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > deadline - 5:
            break
        
        # Try adding each excluded element
        for a in excluded:
            if a in current_set:
                continue
            if (2 * a) in squares:
                continue
            if not (conflict[a] & current_set):
                current_set.add(a)
                excluded.remove(a)
                improved = True
                current_size = len(current_set)
    
    # Try single-element swaps: remove one, add multiple
    if time.time() - start_time > deadline - 10:
        return list(current_set)
    
    for _ in range(3):  # Multiple rounds of swaps
        if time.time() - start_time > deadline - 5:
            break
        swap_improved = False
        for rem in list(current_set):
            if time.time() - start_time > deadline - 5:
                break
            remaining = current_set - {rem}
            # Find all addable
            addable = []
            for a in all_nums:
                if a in remaining or a == rem:
                    continue
                if (2 * a) in squares:
                    continue
                if not (conflict[a] & remaining):
                    addable.append(a)
            
            if len(remaining) + len(addable) > len(current_set):
                current_set = remaining | set(addable)
                swap_improved = True
                break
        
        if not swap_improved:
            break
    
    result = list(current_set)
    if not result:
        result = [1]
    return result
