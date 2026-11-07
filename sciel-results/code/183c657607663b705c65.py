import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 128
    
    def is_valid(S):
        """Check if set S has no forbidden triple."""
        S_list = list(S)
        S_set = set(S_list)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                ab = a + b
                if ab > N:
                    continue
                for k in range(j + 1, n):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac > N or bc > N:
                        continue
                    if ab in S_set and ac in S_set and bc in S_set:
                        return False
        return True
    
    def count_violations(S):
        """Count number of violating triples in S."""
        S_list = list(S)
        S_set = set(S_list)
        n = len(S_list)
        violations = 0
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                ab = a + b
                if ab > N:
                    continue
                for k in range(j + 1, n):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac > N or bc > N:
                        continue
                    if ab in S_set and ac in S_set and bc in S_set:
                        violations += 1
        return violations
    
    # Start with the known valid interval [63, 128]
    S = set(range(63, 129))
    best_S = set(S)
    best_size = len(S)
    
    # Try to add elements from 1 to 62
    candidates = list(range(1, 63))
    random.shuffle(candidates)
    
    # Greedy phase: try to add each candidate if it keeps the set valid
    for x in candidates:
        if time.time() > deadline:
            break
        S_new = S | {x}
        if is_valid(S_new):
            S = S_new
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
    
    # Local search phase: try swapping elements
    # We can try replacing one element with another to potentially allow more additions
    all_elements = list(range(1, N + 1))
    current_S = set(best_S)
    
    # Simulated annealing-like local search
    T = 10.0  # Initial temperature
    cooling = 0.99
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        if T < 0.01:
            break
            
        # Try a random move: add a random element not in S
        not_in_S = [x for x in all_elements if x not in current_S]
        if not not_in_S:
            break
            
        x = random.choice(not_in_S)
        S_new = current_S | {x}
        
        if is_valid(S_new):
            current_S = S_new
            if len(current_S) > best_size:
                best_S = set(current_S)
                best_size = len(current_S)
        else:
            # Try removing a random element and adding x
            if current_S:
                y = random.choice(list(current_S))
                S_swap = (current_S - {y}) | {x}
                if is_valid(S_swap):
                    current_S = S_swap
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_size = len(current_S)
        
        T *= cooling
    
    # Final attempt: try all possible additions to best_S
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x not in best_S:
            S_test = best_S | {x}
            if is_valid(S_test):
                best_S = S_test
    
    result = [int(x) for x in best_S]
    return result
