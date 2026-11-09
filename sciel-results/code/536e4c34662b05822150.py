import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0
    
    # Initial set based on manual analysis: [1..26] U {28,29,30,31,32,48}
    # Size 26 + 6 = 32
    initial_set = set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48])
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    current_set = initial_set
    current_score = evaluate(current_set)
    
    # Hill Climbing
    while time.time() - start_time < timeout:
        # Pick a random element to remove
        remove_idx = random.randint(0, 31)
        x_to_remove = list(current_set)[remove_idx]
        
        # Pick a random element to add
        # Range 1..512
        y_to_add = random.randint(1, 512)
        
        if y_to_add in current_set:
            continue
            
        new_set = set(current_set)
        new_set.remove(x_to_remove)
        new_set.add(y_to_add)
        
        new_score = evaluate(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
        elif new_score == current_score and random.random() < 0.1:
            # Plateau move to escape local minima
            current_set = new_set
    
    return list(current_set)
