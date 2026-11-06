import random
import math

def construct():
    # First get greedy solution as starting point
    def is_valid(selected, x):
        # Check if adding x to selected keeps validity
        # For each a in selected + [x], check no 3-distinct-others sum is divisible by a
        candidate = selected + [x]
        for i, a in enumerate(candidate):
            others = [v for j, v in enumerate(candidate) if j != i]
            # Check all 3-combinations of others
            from itertools import combinations
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True
    
    # Greedy from large to small
    greedy_selected = []
    for x in range(192, 0, -1):
        if is_valid(greedy_selected, x):
            greedy_selected.append(x)
    
    # Now try to improve with simulated annealing
    # State: a set of selected numbers
    # Moves: 
    #   1. Try to add a number not in set (if valid)
    #   2. Remove a number from set
    #   3. Swap: remove one, add another
    
    best = list(greedy_selected)
    best_size = len(best)
    
    # Simulated annealing
    current = list(best)
    current_size = len(current)
    
    T = 10.0
    cooling = 0.999
    iterations = 5000
    
    random.seed(42)
    
    for it in range(iterations):
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a number not in current
            candidates = [x for x in range(1, 193) if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            if is_valid(current, x):
                new_state = current + [x]
                new_size = len(new_state)
                # Always accept if it improves size
                if new_size > current_size:
                    current = new_state
                    current_size = new_size
                    if current_size > best_size:
                        best = list(current)
                        best_size = current_size
                else:
                    # Same size, accept with some probability
                    if random.random() < math.exp(0 / T):  # delta = 0
                        current = new_state
            T *= cooling
            
        elif move_type < 0.7:
            # Remove a random element
            if not current:
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            new_state = current[:idx] + current[idx+1:]
            new_size = len(new_state)
            delta = new_size - current_size  # -1
            # Accept if random < exp(delta/T)
            if delta >= 0 or random.random() < math.exp(delta / T):
                current = new_state
                current_size = new_size
            T *= cooling
            
        else:
            # Swap: remove one, try to add another
            if not current:
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            temp = current[:idx] + current[idx+1:]
            candidates = [x for x in range(1, 193) if x not in temp]
            if candidates:
                x = random.choice(candidates)
                if is_valid(temp, x):
                    new_state = temp + [x]
                    new_size = len(new_state)
                    delta = new_size - current_size  # 0
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current = new_state
                        current_size = new_size
                        if current_size > best_size:
                            best = list(current)
                            best_size = current_size
                T *= cooling
            else:
                T *= cooling
    
    return best
