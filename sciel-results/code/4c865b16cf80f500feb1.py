import random
import math

def construct():
    n = 256
    
    # Helper to check if x can be added to set S without forming a 3-AP
    def is_valid_addition(x, S):
        # x is not in S
        # Check if x forms a 3-AP with any two elements in S
        # Or if x is part of a 3-AP where x is one of the terms.
        
        # Case 1: x is the largest term (x = a + 2d, so a = x - 2d, b = x - d)
        # We need a, b in S.
        # d can range from 1 to (x-1)//2
        for d in range(1, (x + 1) // 2):
            if (x - d) in S and (x - 2 * d) in S:
                return False
        
        # Case 2: x is the middle term (x = a + d, so a = x - d, b = x + d)
        # We need a, b in S.
        # d can range from 1 to min(x-1, n-x)
        for d in range(1, min(x, n - x) + 1):
            if (x - d) in S and (x + d) in S:
                return False
        
        # Case 3: x is the smallest term (x = a, so b = x + d, c = x + 2d)
        # We need b, c in S.
        # d can range from 1 to (n-x)//2
        for d in range(1, (n - x + 1) // 2 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                return False
        
        return True

    # Helper to calculate sum of reciprocals
    def get_score(S):
        return sum(1.0 / x for x in S)

    # Initialization: Greedy construction to get a good starting point
    # This ensures we don't start from a terrible set
    S = set()
    # Try to add elements in order of decreasing weight (smallest x first)
    # This is the same as the seed program's greedy, which is a good baseline.
    for x in range(1, n + 1):
        if is_valid_addition(x, S):
            S.add(x)
    
    current_score = get_score(S)
    best_S = set(S)
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    max_steps = 50000  # Limit steps to ensure < 60s
    
    # Pre-compute all valid additions for each possible state? No, state changes.
    # Instead, we will propose moves.
    
    # We need to be careful with the "valid" check efficiency.
    # is_valid_addition is O(N^2) in worst case if we iterate d.
    # But N=256, so (x+1)//2 is ~128. 128 * 3 checks = ~400 ops.
    # 50,000 steps * 400 ops = 20M ops. This is well within 60s in Python.
    
    import time
    start_time = time.time()
    
    step = 0
    while T > min_T and step < max_steps:
        if time.time() - start_time > 50: # Safety break
            break
            
        # Choose a move type
        # 1. Add a random element not in S
        # 2. Remove a random element from S
        # 3. Swap a random element in S with a random element not in S
        
        move_type = random.choice(['add', 'remove', 'swap'])
        
        delta_score = 0.0
        new_S = set(S)
        valid_move = False
        
        if move_type == 'add':
            candidates = [x for x in range(1, n + 1) if x not in S]
            if candidates:
                x = random.choice(candidates)
                if is_valid_addition(x, S):
                    new_S.add(x)
                    delta_score = 1.0 / x
                    valid_move = True
                else:
                    # If invalid, we could still propose it and reject, 
                    # but it's better to only propose valid moves for 'add' 
                    # or handle invalidity as a penalty. 
                    # Standard SA usually proposes valid moves or penalizes.
                    # Let's stick to proposing only valid adds/removes/swaps to keep state valid.
                    # If no valid add, skip or try remove.
                    pass

        elif move_type == 'remove':
            if S:
                x = random.choice(list(S))
                new_S.remove(x)
                delta_score = -1.0 / x
                valid_move = True
            else:
                valid_move = False

        elif move_type == 'swap':
            if S:
                x_in = random.choice(list(S))
                candidates_out = [x for x in range(1, n + 1) if x not in S]
                if candidates_out:
                    x_out = random.choice(candidates_out)
                    # Try swapping x_in with x_out
                    # Remove x_in, then check if x_out can be added
                    S_temp = S - {x_in}
                    if is_valid_addition(x_out, S_temp):
                        new_S = S_temp | {x_out}
                        delta_score = (1.0 / x_out) - (1.0 / x_in)
                        valid_move = True
                    else:
                        valid_move = False
                else:
                    valid_move = False
            else:
                valid_move = False

        if valid_move:
            new_score = current_score + delta_score
            # Acceptance criterion
            if new_score > current_score:
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
            else:
                # Metropolis criterion
                prob = math.exp((new_score - current_score) / T)
                if random.random() < prob:
                    S = new_S
                    current_score = new_score
        
        T *= cooling_rate
        step += 1
        
    return list(best_S)
