import time
from math import lcm
import random

def construct():
    # Denominators are 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2 to 48
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d_i
    weights = [L // d for d in denominators]
    
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin before 60s limit
    
    best_sum = 0
    best_indices = []
    
    # Helper function to evaluate a set of indices
    def evaluate(indices):
        s = sum(weights[i] for i in indices)
        if s < L and s > best_sum:
            return s, list(indices)
        return None
    
    # Randomized greedy restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Generate a random ordering
        if restart == 0:
            # First restart: sorted by weight descending
            order = sorted(range(n), key=lambda i: -weights[i])
        else:
            order = list(range(n))
            random.shuffle(order)
        
        # Greedy construction
        current_sum = 0
        current_set = []
        in_set = [False] * n
        
        for idx in order:
            w = weights[idx]
            if current_sum + w < L:
                current_sum += w
                current_set.append(idx)
                in_set[idx] = True
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = list(current_set)
        
        # Local search: try swaps and adds
        ls_deadline = min(deadline, time.time() + 0.1)
        while time.time() < ls_deadline:
            best_gain = 0
            best_move = None
            
            # Try Add moves
            for j in range(n):
                if not in_set[j]:
                    w_j = weights[j]
                    if current_sum + w_j < L:
                        gain = w_j
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('add', j)
            
            # Try Swap moves
            for i in range(n):
                if in_set[i]:
                    w_i = weights[i]
                    for j in range(n):
                        if not in_set[j]:
                            w_j = weights[j]
                            new_sum = current_sum - w_i + w_j
                            if new_sum < L and new_sum > current_sum:
                                gain = new_sum - current_sum
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = ('swap', i, j)
            
            if best_move is None:
                break
            
            # Apply best move
            if best_move[0] == 'add':
                j = best_move[1]
                in_set[j] = True
                current_set.append(j)
                current_sum += weights[j]
            else:  # swap
                i, j = best_move[1], best_move[2]
                in_set[i] = False
                in_set[j] = True
                current_set.remove(i)
                current_set.append(j)
                current_sum -= weights[i]
                current_sum += weights[j]
            
            # Update global best
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(current_set)
    
    # Ensure we have a valid non-empty set
    if not best_indices:
        best_indices = [0]  # 1/2 is always valid
        
    return best_indices
