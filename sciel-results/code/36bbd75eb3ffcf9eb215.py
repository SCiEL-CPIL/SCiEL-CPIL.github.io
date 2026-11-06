import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    best_indices = []
    best_deficit = Fraction(1, 1)
    
    def update_best(indices, deficit):
        nonlocal best_indices, best_deficit
        if deficit < best_deficit:
            best_deficit = deficit
            best_indices = indices
            return True
        return False

    def local_search(initial_indices):
        current_indices = set(initial_indices)
        current_sum = Fraction(0)
        for idx in current_indices:
            current_sum += reciprocals[idx]
        
        if current_sum >= 1:
            # Invalid start, try to fix by removing largest
            while current_sum >= 1 and current_indices:
                # Remove item with largest reciprocal (smallest denominator)
                max_val = Fraction(0)
                max_idx = -1
                for idx in current_indices:
                    if reciprocals[idx] > max_val:
                        max_val = reciprocals[idx]
                        max_idx = idx
                if max_idx != -1:
                    current_sum -= reciprocals[max_idx]
                    current_indices.remove(max_idx)
                else:
                    break
            if current_sum >= 1:
                return None, None

        current_deficit = Fraction(1) - current_sum
        update_best(list(current_indices), current_deficit)
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline and iterations < 100:
            improved = False
            iterations += 1
            best_new_deficit = current_deficit
            best_new_indices = None
            best_new_sum = None
            
            # 1. Try adding one item
            for j in range(n):
                if j in current_indices:
                    continue
                new_sum = current_sum + reciprocals[j]
                if new_sum < 1:
                    new_deficit = Fraction(1) - new_sum
                    if new_deficit < best_new_deficit:
                        best_new_deficit = new_deficit
                        best_new_indices = current_indices | {j}
                        best_new_sum = new_sum
                        improved = True
            
            # 2. Try swapping one item out and one in
            for i in list(current_indices):
                for j in range(n):
                    if j in current_indices:
                        continue
                    new_sum = current_sum - reciprocals[i] + reciprocals[j]
                    if new_sum < 1:
                        new_deficit = Fraction(1) - new_sum
                        if new_deficit < best_new_deficit:
                            best_new_deficit = new_deficit
                            new_indices_set = current_indices.copy()
                            new_indices_set.remove(i)
                            new_indices_set.add(j)
                            best_new_indices = new_indices_set
                            best_new_sum = new_sum
                            improved = True

            if best_new_indices is not None:
                current_indices = best_new_indices
                current_sum = best_new_sum
                current_deficit = best_new_deficit
                update_best(list(current_indices), current_deficit)
            else:
                # Stuck. Try a random kick: remove 1 or 2 random items
                if len(current_indices) > 1:
                    kick_size = 1 if random.random() < 0.5 else 2
                    if kick_size > len(current_indices):
                        kick_size = len(current_indices)
                    
                    items_to_remove = random.sample(list(current_indices), kick_size)
                    new_indices = current_indices.copy()
                    new_sum = current_sum
                    for idx in items_to_remove:
                        new_indices.remove(idx)
                        new_sum -= reciprocals[idx]
                    
                    # Greedy add from new state
                    for j in range(n):
                        if j in new_indices:
                            continue
                        if new_sum + reciprocals[j] < 1:
                            # Check if this specific add improves over current? 
                            # No, just accept the kick and let next iteration refine.
                            # But we should ensure the kick state is at least as good or we just accept it to escape.
                            # Actually, just set state to kicked state and continue loop.
                            pass 
                            
                    current_indices = new_indices
                    current_sum = new_sum
                    current_deficit = Fraction(1) - current_sum
                    # Note: This kick might worsen deficit, but allows escape.
                    # We don't update best here, only if it improves.
                    # The loop continues, and "improved" is False, so it would break.
                    # We need to force it to continue.
                    improved = True # Force another iteration to see if we can recover
                    
        return list(current_indices), current_deficit

    # Initial Greedy
    init1 = []
    s = Fraction(0)
    for i in range(n):
        if s + reciprocals[i] < 1:
            init1.append(i)
            s += reciprocals[i]
    
    if init1:
        local_search(init1)

    # Random Restarts
    random.seed(42)
    # Estimate time per restart. Local search takes some time.
    # Let's run until deadline.
    
    while time.time() < deadline - 5.0:
        shuffled_indices = list(range(n))
        random.shuffle(shuffled_indices)
        
        init_r = []
        s = Fraction(0)
        for i in shuffled_indices:
            if s + reciprocals[i] < 1:
                init_r.append(i)
                s += reciprocals[i]
        
        if init_r:
            local_search(init_r)

    return best_indices
