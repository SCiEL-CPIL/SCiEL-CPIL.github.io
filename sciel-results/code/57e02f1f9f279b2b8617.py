import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - start_time
    
    def check_valid(selected):
        """Check if all triple sums are unique."""
        sums = set()
        for triple in combinations_with_replacement(selected, 3):
            s = sum(triple)
            if s in sums:
                return False
            sums.add(s)
        return True
    
    def greedy_with_order(order, selected, used_sums):
        """Try to extend selected set using given order."""
        for x in order:
            if x in selected:
                continue
            # Generate new sums involving x
            new_sums = set()
            for i, j in combinations_with_replacement(range(len(selected)), 2):
                s = selected[i] + selected[j] + x
                new_sums.add(s)
            # Also x+x+x
            new_sums.add(3 * x)
            # And x+x+selected[i]
            for i in range(len(selected)):
                new_sums.add(2 * x + selected[i])
            
            # Check if any new sum conflicts
            if new_sums & used_sums:
                continue
            
            # Accept x
            selected.append(x)
            used_sums |= new_sums
            if elapsed() > time_limit:
                break
        
        return selected, used_sums
    
    # Strategy 1: Greedy from small to large
    best_set = []
    best_size = 0
    
    # Initial greedy: try numbers in increasing order
    selected = []
    used_sums = set()
    for x in range(1, 4097):
        if elapsed() > time_limit:
            break
        new_sums = set()
        for i, j in combinations_with_replacement(range(len(selected)), 2):
            new_sums.add(selected[i] + selected[j] + x)
        new_sums.add(3 * x)
        for i in range(len(selected)):
            new_sums.add(2 * x + selected[i])
        
        if not (new_sums & used_sums):
            selected.append(x)
            used_sums |= new_sums
    
    best_set = selected[:]
    best_size = len(selected)
    
    # Strategy 2: Try different starting points and orderings
    # Use a power-of-3 style construction as a strong baseline
    # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729, 3^7=2187
    # That's 8 elements. Can we do better?
    
    # Let's try a greedy with random perturbations
    random.seed(42)
    
    for trial in range(50):
        if elapsed() > time_limit - 5:
            break
        
        # Create a random ordering biased towards smaller numbers
        # but with some randomness
        order = list(range(1, 4097))
        random.shuffle(order)
        # Bias: sort by a random weight
        weights = [random.random() ** 2 for _ in range(4096)]  # Favor smaller numbers
        order.sort(key=lambda x: weights[x-1])
        
        selected = []
        used_sums = set()
        for x in order:
            if elapsed() > time_limit:
                break
            new_sums = set()
            for i, j in combinations_with_replacement(range(len(selected)), 2):
                new_sums.add(selected[i] + selected[j] + x)
            new_sums.add(3 * x)
            for i in range(len(selected)):
                new_sums.add(2 * x + selected[i])
            
            if not (new_sums & used_sums):
                selected.append(x)
                used_sums |= new_sums
        
        if len(selected) > best_size:
            best_set = selected[:]
            best_size = len(selected)
    
    # Local improvement: try to add more elements
    if elapsed() < time_limit - 10:
        # Try to add elements that weren't in the best set
        current_set = best_set[:]
        current_used = set()
        for triple in combinations_with_replacement(current_set, 3):
            current_used.add(sum(triple))
        
        for x in range(1, 4097):
            if elapsed() > time_limit:
                break
            if x in current_set:
                continue
            new_sums = set()
            for i, j in combinations_with_replacement(range(len(current_set)), 2):
                new_sums.add(current_set[i] + current_set[j] + x)
            new_sums.add(3 * x)
            for i in range(len(current_set)):
                new_sums.add(2 * x + current_set[i])
            
            if not (new_sums & current_used):
                current_set.append(x)
                current_used |= new_sums
        
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
    
    return best_set
