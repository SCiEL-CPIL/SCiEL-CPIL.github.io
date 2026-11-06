import numpy as np
import time

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Use numpy bitset for subset sums up to 200
    # Bit i set means sum i is reachable
    # Start with bit 0 set (empty subset)
    
    def can_add(current_bits, x):
        """Check if adding x would make 200 reachable, and return new bits if safe."""
        new_bits = current_bits | (current_bits << x)
        # Mask to only bits 0..200
        mask = (1 << 201) - 1
        new_bits &= mask
        if (new_bits >> 200) & 1:
            return None  # Would make 200 reachable
        return new_bits
    
    # Strategy 1: Greedy from large to small (current approach, but faster)
    def greedy(order):
        bits = 1  # only sum 0 reachable
        selected = []
        for x in order:
            new_bits = can_add(bits, x)
            if new_bits is not None:
                selected.append(x)
                bits = new_bits
        return selected
    
    # Try multiple orderings
    best = []
    
    # Ordering 1: descending
    order1 = list(range(100, 0, -1))
    result = greedy(order1)
    if len(result) > len(best):
        best = result
    
    # Ordering 2: ascending
    order2 = list(range(1, 101))
    result = greedy(order2)
    if len(result) > len(best):
        best = result
    
    # Ordering 3: Large numbers first, then small (biased toward keeping large)
    # Try: take all > 100/2 = 50, then try to add smaller ones
    order3 = list(range(100, 50, -1)) + list(range(50, 0, -1))
    result = greedy(order3)
    if len(result) > len(best):
        best = result
    
    # Ordering 4: Try to maximize by starting with largest possible set
    # Heuristic: numbers > 200/2 = 100... but max is 100
    # Numbers in (100, 200] can't be in a subset summing to 200 alone, 
    # but we only have up to 100.
    
    # Try: all numbers from 51 to 100 first, then 1 to 50
    order4 = list(range(100, 50, -1)) + list(range(50, 0, -1))
    result = greedy(order4)
    if len(result) > len(best):
        best = result
    
    # Local search: try to swap out excluded elements for included ones
    # or add excluded elements by removing some included ones
    if time.time() < deadline - 5:
        # Try to improve by local search
        current_set = set(best)
        bits = 1
        for x in current_set:
            bits |= (bits << x)
            bits &= (1 << 201) - 1
        
        excluded = [x for x in range(1, 101) if x not in current_set]
        
        # Try adding each excluded element
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in excluded[:]:
                new_bits = can_add(bits, x)
                if new_bits is not None:
                    current_set.add(x)
                    bits = new_bits
                    excluded.remove(x)
                    improved = True
                    break
        
        if len(current_set) > len(best):
            best = list(current_set)
    
    # More aggressive: try random orderings
    if time.time() < deadline - 10:
        import random
        for _ in range(200):
            if time.time() >= deadline - 5:
                break
            order = list(range(1, 101))
            random.shuffle(order)
            result = greedy(order)
            if len(result) > len(best):
                best = result
    
    # Final: try to add any remaining elements to best
    if time.time() < deadline - 2:
        current_set = set(best)
        bits = 1
        for x in current_set:
            bits |= (bits << x)
            bits &= (1 << 201) - 1
        
        for x in range(1, 101):
            if x not in current_set:
                new_bits = can_add(bits, x)
                if new_bits is not None:
                    current_set.add(x)
                    bits = new_bits
        
        if len(current_set) > len(best):
            best = list(current_set)
    
    return best
