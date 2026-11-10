import time
import random

def construct():
    deadline = time.time() + 85
    best_size = 0
    best_set = set()
    
    # Helper to check validity and update reachable sums
    # Using a list of bools or a set of ints. 
    # For speed, a bitset (int) is best.
    
    def get_reachable_bits(elements):
        bits = 1 # 0 is reachable
        for x in elements:
            bits |= (bits << x)
        return bits
    
    # Initial valid set
    current_set = list(range(66, 101))
    current_bits = get_reachable_bits(current_set)
    if (current_bits >> 200) & 1:
        # Should not happen
        current_set = []
        current_bits = 1
    
    best_size = len(current_set)
    best_set = set(current_set)
    
    # Try to improve by adding elements
    # We can try to add any element from 1 to 100 not in set
    candidates = [x for x in range(1, 101) if x not in current_set]
    
    # Greedy add
    for x in sorted(candidates, reverse=True):
        if time.time() > deadline:
            break
        new_bits = current_bits | (current_bits << x)
        if not ((new_bits >> 200) & 1):
            current_set.append(x)
            current_bits = new_bits
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            candidates.remove(x)
            
    # Randomized local search to escape local optima
    # We will try to swap elements or add/remove.
    # Since we want to maximize size, we prefer adding.
    # If we can't add, we might need to remove one to add two?
    # Or just shuffle the order of addition?
    
    # Let's try a few random restarts or perturbations
    # But we must keep the function fast.
    
    # One effective strategy: 
    # Start with empty set, add random elements?
    # No, start with a large valid set and try to expand.
    
    # If greedy didn't find > 35, maybe 35 is the max?
    # Let's try to find a set of size 36.
    
    # If we assume 35 is max, we return 35.
    # But maybe there is a 36.
    
    # Let's try a different initial set.
    # What about all odds?
    odds = [x for x in range(1, 101) if x % 2 == 1]
    bits = get_reachable_bits(odds)
    if not ((bits >> 200) & 1):
        if len(odds) > best_size:
            best_size = len(odds)
            best_set = set(odds)
            
    # What about evens?
    evens = [x for x in range(1, 101) if x % 2 == 0]
    bits = get_reachable_bits(evens)
    if not ((bits >> 200) & 1):
        if len(evens) > best_size:
            best_size = len(evens)
            best_set = set(evens)

    # The set {66..100} has size 35.
    # Odds has size 50.
    # Does {1, 3, ..., 99} have subset sum 200?
    # Sum of 100 odds is even.
    # 1+3+...+199 = 100^2 = 10000.
    # Can we sum to 200?
    # Yes, e.g., 1+3+...+39 (20 terms) = 20^2 = 400. Too big.
    # 1+3+...+13 (7 terms) = 49.
    # 1+3+5+7+9+11+13+15+17+19+21+23+25+27+29 = 225.
    # It is very likely that odds contain a subset summing to 200.
    # Let's check.
    
    # If odds are invalid, we need to remove some.
    
    # Let's rely on the greedy + bitset approach.
    # The seed program's greedy might be slow or stuck.
    # Bitset is fast.
    
    return list(best_set)
