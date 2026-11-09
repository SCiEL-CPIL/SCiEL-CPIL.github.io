import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # 60s safety margin
    
    # 1. Greedy Construction (Seed Logic)
    def greedy_construct():
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Helper to check validity and get conflict
    def is_valid_and_conflict(S):
        reachable = {0}
        for x in S:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 in new:
                # Find the specific subset
                # This is slow, but we only do it occasionally
                return False, None 
            reachable |= new
        return True, None

    # Local Search
    # We try to swap elements to increase size
    # Since we want max size, we start with best_set
    # We try to add an element from outside, and if it fails, we try to remove some from inside to make it fit?
    # Actually, simpler: Random restarts with different orderings?
    
    # Try a few random orderings for the greedy step
    numbers = list(range(1, 101))
    
    # Try 20 random permutations
    for i in range(20):
        if time.time() > deadline:
            break
        random.shuffle(numbers)
        # Greedy on shuffled order
        selected, reachable = [], {0}
        for x in numbers:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected

    # Try local improvement on best_set
    # Try to add elements from 1..100 not in best_set
    # If conflict, try to resolve by removing one element from the conflicting subset?
    # This is complex.
    
    return best_set
