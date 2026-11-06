"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    N = 256
    # Precompute conflict matrix
    # conflict[i][j] is True if i and j cannot be in the same set
    # i, j are 1..256. We use 0-indexed for internal list, so index i-1.
    conflict = [[False] * N for _ in range(N)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i-1][j-1] = True
                conflict[j-1][i-1] = True
                
    def greedy_construct(order):
        selected = []
        selected_set = set()
        # To speed up, we can maintain a list of candidates that are still compatible with all selected
        # But for N=256, simple check is fine.
        # We iterate through the provided order.
        for num in order:
            idx = num - 1
            compatible = True
            for sel in selected:
                if conflict[idx][sel - 1]:
                    compatible = False
                    break
            if compatible:
                selected.append(num)
                selected_set.add(num)
        return selected

    best_set = []
    
    # Run multiple random restarts
    # We can also try a few deterministic orders like "all odds first"
    # But randomization is the new approach.
    
    # Initialize best with a simple greedy (e.g., odds first) to have a baseline
    # Actually, let's just run random restarts.
    
    base_order = list(range(1, N + 1))
    # One run with odds prioritized (known good start)
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    evens = [i for i in range(1, N + 1) if i % 2 == 0]
    # Shuffle evens to be random but keep odds first?
    # The current program does odds then evens descending.
    # Let's do:
    # 1. Odds (random order)
    # 2. Evens (random order)
    
    for _ in range(100): # 100 restarts
        random.shuffle(odds)
        random.shuffle(evens)
        order = odds + evens
        s = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
            
    # Also try pure random orders
    for _ in range(50):
        random.shuffle(base_order)
        s = greedy_construct(base_order)
        if len(s) > len(best_set):
            best_set = s
            
    # Final local improvement: try to add any remaining number
    current_set = set(best_set)
    improved = True
    while improved:
        improved = False
        for num in range(1, N + 1):
            if num not in current_set:
                idx = num - 1
                compatible = True
                for sel in current_set:
                    if conflict[idx][sel - 1]:
                        compatible = False
                        break
                if compatible:
                    current_set.add(num)
                    improved = True
                    break # restart loop to check others again
    
    return list(current_set)
# EVOLVE-BLOCK-END
