import random
import time

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200.
    Maximizes cardinality.
    """
    TARGET = 200
    N = 100
    MASK = (1 << (TARGET + 1)) - 1
    
    def try_greedy(order):
        bits = 1  # Bit 0 is set, representing sum 0
        selected = []
        for x in order:
            # Check if adding x creates a subset sum of 200
            # new_reachable = bits | (bits << x)
            # If bit 200 is set in new_reachable, we cannot add x
            if (bits | (bits << x)) & (1 << TARGET):
                continue
            # Safe to add x
            selected.append(x)
            bits = (bits | (bits << x)) & MASK
        return selected

    # 1. Try deterministic orders
    best_set = try_greedy(range(N, 0, -1))
    best_size = len(best_set)
    
    # Also try ascending
    set_asc = try_greedy(range(1, N + 1))
    if len(set_asc) > best_size:
        best_set = set_asc
        best_size = len(set_asc)

    # 2. Random restarts with a time limit
    # The evaluator kills after 90s, we use 50s for search to be safe.
    start_time = time.time()
    time_limit = 50.0
    
    # Use a fixed seed for reproducibility in local testing, but random for evaluation?
    # The prompt doesn't require determinism, but it's good practice to have a fallback.
    # However, random restarts are the core of the improvement.
    
    while time.time() - start_time < time_limit:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        current_set = try_greedy(random_order)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
    return best_set
