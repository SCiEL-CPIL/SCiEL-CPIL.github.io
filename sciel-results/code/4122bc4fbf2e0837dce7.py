import numpy as np

N = 18

def _generate_odd_even_transposition(n):
    """Generate the standard Odd-Even Transposition network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _is_valid_network(comparators, n):
    """
    Check if the network sorts all 2^n boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Generate all 2^n boolean inputs
    # Shape: (2^n, n)
    # Using np.arange and bit shifting
    size = 1 << n
    # Create an array of indices 0 to 2^n - 1
    indices = np.arange(size, dtype=np.int32)
    
    # Extract bits
    # We want bits 0 to n-1
    # bit k is (indices >> k) & 1
    inputs = (indices[:, None] >> np.arange(n, dtype=np.int32)) & 1
    
    # Simulate the network
    for i, j in comparators:
        a = inputs[:, i]
        b = inputs[:, j]
        
        min_val = np.minimum(a, b)
        max_val = np.maximum(a, b)
        
        inputs[:, i] = min_val
        inputs[:, j] = max_val
    
    # Check if sorted
    # A sequence is sorted if x[i] <= x[i+1] for all i
    # We can check np.all(inputs[:, :-1] <= inputs[:, 1:])
    # But we need it for ALL rows.
    is_sorted = np.all(inputs[:, :-1] <= inputs[:, 1:], axis=1)
    
    return np.all(is_sorted)

def construct():
    """
    Construct a sorting network for 18 wires.
    Uses a heuristic optimization (greedy removal) on the Odd-Even Transposition network.
    """
    n = N
    # Start with a valid network
    comparators = _generate_odd_even_transposition(n)
    
    # Optional: Pre-check validity (should be true)
    # if not _is_valid_network(comparators, n):
    #     return comparators
    
    # Greedy removal
    # We iterate through the list of comparators.
    # If removing a comparator keeps the network valid, we remove it.
    # We do this in a single pass first. We can do multiple passes if time permits.
    
    # To maximize removals, we might want to try removing from the end (last comparators are often redundant for final cleanup)
    # Or from the beginning.
    # Let's try iterating from the end to the beginning.
    
    # We need to be careful with the list modification.
    # We'll build a new list of kept comparators.
    
    # Actually, a better strategy:
    # Try to remove each comparator in the current list.
    # Since the list changes, we can do a while loop or a for loop with a flag.
    
    # Given the time limit (60s), and that each check is ~0.1-0.5s,
    # we can afford ~100-300 checks.
    # The network has 153 comparators.
    # So we can do roughly 1-2 passes of checking all comparators.
    
    current_comparators = comparators[:]
    
    # We'll try to remove comparators one by one.
    # To avoid O(N^2) checks in the worst case if we restart, we just do a single pass
    # checking each comparator in the *current* list.
    
    # However, removing one might make another redundant.
    # A simple single pass might not be optimal, but it's a good start.
    
    # Let's do a few passes if time allows.
    # But we don't have a time checker inside easily.
    # Let's just do one pass from the end to the start.
    
    for idx in range(len(current_comparators) - 1, -1, -1):
        # Try removing current_comparators[idx]
        # Create a new list without it
        candidate = current_comparators[:idx] + current_comparators[idx+1:]
        
        if _is_valid_network(candidate, n):
            # Valid, so remove it
            current_comparators = candidate
    
    return current_comparators
