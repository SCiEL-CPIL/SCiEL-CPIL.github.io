"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200.
    Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Greedy construction from largest to smallest
    # This tends to produce sets with large minimum element,
    # which reduces the number of elements needed to reach the target sum.
    selected = []
    reachable = {0}
    
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        # Check if adding x creates a sum of 200
        # We need to check if there exists s in reachable such that s + x == 200
        # i.e., 200 - x is in reachable
        target = 200 - x
        if target in reachable:
            continue
        
        # If not, we can safely add x
        # Update reachable sums
        # Only sums <= 200 are relevant for future checks (actually < 200)
        new_sums = {s + x for s in reachable if s + x < 200}
        # Note: we don't add 200 itself because we check before adding
        # But we must ensure that no combination of existing sums + x equals 200.
        # The check `if target in reachable` covers `s + x = 200`.
        # What about `s1 + s2 + x = 200`?
        # If `s1 + s2` was in reachable, then `target = s1+s2` would be in reachable.
        # So the check is sufficient.
        
        selected.append(x)
        reachable |= new_sums
        
    # Local Search Improvement
    # Try to swap elements to increase size?
    # Or try to add elements by removing others?
    # Given the time constraint, we will attempt a simple local search:
    # Try to add any element not in selected by removing the minimum number of elements.
    # However, the greedy set {51..100} is likely optimal for this specific target/interval.
    
    # Let's try a few random restarts with different orders if time permits?
    # No, the greedy from high to low is structurally strong.
    
    # Ensure non-empty
    if not selected:
        return [100]
        
    return selected
# EVOLVE-BLOCK-END
