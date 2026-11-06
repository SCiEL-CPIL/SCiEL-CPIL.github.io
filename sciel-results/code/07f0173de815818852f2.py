"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE = time.time() + 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered sums a+b, a<=b distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_blocking(order, deadline):
    """
    Greedy construction: iterate elements in given order, add if it doesn't break Sidon property.
    Returns (selected, sums_set) for the constructed set.
    """
    selected = []
    sums = set()
    for x in order:
        if time.time() > deadline:
            break
        # Check if adding x preserves Sidon property
        new_sums = [x + a for a in selected] + [2 * x]
        # All new sums must be distinct from each other and from existing sums
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(set(new_sums)):
            selected.append(x)
            sums.update(new_sums)
    return selected, sums

def _is_maximal(selected, sums):
    """Check if no missing element can be added while preserving Sidon property."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Try adding x
        new_sums = [x + a for a in selected] + [2 * x]
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(set(new_sums)):
            return False
    return True

def _construct_with_order(order):
    """Build a Sidon set using the given order, then check maximality."""
    selected, sums = _greedy_blocking(order, DEADLINE)
    if not selected:
        return None
    if _is_maximal(selected, sums):
        return selected
    return None

def construct():
    """
    Search for a small inclusion-maximal Sidon set in [1,256].
    Uses randomized greedy with multiple restarts, prioritizing orders that
    lead to early saturation (fewer elements selected).
    """
    best = None
    best_size = N + 1
    
    # Try deterministic ascending first as a baseline
    base_order = list(range(1, N + 1))
    result = _construct_with_order(base_order)
    if result:
        best = result
        best_size = len(result)
    
    # Try descending order
    desc_order = list(range(N, 0, -1))
    result = _construct_with_order(desc_order)
    if result and len(result) < best_size:
        best = result
        best_size = len(result)
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    
    # Strategy: random permutations, but also try orders that favor
    # elements that block many candidates early (heuristic ordering)
    
    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() > DEADLINE - 2.0:
            break
        
        if best is not None and best_size <= 10:
            # If we already have a very small set, stop
            break
        
        # Generate a randomized order with some structure
        order = list(range(1, N + 1))
        
        # Mix strategies:
        if restart % 3 == 0:
            # Pure random shuffle
            rng.shuffle(order)
        elif restart % 3 == 1:
            # Weighted random: bias toward middle values first
            # (middle values tend to create more conflicts)
            weights = [1.0 / (1.0 + abs(x - N/2)) for x in order]
            # Sample without replacement using weights
            chosen = []
            remaining = list(order)
            remaining_weights = list(weights)
            for _ in range(N):
                total_w = sum(remaining_weights)
                if total_w == 0:
                    chosen.extend(remaining)
                    break
                r = rng.random() * total_w
                cumulative = 0
                for i, w in enumerate(remaining_weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen.append(remaining[i])
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break
            order = chosen
        else:
            # Partial random: keep some structure
            # Split into blocks and shuffle within blocks, then shuffle block order
            block_size = rng.choice([8, 16, 32])
            blocks = [order[i:i+block_size] for i in range(0, N, block_size)]
            for block in blocks:
                rng.shuffle(block)
            rng.shuffle(blocks)
            order = [x for block in blocks for x in block]
        
        result = _construct_with_order(order)
        if result and len(result) < best_size:
            best = result
            best_size = len(result)
    
    # Final attempt: if we have a good candidate, try local search
    # to see if we can find something even smaller with more careful ordering
    if best is not None and time.time() < DEADLINE - 5.0:
        # Try to improve by trying more targeted random orders
        for _ in range(500):
            if time.time() > DEADLINE - 2.0:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            result = _construct_with_order(order)
            if result and len(result) < best_size:
                best = result
                best_size = len(result)
    
    # Fallback: if no maximal set found (shouldn't happen), return ascending greedy result
    if best is None:
        selected, sums = _greedy_blocking(base_order, DEADLINE)
        best = selected
    
    return best

# EVOLVE-BLOCK-END
