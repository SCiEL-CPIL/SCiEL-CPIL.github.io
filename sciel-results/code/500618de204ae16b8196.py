import time
import random
from itertools import combinations

START_TIME = time.time()
DEADLINE = 55.0  # seconds (safety margin from 60s)

def elapsed():
    return time.time() - START_TIME

def build_bitsets(elements):
    """Build list of bitsets: bitsets[k] has bit s set iff sum s is achievable with k elements."""
    n = len(elements)
    bitsets = [0] * (n + 1)
    bitsets[0] = 1  # bit 0 set: sum 0 achievable with 0 elements
    
    for elem in elements:
        shift = 1 << elem
        for k in range(n, -1, -1):
            if bitsets[k]:
                bitsets[k + 1] |= (bitsets[k] << elem)
    return bitsets

def is_valid_bitset(bitsets):
    """Check that no sum appears in two different cardinality bitsets."""
    for i in range(len(bitsets)):
        for j in range(i + 1, len(bitsets)):
            if bitsets[i] & bitsets[j]:
                return False
    return True

def count_cross_collisions(bitsets):
    """Count total number of cross-cardinality sum collisions."""
    total = 0
    for i in range(len(bitsets)):
        for j in range(i + 1, len(bitsets)):
            total += bin(bitsets[i] & bitsets[j]).count('1')
    return total

def greedy_construct(order, target_size, max_value=256):
    """Greedy construction following a specific ordering of candidate values."""
    elements = []
    max_value = 256
    
    for v in order:
        if len(elements) >= target_size:
            break
        if elapsed() > DEADLINE - 2:
            break
        
        new_elements = elements + [v]
        new_bitsets = build_bitsets(new_elements)
        
        if is_valid_bitset(new_bitsets):
            elements.append(v)
    
    return elements

def local_search_improve(elements, target_size, max_value=256):
    """Try to add one more element using local search."""
    if len(elements) >= target_size:
        return elements
    
    elements = elements[:]
    used = set(elements)
    
    best_new = None
    best_collisions = float('inf')
    
    for v in range(1, max_value + 1):
        if v in used:
            continue
        new_bitsets = build_bitsets(elements + [v])
        collisions = count_cross_collisions(new_bitsets)
        if collisions == 0:
            if best_new is None or v > best_new:
                best_new = v
                best_collisions = 0
    
    if best_new is not None:
        elements.append(best_new)
    
    return elements

def construct():
    """Main entry point: construct the largest valid set possible."""
    if elapsed() > DEADLINE:
        return []
    
    best = []
    max_value = 256
    
    # Try multiple orderings to find the largest valid set
    orderings = [
        # Ascending
        list(range(1, max_value + 1)),
        # Descending  
        list(range(max_value, 0, -1)),
        # Large values first (known good strategy)
        list(range(max_value, max_value - 60, -1)) + list(range(1, max_value - 60)),
        # Random shuffles
    ]
    
    # Add several random shuffles
    rng = random.Random(42)
    for _ in range(10):
        shuffled = list(range(1, max_value + 1))
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Try each ordering
    for order in orderings:
        if elapsed() > DEADLINE - 3:
            break
        
        result = greedy_construct(order, 60, max_value)
        if len(result) > len(best):
            best = result
            
            # Try to extend further with local search
            for _ in range(5):
                if elapsed() > DEADLINE - 3:
                    break
                improved = local_search_improve(best, len(best) + 1, max_value)
                if len(improved) > len(best):
                    best = improved
                else:
                    break
    
    # Final verification
    if best:
        bitsets = build_bitsets(best)
        if not is_valid_bitset(bitsets):
            # Fallback to known valid set
            best = list(range(226, 257))
    
    return best
