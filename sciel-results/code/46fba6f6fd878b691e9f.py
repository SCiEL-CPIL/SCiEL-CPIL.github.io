import time
import random

MAX_SUM = 7680  # 30 * 256

# Precompute forbidden multiples bitset for each possible sum value
# forbidden_multiples[s] is a bitset where bit i is set if i is a multiple of s (i > s, i <= MAX_SUM)
forbidden_multiples = [0] * (MAX_SUM + 1)
for s in range(1, MAX_SUM + 1):
    mask = 0
    for multiple in range(s * 2, MAX_SUM + 1, s):
        mask |= (1 << multiple)
    forbidden_multiples[s] = mask

def _validate_full(elements):
    """
    Returns True if the set of elements is valid.
    Uses bitset for fast divisibility checking.
    """
    n = len(elements)
    if n == 0:
        return True
    
    # Generate all subset sums using bitset
    sums_bitset = 1  # bit 0 set for empty sum
    
    for e in elements:
        # Shift left by e and OR with existing
        new_sums = sums_bitset << e
        # Check for collision: if any bit is set in both sums_bitset and new_sums (excluding bit 0)
        if (sums_bitset & new_sums) & ~1:
            return False
        sums_bitset |= new_sums
    
    # Remove empty set sum (bit 0)
    sums_bitset &= ~1
    
    # Check divisibility: for each sum s, check if any multiple of s is also in the bitset
    # We can do this by checking if (sums_bitset & forbidden_multiples[s]) != 0 for each s in sums_bitset
    # But we need to iterate over set bits in sums_bitset
    
    # Extract all set bits from sums_bitset
    temp = sums_bitset
    while temp:
        # Get lowest set bit
        lsb = temp & (-temp)
        s = lsb.bit_length() - 1
        # Check if any multiple of s is in sums_bitset
        if sums_bitset & forbidden_multiples[s]:
            return False
        temp &= temp - 1  # Clear the lowest set bit
    
    return True

def _try_add(elements, bitset, x):
    """
    Try to add x to elements. Returns (new_elements, new_bitset) if valid, else (None, None).
    """
    # Check if x is already in elements
    if x in elements:
        return None, None
    
    # New bitset = bitset | (bitset << x)
    new_bitset = bitset | (bitset << x)
    
    # Check for sum collision
    if (bitset & (bitset << x)) & ~1:
        return None, None
    
    # Check divisibility violations
    # We need to check:
    # 1. New sums don't divide existing sums
    # 2. Existing sums don't divide new sums
    # 3. New sums don't divide each other
    
    # For efficiency, let's check by iterating over new sums (bits that are in new_bitset but not in bitset)
    new_sums_only = new_bitset & ~bitset
    
    # Check each new sum against all existing sums and other new sums
    temp = new_sums_only
    while temp:
        lsb = temp & (-temp)
        s = lsb.bit_length() - 1
        
        # Check if s divides any existing sum
        if bitset & forbidden_multiples[s]:
            return None, None
        
        # Check if s divides any other new sum (excluding itself)
        # forbidden_multiples[s] has bits for multiples > s
        # We need to check if any of those are in new_sums_only (excluding s itself)
        if (new_sums_only & ~lsb) & forbidden_multiples[s]:
            return None, None
        
        temp &= temp - 1
    
    # Also need to check if existing sums divide new sums
    # For each existing sum, check if it divides any new sum
    # This is equivalent to: for each existing sum e, check if new_sums_only & forbidden_multiples[e] != 0
    # But we can optimize: only check existing sums that could divide some new sum
    
    # Actually, the above checks cover "new sums divide existing" and "new sums divide new sums"
    # We still need to check "existing sums divide new sums"
    # Let's iterate over existing sums
    temp2 = bitset
    while temp2:
        lsb2 = temp2 & (-temp2)
        e = lsb2.bit_length() - 1
        # Check if e divides any new sum
        if new_sums_only & forbidden_multiples[e]:
            return None, None
        temp2 &= temp2 - 1
    
    return elements + [x], new_bitset

def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    best_set = []
    best_size = 0
    
    # Start with a random large number
    initial = random.randint(100, 256)
    best_set = [initial]
    best_size = 1
    
    # Compute initial bitset
    best_bitset = 1 | (1 << initial)  # bit 0 and bit initial
    
    # Try to grow the set incrementally
    for _ in range(10000):  # Max iterations
        if time.time() > deadline:
            break
        
        # Try to add a random element
        x = random.randint(1, 256)
        new_set, new_bitset = _try_add(best_set, best_bitset, x)
        
        if new_set is not None:
            best_set = new_set
            best_bitset = new_bitset
            best_size = len(best_set)
        
        # If we've reached a good size, try harder to improve
        if best_size >= 5:
            # Do local search: try swapping elements
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                idx = random.randint(0, len(best_set) - 1)
                old_val = best_set[idx]
                new_val = random.randint(1, 256)
                
                if new_val == old_val or new_val in best_set:
                    continue
                
                # Remove old_val, add new_val
                temp_set = best_set[:]
                temp_set[idx] = new_val
                temp_set.sort()
                
                # Recompute bitset for temp_set
                temp_bitset = 1
                valid = True
                for e in temp_set:
                    shifted = temp_bitset << e
                    if (temp_bitset & shifted) & ~1:
                        valid = False
                        break
                    temp_bitset |= shifted
                
                if not valid:
                    continue
                
                temp_bitset &= ~1
                
                # Check divisibility
                valid = True
                temp2 = temp_bitset
                while temp2:
                    lsb = temp2 & (-temp2)
                    s = lsb.bit_length() - 1
                    if temp_bitset & forbidden_multiples[s]:
                        valid = False
                        break
                    temp2 &= temp2 - 1
                
                if valid:
                    # Now try to add an element
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        add_val = random.randint(1, 256)
                        new_set2, new_bitset2 = _try_add(temp_set, temp_bitset, add_val)
                        if new_set2 is not None:
                            best_set = new_set2
                            best_bitset = new_bitset2
                            best_size = len(best_set)
                            break
                    break
    
    # Final validation
    if not _validate_full(best_set):
        # Fallback
        best_set = [20, 21, 23]
    
    return list(best_set)
