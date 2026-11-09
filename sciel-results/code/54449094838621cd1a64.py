import time
import sys
import random

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def build_bitmasks(A):
    """Build bitmask representation of subset sums by cardinality.
    masks[j] is a Python int where bit s is 1 if sum s is achievable with exactly j elements."""
    n = len(A)
    masks = [0] * (n + 1)
    masks[0] = 1  # bit 0 set: empty subset has sum 0
    for x in A:
        shift = 1 << x
        for j in range(n - 1, -1, -1):
            if masks[j]:
                masks[j + 1] |= (masks[j] & shift)
    return masks

def is_valid_bitmasks(masks, n):
    """Check that no two different cardinalities share any sum."""
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if masks[j] & masks[k]:
                return False
    return True

def validate(A):
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    n = len(A)
    masks = build_bitmasks(A)
    return is_valid_bitmasks(masks, n)

def greedy_construct():
    """Greedy: try adding each number 1..256 if it maintains validity."""
    A = []
    masks = [1]  # masks[0] = 1 (bit 0 set)
    for x in range(1, 257):
        if is_done():
            break
        # Try adding x
        new_masks = masks[:]
        n = len(A)
        # Update: new_masks[j+1] |= new_masks[j] shifted by x
        # But we need to use the OLD masks for the update
        for j in range(n, -1, -1):
            if masks[j]:
                new_masks[j + 1] |= (masks[j] << x)
        # Check validity
        valid = True
        for j in range(n + 2):
            for k in range(j + 1, n + 2):
                if new_masks[j] & new_masks[k]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            A.append(x)
            masks = new_masks
    return A

def local_search(A, max_time=30.0):
    """Local search: try swapping elements to find a larger valid set.
    Start with a set, try to grow it by replacing elements with unused ones."""
    start_time = time.time()
    A = list(A)
    n = len(A)
    
    # Try to grow the set by one element at a time
    # Strategy: try all pairs (remove i, add x) where x not in A
    # If valid, keep the swap. Repeat until stuck.
    
    best = A[:]
    best_size = len(A)
    
    used = set(A)
    
    while not is_done() and (time.time() - start_time) < max_time:
        # Try to add one more element
        added = False
        for x in range(1, 257):
            if x in used:
                continue
            if is_done():
                break
            # Try adding x to current set
            candidate = A + [x]
            cn = len(candidate)
            masks = [0] * (cn + 1)
            masks[0] = 1
            for elem in candidate:
                for j in range(cn - 1, -1, -1):
                    if masks[j]:
                        masks[j + 1] |= (masks[j] << elem)
            if is_valid_bitmasks(masks, cn):
                A = candidate
                used.add(x)
                added = True
                if len(A) > best_size:
                    best = A[:]
                    best_size = len(A)
                break  # restart search for next addition
        
        if not added:
            # Try swap-based improvement: remove one element, add a different one
            # This allows exploring non-greedy combinations
            swapped = False
            for i in range(len(A)):
                if is_done() or (time.time() - start_time) > max_time:
                    break
                removed = A[i]
                remaining = A[:i] + A[i+1:]
                rused = used - {removed}
                for x in range(1, 257):
                    if x in rused:
                        continue
                    if is_done() or (time.time() - start_time) > max_time:
                        break
                    candidate = remaining + [x]
                    cn = len(candidate)
                    masks = [0] * (cn + 1)
                    masks[0] = 1
                    for elem in candidate:
                        for j in range(cn - 1, -1, -1):
                            if masks[j]:
                                masks[j + 1] |= (masks[j] << elem)
                    if is_valid_bitmasks(masks, cn):
                        A = candidate
                        used = set(A)
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                break
    
    if len(best) > len(A):
        return best
    return A if len(A) >= len(best) else best

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Greedy construction
    greedy = greedy_construct()
    if len(greedy) > len(base_solution):
        base_solution = greedy
    
    # Local search to improve
    time_left = DEADLINE - elapsed()
    if time_left > 10:
        improved = local_search(base_solution, max_time=min(time_left - 5, 40.0))
        if len(improved) > len(base_solution):
            base_solution = improved
    
    # Final validation
    if validate(base_solution):
        return base_solution
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
