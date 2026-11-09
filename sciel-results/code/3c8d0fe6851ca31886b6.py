"""Direct construction for 'Large sets with distinct subset products'."""

import time
import itertools

DEADLINE = 85.0  # wall-clock seconds (90s limit, safety margin)
T_START = time.time()

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# Pre-compute vectors
BIT_ALLOC = {2: 5, 3: 4, 5: 3, 7: 2}
for p in PRIMES[4:]:
    BIT_ALLOC[p] = 2

SHIFTS = {}
shift = 0
for p in PRIMES:
    SHIFTS[p] = shift
    shift += BIT_ALLOC[p]

def get_vector(n):
    v = 0
    for p in PRIMES:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        if e > 0:
            v |= (e << SHIFTS[p])
    return v

CANDIDATES = [(i, get_vector(i)) for i in range(2, 41)]

def _verify(items):
    """Verify all 2^n subset products are distinct."""
    n = len(items)
    prods = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= items[i]
        if prod in prods:
            return False
        prods.add(prod)
    return True

def _can_add(current_sums, vec):
    """Check if vec can be added to current_sums without collision."""
    # current_sums is a set of ints
    # New sums are s and s+vec
    # If any s+vec is in current_sums or in {s'+vec for s' in current_sums}, collision?
    # Actually, just check if len({s+vec for s in current_sums} | current_sums) == 2 * len(current_sums)
    new_sums = set()
    for s in current_sums:
        new_sums.add(s)
        new_sums.add(s + vec)
    return len(new_sums) == 2 * len(current_sums)

def _get_sums(items, vecs):
    sums = {0}
    for v in vecs:
        new_sums = set()
        for s in sums:
            new_sums.add(s)
            new_sums.add(s + v)
        sums = new_sums
    return sums

def construct():
    global DEADLINE, T_START
    
    # Greedy initialization
    # Sort by value descending
    candidates_sorted = sorted(CANDIDATES, key=lambda x: x[0], reverse=True)
    
    current_items = []
    current_vecs = []
    current_sums = {0}
    
    for val, vec in candidates_sorted:
        if _can_add(current_sums, vec):
            current_items.append(val)
            current_vecs.append(vec)
            # Update sums
            new_sums = set()
            for s in current_sums:
                new_sums.add(s)
                new_sums.add(s + vec)
            current_sums = new_sums
            
    best_items = current_items[:]
    best_size = len(current_items)
    
    # Local Search: Try swaps
    # We have a set of items. Try removing one and adding another.
    # To be efficient, we can just iterate through all pairs (remove, add)
    # But 39*39 is small.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try adding any remaining item
        remaining = [c for c in CANDIDATES if c[0] not in current_items]
        
        # Try simple additions first
        for val, vec in remaining:
            if _can_add(current_sums, vec):
                # Add it
                current_items.append(val)
                current_vecs.append(vec)
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s)
                    new_sums.add(s + vec)
                current_sums = new_sums
                improved = True
                if len(current_items) > best_size:
                    best_size = len(current_items)
                    best_items = current_items[:]
                # Reset loop to check for more additions
                remaining = [c for c in CANDIDATES if c[0] not in current_items]
                break
        
        if improved:
            continue
            
        # Try swaps
        # For each item in current_items, try replacing it with an item in remaining
        for i, (val_out, vec_out) in enumerate(list(zip(current_items, current_vecs))):
            for val_in, vec_in in remaining:
                if val_in == val_out:
                    continue
                # Check if we can remove val_out and add val_in
                # This is complex to check directly on sums.
                # Easier: reconstruct sums without val_out?
                # Or just check the new set.
                
                # Let's just construct the new set and check validity.
                # This is O(2^k) which is fine for k~16.
                
                new_items = current_items[:]
                new_items.pop(i)
                new_items.append(val_in)
                
                # Check validity
                if _verify(new_items):
                    # Update state
                    current_items = new_items
                    current_vecs = [get_vector(v) for v in current_items]
                    current_sums = _get_sums(current_items, current_vecs)
                    improved = True
                    if len(current_items) > best_size:
                        best_size = len(current_items)
                        best_items = current_items[:]
                    # Restart swap search
                    remaining = [c for c in CANDIDATES if c[0] not in current_items]
                    break
            if improved:
                break
                
    # Final verification
    if not _verify(best_items):
        best_items = [int(p) for p in PRIMES]
        
    return [int(x) for x in best_items]
