import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # seconds

def construct():
    t0 = time.time()
    
    # State: A, B are sorted lists; sumA, sumB are sets of pair sums; diffA, diffB are sets of signed differences
    A = []
    B = []
    sumA = set()
    sumB = set()
    diffA = set()
    diffB = set()
    
    def can_add_to(elements, sums, diffs, other_diffs, x):
        """Check if x can be added to the set."""
        if len(elements) >= MAX_EACH:
            return False
        # Check pair sums: for each a in elements, a+x must not be in sums; x+x must not be in sums
        # Actually, we need to check: for all a in elements (including x itself), a+x is new
        # And x+x is new
        # More precisely: new sums are {a+x for a in elements} ∪ {x+x}
        # These must not intersect with existing sums, and must be unique among themselves
        # Since elements is sorted and distinct, a+x for distinct a are distinct.
        # x+x is distinct from a+x iff x != a, which is true since x not in elements.
        # So we just need: ({a+x for a in elements} ∪ {2*x}) ∩ sums == ∅
        for a in elements:
            s = a + x
            if s in sums:
                return False
        if 2 * x in sums:
            return False
        # Check differences: new diffs are {x-a for a in elements} ∪ {a-x for a in elements}
        # These must not intersect with other_diffs
        for a in elements:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
            if d1 in diffs or d2 in diffs:
                # Wait, these should already be in diffs if a is in elements
                # Actually no - diffs contains differences between pairs in the set
                # x-a and a-x are new differences involving x
                # They shouldn't be in diffs already since x wasn't in the set
                # But they could conflict with other_diffs, which we check above
                pass
        return True
    
    def add_to(elements, sums, diffs, x):
        """Add x to the set, updating sums and diffs."""
        elements.append(x)
        elements.sort()
        for a in elements:
            if a != x:
                sums.add(a + x)
                diffs.add(x - a)
                diffs.add(a - x)
        sums.add(2 * x)
    
    def remove_from(elements, sums, diffs, x):
        """Remove x from the set."""
        elements.remove(x)
        for a in elements:
            sums.discard(a + x)
            diffs.discard(x - a)
            diffs.discard(a - x)
        sums.discard(2 * x)
    
    # Greedy construction
    for x in range(1, N + 1):
        if time.time() - t0 > DEADLINE:
            break
        # Try adding to A first, then B
        if can_add_to(A, sumA, diffA, diffB, x):
            add_to(A, sumA, diffA, x)
        elif can_add_to(B, sumB, diffB, diffA, x):
            add_to(B, sumB, diffB, x)
    
    # Local search: try to improve by swapping elements
    # Try: remove an element from A, add a different one
    improved = True
    while improved and time.time() - t0 < DEADLINE:
        improved = False
        for x in range(1, N + 1):
            if time.time() - t0 > DEADLINE:
                break
            if x in A or x in B:
                continue
            # Try adding x to A
            if can_add_to(A, sumA, diffA, diffB, x):
                add_to(A, sumA, diffA, x)
                improved = True
                break
            # Try adding x to B
            if can_add_to(B, sumB, diffB, diffA, x):
                add_to(B, sumB, diffB, x)
                improved = True
                break
    
    # Try swaps: remove one element, add another
    for _ in range(100):  # limited iterations
        if time.time() - t0 > DEADLINE:
            break
        # Try swapping an element in A with a new one
        for i, a in enumerate(A[:]):
            if time.time() - t0 > DEADLINE:
                break
            remove_from(A, sumA, diffA, a)
            for x in range(1, N + 1):
                if x == a or x in A or x in B:
                    continue
                if can_add_to(A, sumA, diffA, diffB, x):
                    add_to(A, sumA, diffA, x)
                    break
            # Check if we improved (size didn't change, but maybe freed up space)
            # Actually, this swap keeps size the same. We need to check if it helps.
            # For now, just do it and hope it helps later additions.
    
    return [A, B]
