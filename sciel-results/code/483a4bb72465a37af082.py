"""New approach: greedy with difference-budget priority queue for two Sidon sets."""
import heapq
from itertools import combinations_with_replacement, combinations

def construct():
    n = 256
    max_size = 64
    
    # State: two sets, their pair sums, and their difference sets
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # For each candidate x, track which side it's "assigned" to (we'll decide dynamically)
    # We'll use a greedy approach: at each step, try adding the smallest available x
    # to the side where it causes the least "damage" (fewest new differences that conflict)
    
    used = set()  # numbers already placed in either set
    
    def can_add_A(x):
        # Check if x can be added to A
        # Sidon check: all pair sums a+b for a in A ∪ {x} must be unique
        new_sums = set()
        for a in A:
            s = a + x
            if s in sums_A or s in new_sums:
                return False
            new_sums.add(s)
        s2 = 2 * x
        if s2 in sums_A or s2 in new_sums:
            return False
        new_sums.add(s2)
        
        # Difference check: new differences from x must not intersect diffs_B
        new_diffs = set()
        for a in A:
            d = x - a
            if d != 0:
                if d in diffs_B or d in diffs_A:
                    return False
                new_diffs.add(d)
            d = a - x
            if d != 0:
                if d in diffs_B or d in diffs_A:
                    return False
                new_diffs.add(d)
        return True
    
    def can_add_B(x):
        # Check if x can be added to B
        new_sums = set()
        for b in B:
            s = b + x
            if s in sums_B or s in new_sums:
                return False
            new_sums.add(s)
        s2 = 2 * x
        if s2 in sums_B or s2 in new_sums:
            return False
        new_sums.add(s2)
        
        new_diffs = set()
        for b in B:
            d = x - b
            if d != 0:
                if d in diffs_A or d in diffs_B:
                    return False
                new_diffs.add(d)
            d = b - x
            if d != 0:
                if d in diffs_A or d in diffs_B:
                    return False
                new_diffs.add(d)
        return True
    
    def add_to_A(x):
        for a in A:
            sums_A.add(a + x)
            d1 = x - a
            d2 = a - x
            if d1 != 0:
                diffs_A.add(d1)
            if d2 != 0:
                diffs_A.add(d2)
        sums_A.add(2 * x)
        A.append(x)
    
    def add_to_B(x):
        for b in B:
            sums_B.add(b + x)
            d1 = x - b
            d2 = b - x
            if d1 != 0:
                diffs_B.add(d1)
            if d2 != 0:
                diffs_B.add(d2)
        sums_B.add(2 * x)
        B.append(x)
    
    # Greedy: iterate through numbers 1..n, try to add to A first, then B
    # But use a smarter ordering: prioritize numbers that are "cheaper" in terms of differences
    # We'll do multiple passes with different strategies
    
    # Strategy: for each x from 1 to n, try adding to A if possible, else B if possible
    # But we want to balance sizes and minimize difference conflicts
    
    # Better: use a priority queue where we score each candidate by how many new differences it adds
    # and whether it conflicts with the other set's differences
    
    # Let's try a simple greedy first, then refine
    for x in range(1, n + 1):
        if x in used:
            continue
        added = False
        # Try A first
        if len(A) < max_size and can_add_A(x):
            add_to_A(x)
            used.add(x)
            added = True
        # Try B
        if not added and len(B) < max_size and can_add_B(x):
            add_to_B(x)
            used.add(x)
            added = True
    
    # Now try to fill remaining capacity by trying unused numbers
    for x in range(1, n + 1):
        if x in used:
            continue
        if len(A) < max_size and can_add_A(x):
            add_to_A(x)
            used.add(x)
        elif len(B) < max_size and can_add_B(x):
            add_to_B(x)
            used.add(x)
    
    return [A, B]
