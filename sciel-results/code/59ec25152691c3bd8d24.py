"""Improved program: conflict-graph based local search for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _is_valid_triple(a, b, c):
    """Check if three distinct elements have all three pairwise gcds equal."""
    g_ab = gcd(a, b)
    g_ac = gcd(a, c)
    g_bc = gcd(b, c)
    return not (g_ab == g_ac == g_bc)

def _conflicts_with_existing(a, selected):
    """Check if adding 'a' to selected creates any invalid triple.
    An invalid triple (x, y, a) exists iff there are x,y in selected with gcd(x,y)==gcd(x,a)==gcd(y,a).
    """
    for x in selected:
        g_xa = gcd(x, a)
        for y in selected:
            if x == y:
                continue
            if gcd(x, y) == g_xa and gcd(y, a) == g_xa:
                return False
    return True

def _greedy_construct(order):
    """Greedy construction following given order."""
    selected = []
    for x in order:
        if _conflicts_with_existing(x, selected):
            selected.append(x)
    return selected

def _local_search(initial, deadline):
    """Local search: try 1-for-1 swaps and additions to improve cardinality."""
    selected = list(initial)
    best = len(selected)
    remaining = [x for x in range(1, 129) if x not in selected]
    
    # Precompute gcd table for speed
    gcd_table = {}
    for a in range(1, 129):
        for b in range(a, 129):
            gcd_table[(a, b)] = gcd(a, b)
    
    def gcd_lookup(a, b):
        if a > b:
            a, b = b, a
        return gcd_table[(a, b)]
    
    def check_add(a, sel):
        """Check if a can be added to sel without creating invalid triple."""
        for i, x in enumerate(sel):
            g_xa = gcd_lookup(x, a)
            for y in sel[i+1:]:
                if gcd_lookup(x, y) == g_xa and gcd_lookup(y, a) == g_xa:
                    return False
        return True
    
    def check_swap_remove_add(a_out, a_in, sel):
        """Check if removing a_out and adding a_in keeps validity."""
        sel_no_out = [x for x in sel if x != a_out]
        if not check_add(a_in, sel_no_out):
            return False
        # Also need to verify that a_in doesn't form bad triple with pairs that included a_out
        # Actually check_add already handles all pairs in sel_no_out
        # But we also need: for any x,y in sel_no_out, the triple (x,y,a_in) is valid - check_add covers this
        # And for any x in sel_no_out, triple (x, a_in, ?) - wait, we need triples of 3 from the new set
        # The new set is sel_no_out + {a_in}. All triples either don't involve a_in (already valid since they were in sel minus a_out, and removing elements can't create new invalid triples) or involve a_in (checked by check_add).
        return True
    
    # Try to add elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in remaining[:]:
            if check_add(a, selected):
                selected.append(a)
                remaining.remove(a)
                improved = True
                best = max(best, len(selected))
        # Try swaps: remove one from selected, add one from remaining
        for i in range(len(selected)):
            for a in remaining:
                if check_swap_remove_add(selected[i], a, selected):
                    # Check that the new set is actually valid
                    new_sel = [x for j, x in enumerate(selected) if j != i] + [a]
                    # Verify full validity
                    valid = True
                    for x, y, z in combinations(new_sel, 3):
                        if not _is_valid_triple(x, y, z):
                            valid = False
                            break
                    if valid:
                        selected = new_sel
                        remaining.remove(a)
                        improved = True
                        best = max(best, len(selected) + 1)
                        break
            if improved:
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Multiple random orders for greedy
    best_set = []
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial % 3 == 0:
            order = list(range(1, 129))
        elif trial % 3 == 1:
            order = list(range(128, 0, -1))
        else:
            order = list(range(1, 129))
            random.shuffle(order)
        
        selected = _greedy_construct(order)
        
        # Local search to improve
        selected = _local_search(selected, deadline)
        
        if len(selected) > len(best_set):
            best_set = selected
    
    # Final attempt with more thorough local search
    if time.time() < deadline - 5 and best_set:
        best_set = _local_search(best_set, deadline)
    
    if not best_set:
        # Fallback: just return {1} which is always valid (no triples possible)
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
