"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin

def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits for a list of elements."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        sum_bits |= (1 << (a + b))
    diff_bits = 0
    for a, b in combinations(elems, 2):
        diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits

def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to the set defined by elems, sum_bits, diff_bits."""
    # Check sum collisions
    # elem + elem
    if (sum_bits >> (elem + elem)) & 1:
        return False
    # elem + a for all a in elems
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check difference collisions within self
    for a in elems:
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
    # Check difference collisions with other set
    if (other_diff_bits >> 0) & 1:  # placeholder, actual check below
        pass
    for a in elems:
        d = abs(elem - a)
        if (other_diff_bits >> d) & 1:
            return False
    return True

def _add_elem(elem, elems, sum_bits, diff_bits):
    """Add elem to elems and update bits."""
    new_sum_bits = sum_bits
    new_diff_bits = diff_bits
    for a in elems:
        new_sum_bits |= (1 << (elem + a))
        new_diff_bits |= (1 << abs(elem - a))
    new_sum_bits |= (1 << (elem + elem))
    return elems + [elem], new_sum_bits, new_diff_bits

def _remove_elem(elem, elems, sum_bits, diff_bits):
    """Remove elem from elems and rebuild bits."""
    new_elems = [e for e in elems if e != elem]
    return new_elems, _rebuild_bits(new_elems)

def _greedy_build(order, other_diff_bits, deadline):
    """Greedy build following the given order, respecting other_diff_bits."""
    elems = []
    sum_bits = 0
    diff_bits = 0
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            elems, sum_bits, diff_bits = _add_elem(x, elems, sum_bits, diff_bits)
            if len(elems) >= MAX_EACH:
                break
    return elems, sum_bits, diff_bits

def _local_search(A, B, deadline):
    """1-for-1 swap local search to improve |A|+|B|."""
    A = list(A)
    B = list(B)
    A_sum, A_diff = _rebuild_bits(A)
    B_sum, B_diff = _rebuild_bits(B)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add to A
        if len(A) < MAX_EACH:
            for x in range(1, N + 1):
                if x not in A and _can_add(x, A, A_sum, A_diff, B_diff):
                    A, A_sum, A_diff = _add_elem(x, A, A_sum, A_diff)
                    improved = True
                    break
        # Try to add to B
        if len(B) < MAX_EACH and time.time() < deadline:
            for x in range(1, N + 1):
                if x not in B and _can_add(x, B, B_sum, B_diff, A_diff):
                    B, B_sum, B_diff = _add_elem(x, B, B_sum, B_diff)
                    improved = True
                    break
        # Try 1-for-1 swaps in A
        if len(A) > 0 and time.time() < deadline:
            for i in range(len(A)):
                for x in range(1, N + 1):
                    if x in A:
                        continue
                    new_A = A[:i] + A[i+1:]
                    new_A_sum, new_A_diff = _rebuild_bits(new_A)
                    if _can_add(x, new_A, new_A_sum, new_A_diff, B_diff):
                        A = new_A + [x]
                        A_sum, A_diff = _rebuild_bits(A)
                        improved = True
                        break
                if improved or time.time() > deadline:
                    break
        # Try 1-for-1 swaps in B
        if len(B) > 0 and time.time() < deadline:
            for i in range(len(B)):
                for x in range(1, N + 1):
                    if x in B:
                        continue
                    new_B = B[:i] + B[i+1:]
                    new_B_sum, new_B_diff = _rebuild_bits(new_B)
                    if _can_add(x, new_B, new_B_sum, new_B_diff, A_diff):
                        B = new_B + [x]
                        B_sum, B_diff = _rebuild_bits(B)
                        improved = True
                        break
                if improved or time.time() > deadline:
                    break
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_A, best_B = [], []
    best_score = 0
    
    # Multiple random restarts
    while time.time() < deadline:
        # Random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Build A first
        A, A_sum, A_diff = _greedy_build(order, 0, deadline)
        if not A:
            A = [1]
            A_sum, A_diff = _rebuild_bits(A)
        
        # Build B with A's differences as constraint
        B, B_sum, B_diff = _greedy_build(order, A_diff, deadline)
        if not B:
            B = [2]
            B_sum, B_diff = _rebuild_bits(B)
        
        # Local search
        A, B = _local_search(A, B, deadline)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
        
        if best_score >= 2 * MAX_EACH:
            break
    
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
