"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin before 90s kill


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
    # elem + a for each a in elems
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check difference collisions with other set
    # |elem - a| must not be in other_diff_bits
    for a in elems:
        d = abs(elem - a)
        if d > 0 and (other_diff_bits >> d) & 1:
            return False
    # Check difference collisions within same set
    for a in elems:
        d = abs(elem - a)
        if d > 0 and (diff_bits >> d) & 1:
            return False
    return True


def _add_elem(elem, elems, sum_bits, diff_bits):
    """Add elem to elems, update sum_bits and diff_bits."""
    for a in elems:
        sum_bits |= (1 << (elem + a))
        d = abs(elem - a)
        if d > 0:
            diff_bits |= (1 << d)
    sum_bits |= (1 << (elem + elem))
    elems.append(elem)
    return sum_bits, diff_bits


def _remove_elem(elem, elems, sum_bits, diff_bits):
    """Remove elem from elems, rebuild sum_bits and diff_bits."""
    elems.remove(elem)
    return _rebuild_bits(elems)


def _greedy_build(order, other_diff_bits):
    """Greedy build a Sidon set using the given order, avoiding other_diff_bits."""
    elems = []
    sum_bits = 0
    diff_bits = 0
    for x in order:
        if len(elems) >= MAX_EACH:
            break
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            sum_bits, diff_bits = _add_elem(x, elems, sum_bits, diff_bits)
    return elems, sum_bits, diff_bits


def _swap_improve(A, B, time_budget):
    """Try 1-for-2 swaps to increase total size."""
    A_list = list(A)
    B_list = list(B)
    A_sum, A_diff = _rebuild_bits(A_list)
    B_sum, B_diff = _rebuild_bits(B_list)
    
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        
        # Try adding to A by removing one from A
        for remove_idx in range(len(A_list)):
            if len(A_list) >= MAX_EACH:
                break
            remove_val = A_list[remove_idx]
            A_trial = A_list[:remove_idx] + A_list[remove_idx+1:]
            A_trial_sum, A_trial_diff = _rebuild_bits(A_trial)
            
            # Try adding each candidate from B to A
            for b_val in B_list:
                if b_val in A_trial:
                    continue
                if _can_add(b_val, A_trial, A_trial_sum, A_trial_diff, B_diff):
                    A_new = A_trial + [b_val]
                    A_new_sum, A_new_diff = _add_elem(b_val, list(A_trial), A_trial_sum, A_trial_diff)
                    # Check B still valid without b_val
                    B_trial = [x for x in B_list if x != b_val]
                    B_trial_sum, B_trial_diff = _rebuild_bits(B_trial)
                    # Check A_new differences don't conflict with B_trial
                    if not (A_new_diff & B_trial_diff):
                        A_list = A_new
                        A_sum, A_diff = A_new_sum, A_new_diff
                        B_list = B_trial
                        B_sum, B_diff = B_trial_sum, B_trial_diff
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
        
        # Try adding to B by removing one from B
        for remove_idx in range(len(B_list)):
            if len(B_list) >= MAX_EACH:
                break
            remove_val = B_list[remove_idx]
            B_trial = B_list[:remove_idx] + B_list[remove_idx+1:]
            B_trial_sum, B_trial_diff = _rebuild_bits(B_trial)
            
            for a_val in A_list:
                if a_val in B_trial:
                    continue
                if _can_add(a_val, B_trial, B_trial_sum, B_trial_diff, A_diff):
                    B_new = B_trial + [a_val]
                    B_new_sum, B_new_diff = _add_elem(a_val, list(B_trial), B_trial_sum, B_trial_diff)
                    A_trial = [x for x in A_list if x != a_val]
                    A_trial_sum, A_trial_diff = _rebuild_bits(A_trial)
                    if not (A_trial_diff & B_new_diff):
                        B_list = B_new
                        B_sum, B_diff = B_new_sum, B_new_diff
                        A_list = A_trial
                        A_sum, A_diff = A_trial_sum, A_trial_diff
                        improved = True
                        break
            if improved:
                break
    
    return A_list, B_list


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_A = []
    best_B = []
    best_size = 0
    
    # Multiple random restarts with greedy build
    for restart in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Random order for A
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        
        # Build A first
        A_elems, A_sum_bits, A_diff_bits = _greedy_build(order_A, 0)
        
        # Random order for B, avoiding A's differences
        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        
        B_elems, B_sum_bits, B_diff_bits = _greedy_build(order_B, A_diff_bits)
        
        # Check if valid (differences disjoint)
        if A_diff_bits & B_diff_bits:
            # Try to fix by removing conflicting elements
            # This is a simple repair: remove elements from the smaller set
            if len(A_elems) <= len(B_elems):
                # Remove from A until no conflict
                changed = True
                while changed and A_diff_bits & B_diff_bits:
                    changed = False
                    for i in range(len(A_elems) - 1, -1, -1):
                        test_A = A_elems[:i] + A_elems[i+1:]
                        test_A_sum, test_A_diff = _rebuild_bits(test_A)
                        if not (test_A_diff & B_diff_bits):
                            A_elems = test_A
                            A_sum_bits, A_diff_bits = test_A_sum, test_A_diff
                            changed = True
                            break
            else:
                changed = True
                while changed and A_diff_bits & B_diff_bits:
                    changed = False
                    for i in range(len(B_elems) - 1, -1, -1):
                        test_B = B_elems[:i] + B_elems[i+1:]
                        test_B_sum, test_B_diff = _rebuild_bits(test_B)
                        if not (A_diff_bits & test_B_diff):
                            B_elems = test_B
                            B_sum_bits, B_diff_bits = test_B_sum, test_B_diff
                            changed = True
                            break
        
        total = len(A_elems) + len(B_elems)
        if total > best_size:
            best_size = total
            best_A = list(A_elems)
            best_B = list(B_elems)
        
        # Try swap improvement if we have time
        if time.time() < deadline - 10 and total >= best_size - 2:
            A_swapped, B_swapped = _swap_improve(A_elems, B_elems, deadline - 10)
            total_swapped = len(A_swapped) + len(B_swapped)
            if total_swapped > best_size:
                best_size = total_swapped
                best_A = A_swapped
                best_B = B_swapped
    
    # Final validation
    if not best_A or not best_B:
        # Fallback: simple greedy
        A_elems, A_sum_bits, A_diff_bits = _greedy_build(list(range(1, N + 1)), 0)
        B_elems, B_sum_bits, B_diff_bits = _greedy_build(list(range(1, N + 1)), A_diff_bits)
        best_A = A_elems
        best_B = B_elems
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
