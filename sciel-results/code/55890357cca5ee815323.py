"""Two Sidon sets with disjoint difference sets - Optimized Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin before 60s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # State representation:
    # sets: [list_A, list_B]
    # sum_bits: [int_A, int_B] where bit s is 1 if sum s is used
    # diff_bits: [int_A, int_B] where bit d is 1 if diff d is used
    
    def check_add(elems, sum_bits, diff_bits, other_diff_bits, elem):
        # Check if elem can be added
        # 1. Check sum elem+elem
        s2 = elem + elem
        if (sum_bits >> s2) & 1:
            return False
        
        # 2. Check sums elem+a and diffs |elem-a|
        # Optimization: Iterate through existing elements
        for a in elems:
            s = elem + a
            if (sum_bits >> s) & 1:
                return False
            d = abs(elem - a)
            if (diff_bits >> d) & 1:
                return False
            if (other_diff_bits >> d) & 1:
                return False
        return True

    def add_elem(state, side, elem):
        elems = state['sets'][side]
        sum_bits = state['sum_bits'][side]
        diff_bits = state['diff_bits'][side]
        other_diff = state['diff_bits'][1 - side]
        
        if not check_add(elems, sum_bits, diff_bits, other_diff, elem):
            return False
        
        # Update sum bits
        s2 = elem + elem
        sum_bits |= (1 << s2)
        for a in elems:
            sum_bits |= (1 << (elem + a))
        
        # Update diff bits
        for a in elems:
            d = abs(elem - a)
            diff_bits |= (1 << d)
            
        state['sets'][side].append(elem)
        state['sum_bits'][side] = sum_bits
        state['diff_bits'][side] = diff_bits
        return True

    def remove_elem(state, side, elem):
        elems = state['sets'][side]
        if elem not in elems:
            return
        elems.remove(elem)
        
        # Rebuild bitsets from scratch (O(k^2), k <= 64, so ~4000 ops, fast enough)
        new_sum = 0
        new_diff = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                a = elems[i]
                b = elems[j]
                new_sum |= (1 << (a + b))
                if i != j:
                    new_diff |= (1 << abs(a - b))
        
        state['sum_bits'][side] = new_sum
        state['diff_bits'][side] = new_diff

    def local_search(state, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # 1. Try to add any unused element to either set
            all_used = set(state['sets'][0]) | set(state['sets'][1])
            candidates = [x for x in range(1, N + 1) if x not in all_used]
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > max_time:
                    return state
                # Try adding to set 0
                if len(state['sets'][0]) < MAX_EACH:
                    if add_elem(state, 0, c):
                        improved = True
                        break
                # Try adding to set 1
                if len(state['sets'][1]) < MAX_EACH:
                    if add_elem(state, 1, c):
                        improved = True
                        break
            if improved:
                continue

            # 2. Try to move elements between sets
            # Move from 0 to 1
            elems0 = list(state['sets'][0])
            random.shuffle(elems0)
            for e in elems0:
                if time.time() > max_time:
                    return state
                if len(state['sets'][1]) >= MAX_EACH:
                    continue
                # Check if e can be added to set 1
                if check_add(state['sets'][1], state['sum_bits'][1], state['diff_bits'][1], state['diff_bits'][0], e):
                    # Remove from 0, Add to 1
                    remove_elem(state, 0, e)
                    if add_elem(state, 1, e):
                        improved = True
                        break
            if improved:
                continue
            
            # Move from 1 to 0
            elems1 = list(state['sets'][1])
            random.shuffle(elems1)
            for e in elems1:
                if time.time() > max_time:
                    return state
                if len(state['sets'][0]) >= MAX_EACH:
                    continue
                if check_add(state['sets'][0], state['sum_bits'][0], state['diff_bits'][0], state['diff_bits'][1], e):
                    remove_elem(state, 1, e)
                    if add_elem(state, 0, e):
                        improved = True
                        break
            if improved:
                continue

            # 3. Try swaps: Remove e1 from set A, Add e2 to set A (where e2 is unused or from set B)
            # This is complex, so we stick to simpler moves if time is tight, 
            # but let's try a simple 1-for-1 swap within a set or between sets.
            
            # Try swapping an element in set 0 with an unused element
            elems0 = list(state['sets'][0])
            random.shuffle(elems0)
            all_used = set(state['sets'][0]) | set(state['sets'][1])
            unused = [x for x in range(1, N + 1) if x not in all_used]
            
            # To save time, only try a few swaps
            for e0 in elems0[:5]: # Limit swaps to avoid O(N^2) blowup
                if time.time() > max_time:
                    return state
                for u in unused[:10]:
                    # Check if u can replace e0 in set 0
                    # Temporarily remove e0
                    remove_elem(state, 0, e0)
                    if add_elem(state, 0, u):
                        improved = True
                        break
                    else:
                        # Restore e0
                        add_elem(state, 0, e0)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

            # 4. Try removing an element to allow more additions later? 
            # This is risky. Instead, let's try a "kick": remove a random element from a full set, 
            # then try to add 2 elements.
            # But we already did add. 
            # If we are stuck, we might just break and restart.
            break
            
        return state

    def greedy_build(order, max_time):
        state = {
            'sets': [[], []],
            'sum_bits': [0, 0],
            'diff_bits': [0, 0]
        }
        for elem in order:
            if time.time() > max_time:
                break
            # Try adding to set 0 first, then set 1
            if len(state['sets'][0]) < MAX_EACH:
                if add_elem(state, 0, elem):
                    continue
            if len(state['sets'][1]) < MAX_EACH:
                if add_elem(state, 1, elem):
                    continue
        return state

    best_state = None
    best_size = 0
    
    # Multi-start
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy phase
        state = greedy_build(order, deadline - 1.0)
        
        # Local search phase
        state = local_search(state, deadline - 0.5)
        
        size = len(state['sets'][0]) + len(state['sets'][1])
        if size > best_size:
            best_size = size
            best_state = {
                'sets': [list(state['sets'][0]), list(state['sets'][1])],
                'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
                'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
            }
            
        if best_size >= 2 * MAX_EACH:
            break

    if best_state is None:
        # Fallback
        order = list(range(1, N + 1))
        state = greedy_build(order, deadline)
        best_state = {
            'sets': [list(state['sets'][0]), list(state['sets'][1])],
            'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
            'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
        }

    return [best_state['sets'][0], best_state['sets'][1]]
# EVOLVE-BLOCK-END
