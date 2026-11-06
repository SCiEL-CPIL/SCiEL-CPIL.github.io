# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A|+|B|.
    """
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s hard kill
    
    N = 256
    MAX_SIZE = 64

    def _can_add_to_set(x, elements, used_sums, diff_mask, other_diff_mask):
        """Check if x can be added to elements. Returns True if valid."""
        for e in elements:
            s = e + x
            if s in used_sums:
                return False
            d = abs(e - x)
            if (1 << d) & diff_mask:
                return False
            if (1 << d) & other_diff_mask:
                return False
        if (2 * x) in used_sums:
            return False
        return True

    def _add_to_set(x, elements, used_sums, diff_mask, other_diff_mask):
        """Add x to set, returning new state."""
        new_sums = set(used_sums)
        new_diff_mask = diff_mask
        for e in elements:
            new_sums.add(e + x)
            new_diff_mask |= (1 << abs(e - x))
        new_sums.add(2 * x)
        return elements + [x], new_sums, new_diff_mask

    def _greedy_construct(order_a, order_b):
        """
        Greedy construction with given candidate orders.
        Returns (A, B, total_size).
        """
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diff_mask_a = 0
        diff_mask_b = 0
        
        for x in order_a:
            if len(A) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            if _can_add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b):
                A, sums_a, diff_mask_a = _add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b)
        
        for x in order_b:
            if len(B) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            if _can_add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a):
                B, sums_b, diff_mask_b = _add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a)
        
        return A, B, len(A) + len(B)

    def _greedy_interleave(order_a, order_b):
        """
        Interleaved greedy: alternate between adding to A and B.
        """
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diff_mask_a = 0
        diff_mask_b = 0
        
        ia, ib = 0, 0
        while (ia < len(order_a) or ib < len(order_b)) and \
              len(A) < MAX_SIZE and len(B) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add to A
            added_a = False
            while ia < len(order_a) and len(A) < MAX_SIZE:
                x = order_a[ia]
                ia += 1
                if _can_add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b):
                    A, sums_a, diff_mask_a = _add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b)
                    added_a = True
                    break
            if not added_a and ia >= len(order_a):
                # A exhausted, fill B
                while ib < len(order_b) and len(B) < MAX_SIZE:
                    x = order_b[ib]
                    ib += 1
                    if _can_add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a):
                        B, sums_b, diff_mask_b = _add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a)
                break
            
            # Try to add to B
            added_b = False
            while ib < len(order_b) and len(B) < MAX_SIZE:
                x = order_b[ib]
                ib += 1
                if _can_add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a):
                    B, sums_b, diff_mask_b = _add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a)
                    added_b = True
                    break
            if not added_b and ib >= len(order_b):
                # B exhausted, fill A
                while ia < len(order_a) and len(A) < MAX_SIZE:
                    x = order_a[ia]
                    ia += 1
                    if _can_add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b):
                        A, sums_a, diff_mask_a = _add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b)
                break
        
        return A, B, len(A) + len(B)

    def _rebuild_state(elements):
        """Rebuild sums and diff_mask for a set."""
        sums = set()
        diff_mask = 0
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sums.add(elements[i] + elements[j])
            for j in range(i + 1, len(elements)):
                diff_mask |= (1 << abs(elements[i] - elements[j]))
        return sums, diff_mask

    def _local_search(A, B):
        """
        Local search: try insertions and swap-based moves.
        """
        A = list(A)
        B = list(B)
        
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            sums_a, diff_mask_a = _rebuild_state(A)
            sums_b, diff_mask_b = _rebuild_state(B)
            
            # Try adding to A
            if len(A) < MAX_SIZE:
                for x in range(1, N + 1):
                    if x in A:
                        continue
                    if _can_add_to_set(x, A, sums_a, diff_mask_a, diff_mask_b):
                        A.append(x)
                        improved = True
                        break
            
            if not improved and len(B) < MAX_SIZE:
                # Try adding to B
                for x in range(1, N + 1):
                    if x in B:
                        continue
                    if _can_add_to_set(x, B, sums_b, diff_mask_b, diff_mask_a):
                        B.append(x)
                        improved = True
                        break
            
            if not improved:
                # Try swap moves: remove one element from A, try to add two
                found_swap = False
                for i in range(len(A)):
                    if time.time() - start_time > DEADLINE:
                        break
                    orig = A[i]
                    A_temp = A[:i] + A[i+1:]
                    sums_a_temp, diff_mask_a_temp = _rebuild_state(A_temp)
                    
                    # Try to add two new elements to A_temp
                    added = 0
                    for x in range(1, N + 1):
                        if x in A_temp:
                            continue
                        if _can_add_to_set(x, A_temp, sums_a_temp, diff_mask_a_temp, diff_mask_b):
                            A_temp, sums_a_temp, diff_mask_a_temp = _add_to_set(x, A_temp, sums_a_temp, diff_mask_a_temp, diff_mask_b)
                            added += 1
                            if added >= 2:
                                break
                    
                    if added >= 2:
                        A = A_temp
                        improved = True
                        found_swap = True
                        break
                
                if not found_swap:
                    # Try swap moves on B
                    for i in range(len(B)):
                        if time.time() - start_time > DEADLINE:
                            break
                        orig = B[i]
                        B_temp = B[:i] + B[i+1:]
                        sums_b_temp, diff_mask_b_temp = _rebuild_state(B_temp)
                        
                        added = 0
                        for x in range(1, N + 1):
                            if x in B_temp:
                                continue
                            if _can_add_to_set(x, B_temp, sums_b_temp, diff_mask_b_temp, diff_mask_a):
                                B_temp, sums_b_temp, diff_mask_b_temp = _add_to_set(x, B_temp, sums_b_temp, diff_mask_b_temp, diff_mask_a)
                                added += 1
                                if added >= 2:
                                    break
                        
                        if added >= 2:
                            B = B_temp
                            improved = True
                            break
            
            if not improved:
                break
        
        return A, B, len(A) + len(B)

    # Main search loop
    best_A = [1]
    best_B = [2]
    best_total = 2
    
    # Deterministic greedy pass
    order_a = list(range(1, N + 1))
    order_b = list(range(1, N + 1))
    A, B, total = _greedy_construct(order_a, order_b)
    if total > best_total:
        best_A, best_B, best_total = A, B, total
    
    # Interleaved greedy
    A, B, total = _greedy_interleave(order_a, order_b)
    if total > best_total:
        best_A, best_B, best_total = A, B, total
    
    # Random restarts
    random.seed(42)
    for restart in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        random.shuffle(order_a)
        random.shuffle(order_b)
        
        # Try both sequential and interleaved
        A1, B1, total1 = _greedy_construct(order_a, order_b)
        A2, B2, total2 = _greedy_interleave(order_a, order_b)
        
        A, B, total = (A1, B1, total1) if total1 >= total2 else (A2, B2, total2)
        
        if total > best_total:
            A_ls, B_ls, total_ls = _local_search(A, B)
            if total_ls > best_total:
                best_A, best_B, best_total = A_ls, B_ls, total_ls
            else:
                best_A, best_B, best_total = A, B, total
    
    # Final local search on best solution
    A_final, B_final, total_final = _local_search(best_A, best_B)
    if total_final > best_total:
        best_A, best_B, best_total = A_final, B_final, total_final
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
