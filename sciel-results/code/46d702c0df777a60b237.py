import time
import random

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # 1. Precompute all 4-term arithmetic progressions
    # An AP is (a, a+d, a+2d, a+3d)
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each number to the list of AP indices it belongs to
    # This allows O(deg(x)) check for validity instead of O(N)
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)

    def can_add_element(x, S):
        """
        Check if x can be added to set S without creating a 4-AP.
        Uses precomputed AP indices for speed.
        """
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            # Check if the other 3 elements are already in S
            # If so, adding x completes the AP
            count_in_set = 0
            for elem in ap:
                if elem == x:
                    continue
                if elem in S:
                    count_in_set += 1
                    if count_in_set == 3:
                        return False
            # Optimization: if count is already 3, we can break early? 
            # No, we need to know if ALL 3 are present. 
            # The loop above breaks if count reaches 3? 
            # Wait, if count reaches 3, it means the other 3 are in S. 
            # So yes, if count_in_set becomes 3, return False immediately.
            # My loop above does: if count_in_set == 3: return False.
            # This is correct.
        return True

    def greedy_construct(order):
        """
        Greedy construction: add elements in given order if valid.
        """
        S = set()
        for x in order:
            if can_add_element(x, S):
                S.add(x)
        return S

    def generate_base4_set():
        """
        Generate a set based on numbers with only 0s and 1s in base 4.
        These sets are 3-AP free, hence also 4-AP free.
        """
        base4_set = set()
        for i in range(1, N + 1):
            n = i
            valid = True
            while n > 0:
                if n % 4 > 1:
                    valid = False
                    break
                n //= 4
            if valid:
                base4_set.add(i)
        return base4_set

    # 2. Initial Construction
    # Try several greedy orderings
    best_set = set()
    best_size = 0
    
    # Standard ascending
    s1 = greedy_construct(range(1, N + 1))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # Descending
    s2 = greedy_construct(range(N, 0, -1))
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # Base-4 set (Salem-Spencer)
    s3 = generate_base4_set()
    if len(s3) > best_size:
        best_set = s3
        best_size = len(s3)

    # 3. Local Search Improvement
    # Strategy: Remove one element, then try to add any available elements.
    # If we can add more than 1 element (net gain > 0), keep the change.
    # We use randomized restarts to escape local optima.
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Pick a random element to remove
        if not best_set:
            break
            
        current_set = set(best_set)
        elements = list(current_set)
        rng.shuffle(elements)
        
        # Try removing each element in a random order
        found_improvement = False
        for remove_elem in elements:
            if time.time() > deadline:
                break
                
            # Tentatively remove
            current_set.remove(remove_elem)
            
            # Try to add any element from 1..N not in current_set
            # We iterate through candidates. To be fast, we can just check all.
            # Since N=512, this is small.
            added_any = False
            new_set = set(current_set)
            
            # Optimization: Only check candidates that might be valid.
            # But can_add_element is fast enough.
            for x in range(1, N + 1):
                if x in new_set:
                    continue
                if can_add_element(x, new_set):
                    new_set.add(x)
                    added_any = True
            
            # If we added anything, we have a new set.
            # Check if it's better (larger) than best_set
            if len(new_set) > best_size:
                best_set = new_set
                best_size = len(new_set)
                found_improvement = True
                # Break to restart with the new best set immediately
                break
        
        # If no improvement found in this pass, we might be stuck.
        # We could try a more aggressive move: remove 2, add 3?
        # But single swap is usually sufficient for this N.
        # To avoid infinite loop on no improvement, we rely on the fact that
        # if we shuffle elements, we might find a different removal that works.
        # However, if we tried all removals and none worked, we are at a local optimum.
        # We should break out and maybe try a different seed?
        # For now, if found_improvement is False, we break the while loop 
        # because trying the same set with different random orders of removal
        # yields the same result (if removal i fails, it fails regardless of order).
        # Wait, if removal i fails to allow *any* addition, it's a dead end.
        # If removal i allows addition but not enough to beat best_size, it's not an improvement.
        # If we tried all i, and none improved, we are stuck.
        if not found_improvement:
            break

    # Ensure nonempty
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
