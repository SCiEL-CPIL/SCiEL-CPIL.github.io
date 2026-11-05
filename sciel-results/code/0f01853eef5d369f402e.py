import random
import time
from itertools import combinations

def is_valid_fast(s):
    """
    Fast validity check for the entire set.
    For each element a, checks if any 3 distinct others sum to a multiple of a.
    Uses pair-sum + set-lookup for efficiency.
    """
    lst = sorted(s)
    n = len(lst)
    if n < 4:
        return True
    
    for idx_a, a in enumerate(lst):
        others = lst[:idx_a] + lst[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        
        # If a is larger than the max possible sum of 3 others, no violation possible
        max_sum3 = others[-1] + others[-2] + others[-3]
        if a > max_sum3:
            continue
        
        min_sum3 = others[0] + others[1] + others[2]
        if min_sum3 > max_sum3:
            continue
            
        others_set = set(others)
        
        # We need to check if any triple (x,y,z) from others sums to k*a
        # Instead of iterating k, we iterate pairs (x,y) and check if z = k*a - x - y is in others.
        # However, k varies. 
        # A better approach for small n: just check all triples? No, O(n^3) is too slow for n=48.
        # The multiple-iteration approach in the seed was decent. Let's stick to a refined version.
        
        # Refined approach:
        # For each multiple M of a in [min_sum3, max_sum3]:
        #   Check if any pair (x,y) in others sums to M - z for some z?
        #   Actually, check if any triple sums to M.
        #   Standard 3-sum check: for i, j, check if M - others[i] - others[j] is in others.
        
        start_mult = (min_sum3 + a - 1) // a
        end_mult = max_sum3 // a
        
        if start_mult > end_mult:
            continue
            
        for k in range(start_mult, end_mult + 1):
            target = k * a
            found = False
            # Check if any 3 distinct elements in 'others' sum to 'target'
            # Iterate pairs (i, j)
            for i in range(m - 2):
                xi = others[i]
                # Early break: if xi + others[i+1] + others[i+2] > target, further i will be larger
                if xi + others[i+1] + others[i+2] > target:
                    break
                for j in range(i + 1, m - 1):
                    xj = others[j]
                    # Early break: if xi + xj + others[j+1] > target, further j will be larger
                    if xi + xj + others[j+1] > target:
                        break
                    rem = target - xi - xj
                    # rem must be in others, distinct from xi and xj
                    # Since others is sorted, and we picked i < j, rem must be > xj to be distinct and not duplicate i or j (unless values are equal, but they are distinct integers)
                    # Wait, distinct integers means values are unique.
                    # If rem == xi or rem == xj, it's not distinct.
                    # Since rem = target - xi - xj, and xi, xj < rem is not guaranteed.
                    # But if rem is in others_set, we need to ensure it's not xi or xj.
                    # Since xi, xj are distinct and positive, and rem is the third element.
                    # If rem == xi, then 2*xi + xj = target.
                    # If rem == xj, then xi + 2*xj = target.
                    # We can just check if rem is in others_set and rem != xi and rem != xj.
                    if rem in others_set and rem != xi and rem != xj:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def is_valid_incremental(existing_set, new_elem):
    """
    Check if adding new_elem to existing_set maintains validity.
    1. new_elem doesn't divide sum of any 3 existing elements
    2. No existing element divides sum of new_elem + 2 existing elements
    """
    if not existing_set:
        return True
    
    existing_sorted = sorted(existing_set)
    n = len(existing_sorted)
    
    # Check 1: new_elem vs triples from existing_set
    a = new_elem
    if n >= 3:
        max_sum3 = existing_sorted[-1] + existing_sorted[-2] + existing_sorted[-3]
        if a <= max_sum3:
            min_sum3 = existing_sorted[0] + existing_sorted[1] + existing_sorted[2]
            start_mult = (min_sum3 + a - 1) // a
            end_mult = max_sum3 // a
            if start_mult <= end_mult:
                existing_set_s = set(existing_sorted)
                for k in range(start_mult, end_mult + 1):
                    target = k * a
                    found = False
                    for i in range(n - 2):
                        xi = existing_sorted[i]
                        if xi + existing_sorted[i+1] + existing_sorted[i+2] > target:
                            break
                        for j in range(i + 1, n - 1):
                            xj = existing_sorted[j]
                            if xi + xj + existing_sorted[j+1] > target:
                                break
                            rem = target - xi - xj
                            if rem in existing_set_s and rem != xi and rem != xj:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    
    # Check 2: Each existing element vs triples including new_elem
    for idx_a, a in enumerate(existing_sorted):
        # Triples that include new_elem: new_elem + 2 from existing_set (excluding a)
        others_excluding_a = [x for x in existing_sorted if x != a]
        m = len(others_excluding_a)
        if m < 2:
            continue
        
        # We need: a | (new_elem + x + y) for some distinct x, y in others_excluding_a
        # i.e., x + y ≡ -new_elem (mod a)
        
        max_sum = new_elem + others_excluding_a[-1] + others_excluding_a[-2]
        if a > max_sum:
            continue
        
        min_sum = new_elem + others_excluding_a[0] + others_excluding_a[1]
        
        start_mult = (min_sum + a - 1) // a
        end_mult = max_sum // a
        
        if start_mult > end_mult:
            continue
        
        others_set_s = set(others_excluding_a)
        for k in range(start_mult, end_mult + 1):
            target = k * a
            need = target - new_elem
            if need < 0:
                continue
            # Check if any pair in others_excluding_a sums to need
            found = False
            for i in range(m - 1):
                xi = others_excluding_a[i]
                if xi + others_excluding_a[i+1] > need:
                    break
                rem = need - xi
                if rem in others_set_s and rem != xi:
                    found = True
                    break
            if found:
                return False
    
    return True

def greedy_construct(seed_rng=None):
    """
    Greedy construction with priority on larger numbers.
    """
    best_result = []
    
    # Strategy 1: Descending order
    selected = []
    for x in range(192, 0, -1):
        if is_valid_incremental(set(selected), x):
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
    
    # Strategy 2: Descending, but stricter on small numbers
    selected = []
    for x in range(192, 0, -1):
        if x <= 15:
            if len(selected) < 42:
                if is_valid_incremental(set(selected), x):
                    selected.append(x)
        else:
            if is_valid_incremental(set(selected), x):
                selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected
        
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    rng = random.Random(42)
    
    # Start with multiple greedy strategies
    best = greedy_construct()
    best_set = frozenset(best)
    best_size = len(best)
    
    all_numbers = set(range(1, 193))
    
    def local_search(s, rng, max_iters=500):
        """
        Improved hill-climb with better swap strategies.
        """
        s = set(s)
        cur_size = len(s)
        stall_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding elements (prioritize larger numbers as they are "safer")
            candidates = sorted(all_numbers - s, reverse=True)
            rng.shuffle(candidates)
            for x in candidates[:15]:
                if is_valid_incremental(s, x):
                    s.add(x)
                    cur_size += 1
                    improved = True
                    stall_count = 0
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add another
            elems = list(s)
            # Prioritize removing smaller numbers (they are more likely to be divisors)
            elems.sort(key=lambda x: x)
            
            for rem in elems[:8]:
                candidates = sorted(all_numbers - s, reverse=True)
                rng.shuffle(candidates)
                for add in candidates[:12]:
                    if add == rem:
                        continue
                    new_s = (s - {rem}) | {add}
                    if is_valid_fast(new_s):
                        s = new_s
                        improved = True
                        stall_count = 0
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove 2, add 2
            # This helps escape local minima where single swaps don't work
            if rng.random() < 0.5:
                elems = list(s)
                elems.sort(key=lambda x: x)
                rem_pairs = list(combinations(elems[:10], 2))
                rng.shuffle(rem_pairs)
                
                for rem1, rem2 in rem_pairs[:5]:
                    candidates = sorted(all_numbers - s, reverse=True)
                    rng.shuffle(candidates)
                    add_pairs = list(combinations(candidates[:10], 2))
                    for add1, add2 in add_pairs[:5]:
                        new_s = (s - {rem1, rem2}) | {add1, add2}
                        if is_valid_fast(new_s):
                            s = new_s
                            improved = True
                            stall_count = 0
                            break
                    if improved:
                        break
            
            if not improved:
                stall_count += 1
                if stall_count > 40:
                    break
        return s
    
    # Run multiple local search rounds with diverse starting points
    # We run this for a fixed number of rounds or until time runs out
    rounds = 0
    while time.time() < deadline:
        rounds += 1
        
        # Generate diverse starting points
        if rounds == 1:
            start_s = set(best)
        elif rounds == 2:
            start_s = set(greedy_construct())
        else:
            # Random start: take a random subset biased toward larger numbers
            k = rng.randint(35, 45)
            # Weight toward larger numbers to get a valid base faster
            weights = [x for x in range(1, 193)]
            start_s = set(rng.choices(range(1, 193), weights=weights, k=k))
            
            # Filter to valid set by greedily adding (larger first)
            valid_s = set()
            for x in sorted(start_s, reverse=True):
                if is_valid_incremental(valid_s, x):
                    valid_s.add(x)
            start_s = valid_s
        
        # Local search
        result = local_search(start_s, rng, max_iters=300)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = frozenset(result)
            best = list(result)
            
        # If we hit the max possible size (48), we can stop early if we want, 
        # but let's keep searching for safety in case of invalidity bugs, 
        # though the problem says max 48.
        if best_size >= 48:
            break
            
    return list(best_set)
