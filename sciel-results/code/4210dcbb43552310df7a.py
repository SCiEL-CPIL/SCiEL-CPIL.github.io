import bisect
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2
    d1 = denominators[:mid]
    d2 = denominators[mid:]
    
    def get_valid_subsets(d_list):
        # Returns (list_of_(sum, mask), items_list)
        # items_list is sorted by value descending, containing (Fraction, orig_index_in_d_list)
        
        items = [(Fraction(1, d), i) for i, d in enumerate(d_list)]
        items.sort(key=lambda x: x[0], reverse=True)
        
        valid = []
        n_items = len(items)
        
        # Use iterative DFS to avoid recursion overhead?
        # Stack: (idx, current_sum, current_mask)
        stack = [(0, Fraction(0), 0)]
        
        while stack:
            idx, current_sum, current_mask = stack.pop()
            
            if idx == n_items:
                valid.append((current_sum, current_mask))
                continue
            
            # Option 1: Skip
            # We push skip first so that take is processed first? Order doesn't matter for finding best.
            stack.append((idx + 1, current_sum, current_mask))
            
            # Option 2: Take
            new_sum = current_sum + items[idx][0]
            if new_sum < 1:
                stack.append((idx + 1, new_sum, current_mask | (1 << idx)))
                
        return valid, items

    subs1, items1 = get_valid_subsets(d1)
    subs2, items2 = get_valid_subsets(d2)
    
    # Sort subs2 by sum
    subs2.sort(key=lambda x: x[0])
    sums2 = [s for s, m in subs2]
    
    best_deficit = Fraction(1)
    best_mask1 = 0
    best_mask2 = 0
    
    for s1, m1 in subs1:
        target = 1 - s1
        # Find largest s2 < target
        idx = bisect.bisect_left(sums2, target)
        
        if idx > 0:
            s2 = sums2[idx - 1]
            m2 = subs2[idx - 1][1]
            
            total_sum = s1 + s2
            deficit = 1 - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = m1
                best_mask2 = m2
                
    result_indices = []
    
    for i in range(len(items1)):
        if best_mask1 & (1 << i):
            idx_in_d1 = items1[i][1]
            result_indices.append(idx_in_d1)
            
    for i in range(len(items2)):
        if best_mask2 & (1 << i):
            idx_in_d2 = items2[i][1]
            result_indices.append(mid + idx_in_d2)
            
    if not result_indices:
        result_indices = [0]
        
    return result_indices
