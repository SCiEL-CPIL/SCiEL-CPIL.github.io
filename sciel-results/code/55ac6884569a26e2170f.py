import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # sum_to_card maps sum -> cardinality
    # If a sum is achievable, all subsets achieving it must have this cardinality.
    sum_to_card = {0: 0}
    result = []
    
    # Iterate from largest to smallest
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
            
        new_sum_to_card = dict(sum_to_card)
        ok = True
        
        # We need to check all existing sums
        # To avoid modification during iteration issues, we iterate over a snapshot
        # But we are writing to new_sum_to_card, so it's safe.
        # However, if we add a sum to new_sum_to_card, and later in the same x-loop
        # we encounter another s' that produces the same s_new, we must ensure consistency.
        
        for s, k in sum_to_card.items():
            s_new = s + x
            k_new = k + 1
            
            if s_new in new_sum_to_card:
                if new_sum_to_card[s_new] != k_new:
                    ok = False
                    break
            else:
                new_sum_to_card[s_new] = k_new
        
        if ok:
            sum_to_card = new_sum_to_card
            result.append(x)
            
    return list(result)
