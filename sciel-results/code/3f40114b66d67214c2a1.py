import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    best_set = None
    best_size = -1
    
    def run_greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            # Check if adding x creates a sum of 200
            # We need to check if 200 - x is in reachable
            # But reachable contains all subset sums.
            # If 200-x is in reachable, then there is a subset S' such that sum(S') = 200-x.
            # Then S' U {x} sums to 200.
            # So we must NOT add x.
            
            # Optimization:
            # new_sums = {s + x for s in reachable if s + x <= 200}
            # if 200 in new_sums: reject
            # else: accept, reachable |= new_sums
            
            # Faster check:
            # if (200 - x) in reachable:
            #     reject
            # else:
            #     accept
            #     # Update reachable
            #     # We only need to add sums <= 200
            #     # new_sums = {s + x for s in reachable if s + x <= 200}
            #     # reachable |= new_sums
            #     pass
            
            # Wait, is `if (200-x) in reachable` sufficient?
            # Yes, because if 200-x is a subset sum, adding x makes 200.
            # If 200-x is NOT a subset sum, adding x does NOT make 200.
            # (Because any subset summing to 200 involving x must be x + subset_summing_to_200-x).
            
            if (200 - x) in reachable:
                continue # Reject
            
            # Accept x
            selected.append(x)
            
            # Update reachable
            # We need to add all new sums formed by adding x to existing sums
            # But we only care about sums <= 200
            # And specifically, we need to update reachable so future checks work.
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            reachable |= new_sums
            
        return selected

    # Define orders
    orders = []
    
    # 1. Descending (Seed)
    orders.append(list(range(100, 0, -1)))
    
    # 2. Ascending
    orders.append(list(range(1, 101)))
    
    # 3. Center Out (50, 51, 49, 52, 48...)
    center_order = [50]
    for i in range(1, 51):
        if 50 + i <= 100:
            center_order.append(50 + i)
        if 50 - i >= 1:
            center_order.append(50 - i)
    orders.append(center_order)
    
    # 4. Ends In (1, 100, 2, 99...)
    ends_order = []
    for i in range(1, 51):
        ends_order.append(i)
        ends_order.append(101 - i)
    orders.append(ends_order)

    # 5. Random Orders
    # Run as many as possible within time budget
    base_elements = list(range(1, 101))
    
    # Evaluate fixed orders first
    for order in orders:
        if time.time() > deadline:
            break
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # Evaluate random orders
    # Each run is very fast, ~1ms
    # 50s / 1ms = 50,000 runs possible
    # But let's be safe, say 10,000 runs
    while time.time() < deadline:
        random.shuffle(base_elements)
        res = run_greedy(base_elements)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    if best_set is None:
        # Fallback to seed
        best_set = run_greedy(list(range(100, 0, -1)))
        
    return best_set
