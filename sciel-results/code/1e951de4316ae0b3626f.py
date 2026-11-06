def construct():
    import time
    from math import gcd, lcm
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 512
    # Build list of numbers 1..512
    nums = list(range(1, n + 1))
    
    # Precompute lcm compatibility
    # compat[i][j] = True if lcm(nums[i], nums[j]) <= 512
    # This is a 512x512 matrix
    
    # First, let's try a greedy approach with multiple orderings
    # Then try local search / branch and bound if time permits
    
    def check_pair(a, b):
        return lcm(a, b) <= n
    
    # Greedy: sort by number of incompatible partners (ascending), add if compatible with all selected
    # Try multiple strategies
    
    best_set = None
    best_size = 0
    
    # Strategy 1: Greedy by ascending value
    selected = []
    for x in nums:
        if all(check_pair(x, y) for y in selected):
            selected.append(x)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]
    
    # Strategy 2: Greedy by descending value
    selected = []
    for x in reversed(nums):
        if all(check_pair(x, y) for y in selected):
            selected.append(x)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]
    
    # Strategy 3: Start with 1, add numbers in order of increasing "conflict"
    # Conflict = number of y in [1,512] such that lcm(x,y) > 512
    conflicts = {}
    for x in nums:
        c = 0
        for y in nums:
            if x != y and lcm(x, y) > n:
                c += 1
        conflicts[x] = c
    
    # Sort by conflict count ascending
    sorted_by_conflict = sorted(nums, key=lambda x: conflicts[x])
    selected = []
    for x in sorted_by_conflict:
        if all(check_pair(x, y) for y in selected):
            selected.append(x)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]
    
    # Strategy 4: Local search - try to swap elements to improve
    # Start from best_set, try removing one and adding another
    if best_set:
        current = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_list = list(current)
            for i, rem in enumerate(current_list):
                # Try removing rem and adding some other number
                remaining = current - {rem}
                # Find candidates that are compatible with all in remaining
                candidates = []
                for x in nums:
                    if x not in remaining and x != rem:
                        if all(check_pair(x, y) for y in remaining):
                            candidates.append(x)
                if len(candidates) > 0:
                    # Adding any candidate keeps same size, so no improvement in size
                    # But maybe we can add TWO by removing one? No, we remove one add one.
                    pass
            # Try: can we add any number to current?
            for x in nums:
                if x not in current:
                    if all(check_pair(x, y) for y in current):
                        current.add(x)
                        improved = True
                        break
    
    best_set = list(current) if 'current' in dir() else best_set
    
    # Strategy 5: More systematic - build from scratch with backtracking on odd parts
    # Group numbers by odd part
    from collections import defaultdict
    odd_groups = defaultdict(list)
    for x in nums:
        m = x
        while m % 2 == 0:
            m //= 2
        odd_groups[m].append(x)
    
    # For each odd part m, the numbers are m * 2^k for k=0,1,...,floor(log2(512/m))
    # We need to choose which odd parts to include, and for each, which powers of 2
    
    # Let's try: include all numbers whose odd part is from a compatible set of odd parts
    # Two odd parts m1, m2 are "compatible" if there exist k1, k2 such that
    # 2^k1 * m1 <= 512, 2^k2 * m2 <= 512, and 2^max(k1,k2) * lcm(m1,m2) <= 512
    
    # Actually, for the best case, we want to include as many numbers as possible.
    # For a fixed set S of odd parts, for each m in S, we can include 2^k * m 
    # for all k such that 2^k * m <= 512 AND for all m' in S, 2^max(k, k') * lcm(m, m') <= 512
    # where k' is the max power we use for m'.
    
    # This is getting complex. Let me try a different approach:
    # Just do a more careful greedy with random restarts
    
    import random
    random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        # Random ordering with bias toward low-conflict numbers
        order = nums[:]
        random.shuffle(order)
        # Sort by a random-weighted conflict score
        weights = {x: random.random() * 0.1 + conflicts[x] / 512 for x in nums}
        order.sort(key=lambda x: weights[x])
        
        selected = []
        for x in order:
            if all(check_pair(x, y) for y in selected):
                selected.append(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    return best_set
