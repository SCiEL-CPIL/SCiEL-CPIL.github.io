import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..128
    
    def to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]
    
    def is_valid(mask):
        # Check no triple a<b<c in set with a+b, a+c, b+c all in set
        # For efficiency, iterate over pairs and check
        elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                ab = a + b
                if ab > N or not ((mask >> ab) & 1):
                    continue
                for k in range(j + 1, n):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac <= N and bc <= N and ((mask >> ac) & 1) and ((mask >> bc) & 1):
                        return False
        return True
    
    # Greedy construction: start empty, add elements in a smart order
    # Try adding from high to low first (like seed), then try to add lower ones
    def greedy_construct():
        mask = 0
        # First, add high elements (they're "safe" as sums go out of range)
        for x in range(N, 0, -1):
            # Check if adding x creates a conflict
            # New conflicts: for pairs (a,b) in current set, if x+a in set and x+b in set and a+b in set
            # More precisely: for all a,b in current set, if {x,a,b} subset of new set and 
            # x+a in new set and x+b in new set and a+b in new set -> conflict
            # Since we're adding x, the new set has x plus current.
            # A conflict triple must include x (otherwise it was already a conflict).
            # So check: for all a,b in current set, a != b:
            #   if x+a <= N and x+b <= N and a+b <= N:
            #     if (new_mask >> (x+a)) & 1 and (new_mask >> (x+b)) & 1 and (new_mask >> (a+b)) & 1:
            #       conflict
            conflict = False
            for a in range(1, N + 1):
                if not ((mask >> a) & 1):
                    continue
                for b in range(a + 1, N + 1):
                    if not ((mask >> b) & 1):
                        continue
                    xa = x + a
                    xb = x + b
                    ab = a + b
                    if xa > N or xb > N or ab > N:
                        continue
                    if ((mask >> xa) & 1) and ((mask >> xb) & 1) and ((mask >> ab) & 1):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                mask |= (1 << x)
        return mask
    
    # Local search to improve
    def local_search(mask, time_limit):
        best_mask = mask
        best_size = bin(mask).count('1')
        while time.time() < time_limit:
            improved = False
            # Try adding each missing element
            for x in range(1, N + 1):
                if (mask >> x) & 1:
                    continue
                new_mask = mask | (1 << x)
                # Check validity of new_mask
                if is_valid(new_mask):
                    mask = new_mask
                    best_mask = mask
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try removing one and adding another (swap)
            found_swap = False
            for x in range(1, N + 1):
                if not ((mask >> x) & 1):
                    continue
                for y in range(1, N + 1):
                    if (mask >> y) & 1 or y == x:
                        continue
                    new_mask = (mask & ~(1 << x)) | (1 << y)
                    if is_valid(new_mask):
                        mask = new_mask
                        best_mask = mask
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap and not improved:
                break
        return best_mask, best_size
    
    # Start with greedy
    mask = greedy_construct()
    size = bin(mask).count('1')
    
    # If greedy is worse than seed, use seed
    seed_mask = 0
    for i in range(65, 129):
        seed_mask |= (1 << i)
    if size < 64:
        mask = seed_mask
        size = 64
    
    # Local search with time budget
    mask, size = local_search(mask, deadline - 2)
    
    # Ensure we have at least the seed
    if size < 64:
        mask = seed_mask
        size = 64
    
    result = to_set(mask)
    # Convert to plain Python ints
    return [int(x) for x in result]

