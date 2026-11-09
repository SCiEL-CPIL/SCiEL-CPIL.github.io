import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def has_k4_circulant(n, S_set):
        for s in S_set:
            common = []
            for x in S_set:
                if (x - s) % n in S_set:
                    common.append(x)
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    x = common[i]
                    y = common[j]
                    if (y - x) % n in S_set or (x - y) % n in S_set:
                        return True
        return False

    def has_k6_circulant(n, S_set):
        for s in S_set:
            common = []
            for x in S_set:
                if (x - s) % n in S_set:
                    common.append(x)
            if len(common) < 4:
                continue
            import itertools
            for quad in itertools.combinations(common, 4):
                a,b,c,d = quad
                if ((b-a)%n in S_set or (a-b)%n in S_set) and \
                   ((c-a)%n in S_set or (a-c)%n in S_set) and \
                   ((d-a)%n in S_set or (a-d)%n in S_set) and \
                   ((c-b)%n in S_set or (b-c)%n in S_set) and \
                   ((d-b)%n in S_set or (b-d)%n in S_set) and \
                   ((d-c)%n in S_set or (c-d)%n in S_set):
                    return True
        return False

    def count_k4_circulant(n, S_set):
        count = 0
        for s in S_set:
            common = []
            for x in S_set:
                if (x - s) % n in S_set:
                    common.append(x)
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    x = common[i]
                    y = common[j]
                    if (y - x) % n in S_set or (x - y) % n in S_set:
                        count += 1
        return count // 6 # Each K4 counted 6 times

    def count_k6_circulant(n, S_set):
        # This is expensive, maybe just use boolean for now?
        # Let's just use boolean for the search, and verify at the end.
        return 0

    def get_cost(n, mask):
        half = n // 2
        S = set()
        for i in range(half):
            if (mask >> i) & 1:
                S.add(i + 1)
        
        k4 = count_k4_circulant(n, S)
        
        S_comp = set(range(1, half + 1)) - S
        # Check K6 in comp
        # We don't have a fast count, so we just check existence?
        # If existence is True, cost is high.
        # Let's just use existence for pruning.
        if has_k6_circulant(n, S_comp):
             return 1000 + k4
        return k4

    for n in range(40, 16, -1):
        half = n // 2
        # Local search on mask
        best_mask = (1 << half) - 1 # Start with complete graph? No, that has K4.
        best_mask = random.getrandbits(half)
        best_cost = get_cost(n, best_mask)
        
        for step in range(10000):
            if time.time() > deadline:
                break
            
            # Flip a random bit
            bit_idx = random.randint(0, half - 1)
            new_mask = best_mask ^ (1 << bit_idx)
            new_cost = get_cost(n, new_mask)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_mask = new_mask
                
                if best_cost == 0:
                    # Verify
                    S = set()
                    for i in range(half):
                        if (best_mask >> i) & 1:
                            S.add(i + 1)
                    S_comp = set(range(1, half + 1)) - S
                    if not has_k4_circulant(n, S) and not has_k6_circulant(n, S_comp):
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                diff = (j - i) % n
                                if diff in S or (n - diff) in S:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
        
        # If no valid graph found for this n, continue to n-1
        # But we should keep the best graph found so far?
        # No, we want the max n.
        
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
