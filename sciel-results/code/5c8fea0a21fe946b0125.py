import random
import time

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute for each vertex: bitmask of all neighbors
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 110.0  # Safety margin
    rnd = random.Random(20260715)
    best_path = [0]
    best_len = 1

    def greedy(priority, start=0):
        """
        Construct a snake using a fixed bit priority order.
        Returns the path.
        """
        path = [start]
        used_mask = 1 << start
        cur = start
        while True:
            nxt = None
            # Try bits in priority order
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                # Check if v is adjacent to any previous vertex (except cur)
                # NEIGHBOR_MASK[v] & used_mask should only contain cur
                # If it contains anything else, v is adjacent to a non-consecutive vertex.
                if (NEIGHBOR_MASK[v] & used_mask) != (1 << cur):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        return path

    # Phase 1: Extensive Randomized Greedy
    # We try many different bit permutations and starting vertices.
    
    # Initial best
    p = greedy(list(range(N)))
    if len(p) > best_len:
        best_len = len(p)
        best_path = p

    # Try a few fixed "smart" orders
    smart_orders = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [0, 2, 4, 6, 8, 1, 3, 5, 7],
        [1, 3, 5, 7, 0, 2, 4, 6, 8],
    ]
    for order in smart_orders:
        if time.time() > deadline:
            break
        for start in range(0, SIZE, 100): # Sample starts
            if time.time() > deadline:
                break
            p = greedy(order, start)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    # Random restarts
    while time.time() < deadline - 5.0:
        priority = list(range(N))
        rnd.shuffle(priority)
        start = rnd.randint(0, SIZE - 1)
        
        p = greedy(priority, start)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    return best_path
