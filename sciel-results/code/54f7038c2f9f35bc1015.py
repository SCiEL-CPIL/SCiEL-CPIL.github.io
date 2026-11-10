import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer for 60s limit

    # Fast bitset-based subset sum computation by cardinality
    def compute_sums_bitset(s):
        """Returns list of bitsets, where sums_by_k[k] has bit i set iff sum i is achievable with k elements."""
        n = len(s)
        sums_by_k = [0] * (n + 1)
        sums_by_k[0] = 1  # bit 0 set
        for v in s:
            for k in range(n, 0, -1):
                sums_by_k[k] |= (sums_by_k[k - 1] << v)
        return sums_by_k

    def count_collisions(sums_by_k):
        """Count cross-cardinality sum collisions."""
        n = len(sums_by_k) - 1
        total = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                inter = sums_by_k[i] & sums_by_k[j]
                if inter:
                    total += inter.bit_count()
        return total

    def is_valid(s):
        sums_by_k = compute_sums_bitset(s)
        return count_collisions(sums_by_k) == 0

    # Initial best
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)

    # Strategy 1: Greedy with various orderings
    def greedy(order):
        curr = []
        for v in order:
            if time.time() > deadline - 5:
                break
            new_curr = curr + [v]
            if is_valid(new_curr):
                curr = new_curr
        return curr

    orderings = [
        list(range(256, 0, -1)),  # descending
        list(range(1, 257)),      # ascending
        list(range(1, 257, 2)),   # odds
        list(range(2, 257, 2)),   # evens
        list(range(256, 0, -2)),  # descending evens
        list(range(255, 0, -2)),  # descending odds
    ]

    for order in orderings:
        if time.time() > deadline - 10:
            break
        g = greedy(order)
        if len(g) > best_size:
            best_set = g
            best_size = len(g)

    # Strategy 2: Random large sets + conflict removal
    random.seed(12345)
    for attempt in range(60):
        if time.time() > deadline - 8:
            break
        size = random.randint(20, 45)
        rand_set = random.sample(range(1, 257), size)
        
        # Conflict removal: iteratively remove element with most conflicts
        current = list(rand_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            sums_by_k = compute_sums_bitset(current)
            current_coll = count_collisions(sums_by_k)
            if current_coll == 0:
                break
            
            n = len(current)
            best_idx = -1
            best_coll = current_coll
            for idx in range(n):
                new_set = current[:idx] + current[idx + 1:]
                new_sums = compute_sums_bitset(new_set)
                new_coll = count_collisions(new_sums)
                if new_coll < best_coll:
                    best_coll = new_coll
                    best_idx = idx
            
            if best_idx >= 0 and best_coll < current_coll:
                current = current[:best_idx] + current[best_idx + 1:]
                improved = True
        
        if is_valid(current) and len(current) > best_size:
            best_set = current
            best_size = len(current)

    # Strategy 3: Extend best set by trying to add elements
    def extend(s):
        curr = list(s)
        in_set = set(curr)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for v in range(1, 257):
                if v in in_set:
                    continue
                new_set = curr + [v]
                if is_valid(new_set):
                    curr = new_set
                    in_set.add(v)
                    improved = True
                    break
        return curr

    extended = extend(best_set)
    if len(extended) > best_size:
        best_set = extended
        best_size = len(best_set)

    # Strategy 4: Swap-based local search to find better composition
    def swap_search(s):
        curr = list(s)
        in_set = set(curr)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(len(curr)):
                if time.time() > deadline - 3:
                    break
                removed = curr[idx]
                base = curr[:idx] + curr[idx + 1:]
                for v in range(1, 257):
                    if v in in_set or v == removed:
                        continue
                    new_set = base + [v]
                    if is_valid(new_set):
                        curr = new_set
                        in_set = set(curr)
                        improved = True
                        break
                if improved:
                    break
        return curr

    swapped = swap_search(best_set)
    if len(swapped) > best_size:
        best_set = swapped
        best_size = len(best_set)
    else:
        # Try extending from swapped version
        if time.time() < deadline - 3:
            ext2 = extend(swapped)
            if len(ext2) > best_size:
                best_set = ext2
                best_size = len(best_set)

    return [int(x) for x in best_set]

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Result: {len(result)} elements, took {time.time() - t0:.2f}s")
    print(f"Set: {result}")
