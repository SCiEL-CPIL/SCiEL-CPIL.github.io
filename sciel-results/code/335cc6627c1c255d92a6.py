import time

def construct():
    deadline = time.time() + 55  # Safety margin
    
    n = 12
    maximum = 1048576
    
    def get_subset_sums(arr):
        sums = {0}
        for x in arr:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums |= new_sums
        return sums

    def is_valid(arr):
        """Check if all subset sums are distinct."""
        sums = get_subset_sums(arr)
        return len(sums) == (1 << len(arr))

    def greedy_construct():
        """Greedy construction: pick smallest valid number at each step."""
        arr = []
        for i in range(n):
            # Start from 1, find smallest valid addition
            v = 1
            while v <= maximum:
                candidate = arr + [v]
                if is_valid(candidate):
                    arr.append(v)
                    break
                v += 1
            else:
                # Fallback to power of 2
                arr.append(1 << len(arr))
        return arr

    def try_reduce(arr, idx):
        """Try to reduce arr[idx] to the smallest possible value."""
        current = arr[idx]
        others = arr[:idx] + arr[idx+1:]
        other_sums = get_subset_sums(others)
        used = set(others)
        
        # Compute the difference set of other_sums for fast collision checking
        # v causes a collision if v is in (other_sums - other_sums)
        diff_set = set()
        other_sums_list = list(other_sums)
        for i in range(len(other_sums_list)):
            for j in range(len(other_sums_list)):
                diff_set.add(other_sums_list[i] - other_sums_list[j])
        
        # We want the smallest v < current such that:
        # 1. v is not in used (distinctness)
        # 2. v is not in diff_set (no subset sum collision)
        
        for v in range(1, current):
            if v in used:
                continue
            if v in diff_set:
                continue
            return v
        return None

    # Start with greedy construction
    arr = greedy_construct()
    
    # Local search loop to further reduce
    indices = list(range(n))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in indices:
            if time.time() > deadline:
                break
            new_val = try_reduce(arr, idx)
            if new_val is not None and new_val < arr[idx]:
                arr[idx] = new_val
                improved = True
        # Sort to maintain consistent order for next iteration
        arr.sort()
        indices = list(range(n))
    
    return arr
