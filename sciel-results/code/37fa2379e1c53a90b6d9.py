import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 12
    
    def subset_sums(seq):
        sums = {0}
        for x in seq:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums
    
    def is_valid(seq):
        return len(subset_sums(seq)) == (1 << len(seq))
    
    # Greedy superincreasing
    seq = [1]
    for i in range(1, n):
        seq.append(sum(seq) + 1)
    
    best = list(seq)
    
    # Aggressive local search
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            orig = best[i]
            # Try binary search for the smallest valid value at position i
            # Actually just try reducing by 1 each time in a loop
            for new_val in range(1, orig):
                if time.time() > deadline:
                    break
                cand = best[:]
                cand[i] = new_val
                if is_valid(cand):
                    best[i] = new_val
                    improved = True
                    break
        if not improved:
            break
    
    if is_valid(best):
        return best
    return [1 << i for i in range(n)]
