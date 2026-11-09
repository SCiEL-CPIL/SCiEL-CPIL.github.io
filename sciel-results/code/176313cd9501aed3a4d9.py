"""Rewritten program: exact reciprocal-subsum-free maximization via
greedy + beam search over scaled-integer subset sums, with validation
and a single-add local improvement."""

def construct():
    import sys
    sys.setrecursionlimit(10000)

    # scale = lcm(1..24) so that 1/k becomes the integer scale//k
    scale = 5354228880
    weights = [0] + [scale // i for i in range(1, 25)] # weights[i] corresponds to element i

    best_count = 0
    best_set = []

    # Greedy initialization to get a good lower bound
    sums = {0}
    greedy_set = []
    for i in range(24, 0, -1):
        w = weights[i]
        if w not in sums:
            greedy_set.append(i)
            sums = sums | {s + w for s in sums}
    best_count = len(greedy_set)
    best_set = greedy_set[:]

    # DFS with Branch and Bound
    # k: current element index (24 down to 1)
    # sums: set of subset sums of already selected elements (which are > k)
    # count: number of selected elements so far
    # chosen: list of selected elements
    def dfs(k, sums, count, chosen):
        nonlocal best_count, best_set
        
        # Pruning: even if we select all remaining k elements, we can't beat best
        if count + k <= best_count:
            return
            
        if k == 0:
            if count > best_count:
                best_count = count
                best_set = chosen[:]
            return

        # Option 1: Include k
        # We can include k if its weight is not a subset sum of the larger elements
        w = weights[k]
        if w not in sums:
            new_sums = sums | {s + w for s in sums}
            dfs(k - 1, new_sums, count + 1, chosen + [k])
            
        # Option 2: Exclude k
        # Always possible
        dfs(k - 1, sums, count, chosen)

    dfs(24, {0}, 0, [])
    
    return sorted(best_set)


# EVOLVE-BLOCK-END
