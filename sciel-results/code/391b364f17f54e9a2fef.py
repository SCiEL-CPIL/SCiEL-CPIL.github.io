"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We want a nonempty subset S of [1,100] with no subset summing to 200.
    # Maximize |S|.
    #
    # Key observation: if we take all numbers from 1 to k, any subset sum up to k(k+1)/2
    # is achievable. But more importantly, we can reason about which sums are possible.
    #
    # A well-known strategy: take all numbers in [1, 100] except those that could
    # form a subset summing to 200. However, checking all subsets is exponential.
    #
    # Better approach: Use the fact that if S contains all of [1, m] for some m,
    # then every sum from 1 to m(m+1)/2 is achievable as a subset sum.
    # If we include [1, 19], sum = 190. Including 20 gives sum 210 > 200, but
    # 20 itself is fine. The issue is whether some subset of the remaining numbers
    # can sum to 200.
    #
    # Let's think differently. The maximum size subset of [1,100] with no subset
    # summing exactly to 200.
    #
    # Consider: if we include all numbers from 1 to 100, the subset sum 200 is
    # trivially achievable (e.g., 1+2+...+something, or 100+100 not allowed, but
    # 100+99+1=200, or 199+1, etc.). So we need to exclude some numbers.
    #
    # Strategy: For each x in [1,100], if there exists a subset T of S\{x} such
    # that sum(T) = 200 - x, then including x is "dangerous". We want to exclude
    # the minimum number of elements to prevent any subset from summing to 200.
    #
    # This is equivalent to: find the maximum-size subset S of [1,100] such that
    # 200 is not in the subset-sum set of S.
    #
    # Greedy approach from the seed: iterate from 100 down to 1, adding x if it
    # doesn't create 200 as a subset sum. This is a standard greedy for subset-sum
    # avoidance.
    #
    # Alternative: Use a bitmask/DP approach. Since sums only go up to 200, we can
    # track reachable sums. Start with the full set [1,100] and remove elements
    # one by one until 200 is no longer reachable. To maximize size, remove the
    # "least useful" elements first.
    #
    # Actually, let's use a different approach: 
    # Take S = [1, 100] \ F where F is a minimum set whose removal breaks all
    # subset sums to 200.
    #
    # A simpler constructive approach: 
    # Note that if all elements in S are > 200, no subset can sum to 200, but we
    # only have up to 100, so that doesn't help.
    #
    # Another approach: partition [1,100] into pairs (x, 200-x) for x < 200-x,
    # i.e., x from 1 to 99, partner 199, 198, ..., 101. Wait, 200-x for x in [1,100]
    # gives 199 down to 100. So pairs: (1,199), (2,198), ..., (99,101), and 100 is
    # self-paired (200-100=100). Also, we can have more than two elements summing
    # to 200.
    #
    # The greedy in the seed actually works well. Let me verify its output size
    # and see if we can do better with a smarter greedy or a different construction.
    #
    # Let me try a different greedy: process from 1 to 100 (ascending), adding x
    # if it doesn't create sum 200.
    #
    # Or: use a randomized restart approach with the DP check.
    #
    # Given the constraint that we need to finish in 60s and this is a small problem,
    # let me try: for each element x from 1 to 100, determine if it MUST be excluded
    # given the others. But that's complex.
    #
    # Let me just use a clean DP-based greedy that tries both ascending and descending
    # orders and picks the better one, plus a random shuffle variant.
    
    import random
    
    def subset_sum_reachable(S, target):
        """Check if target is reachable as a subset sum of S."""
        reachable = {0}
        for x in S:
            if x > target:
                continue
            new = {s + x for s in reachable if s + x <= target}
            reachable |= new
        return target in reachable
    
    def greedy_greedy(order):
        """Greedy: process elements in given order, add if doesn't create sum 200."""
        selected = []
        reachable = {0}
        for x in order:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected
    
    # Try different orders
    best = set()
    
    # Descending order (original)
    desc = list(range(100, 0, -1))
    best = set(greedy_greedy(desc))
    
    # Ascending order
    asc = list(range(1, 101))
    cand = greedy_greedy(asc)
    if len(cand) > len(best):
        best = set(cand)
    
    # Try several random shuffles
    rng = random.Random(42)
    for _ in range(200):
        shuffled = list(range(1, 101))
        rng.shuffle(shuffled)
        cand = greedy_greedy(shuffled)
        if len(cand) > len(best):
            best = set(cand)
    
    # Verify and return
    if len(best) == 0:
        best = {1}
    
    # Final safety check: ensure no subset sums to 200
    def verify(S):
        S = list(S)
        reachable = {0}
        for x in S:
            if x > 200:
                continue
            new = {s + x for s in reachable if s + x <= 200}
            reachable |= new
        return 200 not in reachable
    
    if not verify(best):
        # Fallback: remove elements until valid
        lst = sorted(best, reverse=True)
        while not verify(set(lst)):
            # Remove the largest element
            lst = lst[:-1]
            if len(lst) == 0:
                break
        best = set(lst)
    
    return list(best)
# EVOLVE-BLOCK-END
