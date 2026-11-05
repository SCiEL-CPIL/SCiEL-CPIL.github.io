# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)

    # For each t-subset, track which blocks cover it
    # covered_by: list of lists of block indices for each t-subset
    covered_by = [[] for _ in range(n_t_subsets)]
    # Map from tuple to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    blocks = []
    block_t_sets = []  # list of lists of t-subset tuples in each block

    # Precompute for each point x and each (t-1)-subset of other points,
    # which t-subsets are formed
    # Actually, let's use a different approach:
    # For a block B, the t-subsets it covers are all combinations of t from B.

    def get_t_subsets_of_block(block):
        return list(itertools.combinations(block, t))

    def is_covered(t_subset):
        return len(covered_by[t_to_idx[t_subset]]) > 0

    def uncovered_count():
        # We'll maintain this incrementally
        pass

    uncovered_count = n_t

    # Greedy with lookahead: pick the block that maximizes newly covered t-subsets
    # For efficiency, we'll use a heuristic: start from an uncovered t-subset,
    # then greedily add points that cover the most uncovered t-subsets

    # Precompute for each point x, and each (t-1)-subset S, the t-subset S ∪ {x}
    # This is used to quickly compute gain when adding x to a partial block

    # For a partial block P (size < k), adding point x covers all t-subsets
    # that contain x and t-1 points from P, and are not yet covered.

    # Let's precompute: for each x in 0..v-1, for each (t-1)-subset S of points != x,
    # the t-subset S ∪ {x}
    # But that's a lot. Instead, let's compute gain on the fly but efficiently.

    # Key insight: for a partial block P of size m, adding x covers
    # C(m, t-1) t-subsets: all (t-1)-subsets of P combined with x.
    # We need to count how many of these are uncovered.

    # Let's maintain for each t-subset its coverage count.
    # When we add a block, we increment coverage for all its t-subsets.
    # When counting gain for adding x to P, we check each (t-1)-subset of P
    # and see if the resulting t-subset is uncovered.

    # To speed this up, let's precompute for each (t-1)-subset S, 
    # the list of points x such that S ∪ {x} is a t-subset (i.e., x not in S).
    # Actually, for a given (t-1)-subset S, the t-subsets containing S are
    # {S ∪ {x} : x not in S}. There are v - (t-1) such t-subsets.

    # Let's precompute: for each (t-1)-subset S, a list of (x, t_subset_idx) pairs
    # where t_subset = S ∪ {x} and x not in S.
    # But S is a tuple, so we can use a dict.

    # Actually, let's just compute gain directly. For a partial block P of size m,
    # and candidate point x not in P, the gain is:
    # sum over all (t-1)-subsets S of P: 1 if S ∪ {x} is uncovered
    # This is C(m, t-1) checks. For m up to k-1=6, t-1=3, C(6,3)=20. That's fine.

    # Let's implement this.

    # We also want to consider: after building a block, should we try multiple
    # starting t-subsets and pick the best block? Yes, let's try a few.

    # Strategy:
    # 1. Find some uncovered t-subsets (e.g., the first few, or random ones).
    # 2. For each, build a greedy block starting from that t-subset.
    # 3. Pick the block with maximum gain (newly covered t-subsets).
    # 4. Add it, update coverage.
    # 5. Repeat until all covered or time runs out.

    # To find uncovered t-subsets, we can maintain a set of uncovered indices.
    uncovered_set = set(range(n_t_subsets))

    def build_block_from_seed(seed_t_subset, max_time=None):
        """Build a block greedily starting from the given t-subset."""
        block = list(seed_t_subset)
        block.sort()
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many uncovered t-subsets are covered by adding x
                gain = 0
                # All (t-1)-subsets of block
                for rest in itertools.combinations(block, t - 1):
                    t_sub = tuple(sorted(rest + (x,)))
                    idx = t_to_idx.get(t_sub)
                    if idx is not None and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                # No point adds value; just add the smallest available
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
            block.append(best_x)
        
        block.sort()
        return block

    def compute_block_gain(block):
        """Compute how many currently uncovered t-subsets this block covers."""
        gain = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_to_idx[t_sub]
            if idx in uncovered_set:
                gain += 1
        return gain

    def apply_block(block):
        """Add block to the covering and update coverage."""
        nonlocal uncovered_count
        blocks.append(block)
        t_subs = list(itertools.combinations(block, t))
        block_t_sets.append(t_subs)
        newly_covered = 0
        for t_sub in t_subs:
            idx = t_to_idx[t_sub]
            if idx in uncovered_set:
                uncovered_set.remove(idx)
                newly_covered += 1
        uncovered_count -= newly_covered

    # Main loop
    while uncovered_set:
        if time.time() > deadline - 2.0:
            break

        # Find some uncovered t-subsets to use as seeds
        # Take the first few uncovered ones (they're stored as indices)
        # Let's take up to 10 random uncovered t-subsets for diversity
        uncov_list = list(uncovered_set)
        if len(uncov_list) <= 10:
            seeds = uncov_list
        else:
            seeds = random.sample(uncov_list, 10)

        best_block = None
        best_gain = -1

        for seed_idx in seeds:
            seed_t_sub = all_t_subsets[seed_idx]
            block = build_block_from_seed(seed_t_sub)
            gain = compute_block_gain(block)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None or best_gain <= 0:
            # Fallback: just take the first uncovered t-subset and build a block
            seed_idx = next(iter(uncovered_set))
            seed_t_sub = all_t_subsets[seed_idx]
            best_block = build_block_from_seed(seed_t_sub)
            best_gain = compute_block_gain(best_block)

        apply_block(best_block)

    # Convert blocks to lists of ints (they're already tuples/lists of ints)
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
