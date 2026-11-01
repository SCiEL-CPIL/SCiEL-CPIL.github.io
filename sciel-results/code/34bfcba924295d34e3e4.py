"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter

# Precompute all t-subsets as frozensets for fast membership
def _all_tsubsets(v, t):
    return set(itertools.combinations(range(v), t))


def _greedy_cover(v, k, t, seed=None, time_limit=110.0):
    """
    Randomized greedy covering with redundant block removal.
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    start_time = time.time()
    deadline = start_time + time_limit

    # Represent each point set as a frozenset for hashing
    # Precompute all t-subsets as frozensets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_frozen = [frozenset(sub) for sub in all_tsubs]
    
    # For fast counting, we track which t-subsets are covered by which blocks
    # Use a Counter mapping frozenset -> count of blocks covering it
    coverage = Counter()
    blocks = []  # list of frozensets

    # For efficiency, precompute for each point x, the list of t-subsets containing x
    # and for each (t-1)-subset s, the list of points x such that s ∪ {x} is a t-subset
    # Actually, let's think about this differently.
    
    # A block B covers a t-subset S if S ⊆ B.
    # When we add point x to a partial block P, the new t-subsets covered are
    # {S : S ⊆ P ∪ {x}, |S| = t, S ⊄ P} = {S : S ∩ {x} ≠ ∅, S ⊆ P ∪ {x}, |S|=t}
    # = {S : S \ {x} ⊆ P, |S \ {x}| = t-1} ∪ {S : x ∈ S, S ⊆ P ∪ {x}, |S|=t, S\{x} ⊆ P}
    # More simply: new t-subsets = {S : S ⊆ P ∪ {x}, |S| = t, x ∈ S}
    # = {S : S = R ∪ {x}, R ⊆ P, |R| = t-1}
    
    # So for a partial block P (as a set), adding x covers all t-subsets of the form R ∪ {x}
    # where R is a (t-1)-subset of P.

    # Precompute: for each point x, and each (t-1)-subset R not containing x,
    # the t-subset R ∪ {x}
    # But this is large. Let's use a different approach.

    # For greedy selection, we need to compute for each candidate x not in P:
    # gain(x) = number of uncovered t-subsets S such that S ⊆ P ∪ {x} and x ∈ S
    # = number of (t-1)-subsets R of P such that R ∪ {x} is uncovered.

    # Let's precompute for each (t-1)-subset R, the list of x not in R such that R ∪ {x} is a valid t-subset
    # Actually, for each R (a (t-1)-subset), the possible x are all points not in R, and R ∪ {x} is always a valid t-subset.
    # But we only care if R ∪ {x} is uncovered.

    # Let's use a different representation:
    # uncovered_count[S] for each t-subset S, initially 1, decremented when covered.
    # But we need to know which t-subsets are covered by a given block.

    # Let's just use the direct approach with frozensets and be careful about performance.

    # Precompute all (t-1)-subsets for each point set? No, let's compute on the fly.

    # For speed, let's precompute for each point x, a list of all t-subsets containing x.
    # tsubsets_with[x] = [S for S in all_tsubs if x in S]
    # This is C(v-1, t-1) per point, total v * C(v-1, t-1) = C(v, t) * t / v * v = C(v,t) * t
    # For v=24, t=3: C(24,3)=2024, so 2024*3=6072 entries total. That's fine.

    tsubsets_with = [[] for _ in range(v)]
    for S in all_tsubs:
        for x in S:
            tsubsets_with[x].append(frozenset(S))

    # Also, for a partial block P (as a frozenset) and candidate x, the new t-subsets covered
    # are those S containing x such that S \ {x} ⊆ P.
    # Equivalently, S = R ∪ {x} where R ⊆ P, |R| = t-1.
    # So we need to check for each R ⊆ P with |R| = t-1, whether R ∪ {x} is uncovered.

    # Let's precompute for each (t-1)-subset R, the frozenset R and the list of x not in R.
    # Actually, let's just compute the (t-1)-subsets of P on the fly.

    def compute_gain(P_frozen, x, coverage_counter):
        """
        Compute the number of uncovered t-subsets that would be newly covered
        if x is added to the partial block P_frozen.
        P_frozen is a frozenset of points already in the block.
        """
        # Find all (t-1)-subsets R of P_frozen
        gain = 0
        P_list = list(P_frozen)
        for R in itertools.combinations(P_list, t - 1):
            S = frozenset(R + (x,))
            if coverage_counter[S] == 0:
                gain += 1
        return gain

    while True:
        if time.time() > deadline - 5:
            break

        # Find an uncovered t-subset to seed a new block
        uncovered = None
        for S in all_tsubs_frozen:
            if coverage[S] == 0:
                uncovered = S
                break
        if uncovered is None:
            break  # All covered

        # Start a new block with one random point from uncovered
        block_points = set()
        # Pick a random point from the uncovered t-subset
        start_point = rng.choice(list(uncovered))
        block_points.add(start_point)
        P_frozen = frozenset(block_points)

        # Greedily add points
        while len(block_points) < k:
            if time.time() > deadline - 5:
                break

            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_points:
                    continue
                g = compute_gain(P_frozen, x, coverage)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None or best_gain <= 0:
                # No point improves coverage; pick a random one to complete the block
                candidates = [x for x in range(v) if x not in block_points]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            block_points.add(best_x)
            P_frozen = frozenset(block_points)

        if len(block_points) < k:
            # Try to fill with random points
            remaining = [x for x in range(v) if x not in block_points]
            rng.shuffle(remaining)
            for x in remaining:
                if len(block_points) >= k:
                    break
                block_points.add(x)
            if len(block_points) < k:
                continue  # Shouldn't happen if v >= k

        block_frozen = frozenset(block_points)
        blocks.append(block_frozen)

        # Update coverage for all t-subsets of this block
        for S in itertools.combinations(sorted(block_points), t):
            S_frozen = frozenset(S)
            coverage[S_frozen] += 1

    # Remove redundant blocks
    # A block is redundant if every t-subset it covers is also covered by other blocks
    # We need to be careful: we can only remove a block if after removal, all t-subsets
    # are still covered.
    
    # First, compute coverage without considering removals, then try removing blocks one by one.
    # For each block, check if it's the only block covering some t-subset.
    
    # Build a map: for each t-subset, which blocks cover it?
    # This is expensive. Let's do it differently.
    
    # For each block B, find a t-subset S ⊆ B that is covered only by B (if any).
    # If no such S exists, B is redundant.
    
    # To check this, for each S ⊆ B, count how many blocks cover S.
    # We already have coverage[S] = total number of blocks covering S.
    # But we need to know if B is the only one. So we need to check if there's another block B' ≠ B with S ⊆ B'.
    
    # Let's do a pass: for each block, tentatively remove it and check if any t-subset becomes uncovered.
    # This is O(num_blocks * C(k,t) * num_blocks) which might be too slow.
    
    # Alternative: for each t-subset S, record the list of blocks covering it.
    # Then a block B is redundant if for all S ⊆ B, there exists B' ≠ B with S ⊆ B'.
    
    # Let's build block_to_tsubs and tsub_to_blocks.
    tsub_to_blocks = {S: [] for S in all_tsubs_frozen}
    for i, B in enumerate(blocks):
        for S in itertools.combinations(sorted(B), t):
            S_frozen = frozenset(S)
            tsub_to_blocks[S_frozen].append(i)
    
    # Now, mark redundant blocks
    redundant = [False] * len(blocks)
    for i, B in enumerate(blocks):
        is_redundant = True
        for S in itertools.combinations(sorted(B), t):
            S_frozen = frozenset(S)
            covering_blocks = tsub_to_blocks[S_frozen]
            if len(covering_blocks) == 1 and covering_blocks[0] == i:
                is_redundant = False
                break
        if is_redundant:
            redundant[i] = True
    
    # Remove redundant blocks
    if any(redundant):
        blocks = [B for i, B in enumerate(blocks) if not redundant[i]]
    
    # Convert frozensets to sorted lists
    return [sorted(B) for B in blocks]


def _verify_covering(v, k, t, blocks):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for B in blocks:
        for S in itertools.combinations(B, t):
            covered.add(S)
    all_tsubs = set(itertools.combinations(range(v), t))
    return covered == all_tsubs


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    total_deadline = start_time + 115.0  # Leave 5s margin
    
    # Try multiple random restarts
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed=0)
    blocks0 = _greedy_cover(v, k, t, seed=0, time_limit=total_deadline - start_time - 5)
    if blocks0 and _verify_covering(v, k, t, blocks0):
        best_blocks = blocks0
        best_count = len(blocks0)
    
    # Try more random restarts if time permits
    seed = 1
    while time.time() < total_deadline - 10:
        remaining = total_deadline - time.time() - 5
        if remaining < 2:
            break
        # Use a fraction of the remaining time for each restart
        time_per_restart = min(remaining * 0.8, 30.0)
        blocks = _greedy_cover(v, k, t, seed=seed, time_limit=time_per_restart)
        if blocks and _verify_covering(v, k, t, blocks):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        seed += 1
    
    if best_blocks is None:
        # Fallback: return empty list (shouldn't happen)
        return []
    
    return best_blocks
# EVOLVE-BLOCK-END
