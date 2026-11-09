import itertools
import random
import time
import numpy as np


def _build_pre(v, k, t, seed=12345):
    """Precompute:
       - all_t: list of all t-subsets (sorted tuples)
       - t_to_idx: mapping tuple -> index
       - block_to_mask: array [num_blocks, num_t_subsets] uint16 membership counts
    """
    rng = random.Random(seed)
    all_t = list(itertools.combinations(range(v), t))
    t_to_idx = {sub: i for i, sub in enumerate(all_t)}
    nt = len(all_t)

    # Precompute for each point p: list of (t-subset-index) that contain p
    # and for each pair (a,b): list of t-subset-indices that contain both a and b
    # We'll precompute per-point membership for speed.
    point_t_sets = [None] * v
    for p in range(v):
        point_t_sets[p] = np.fromiter(
            (t_to_idx[sub] for sub in all_t if p in sub), dtype=np.int32
        )

    # Pair membership: (a,b) -> indices of t-subsets containing both
    pair_t_sets = {}
    for a in range(v):
        for b in range(a + 1, v):
            pair_t_sets[(a, b)] = np.fromiter(
                (t_to_idx[sub] for sub in all_t if a in sub and b in sub),
                dtype=np.int32,
            )

    # Enumerate all C(v,k) blocks
    all_blocks = list(itertools.combinations(range(v), k))

    # For each block, build its mask
    # We compute incrementally: a block's t-subsets are exactly the t-subsets
    # contained in the block. We can compute by iterating combinations of block.
    # num_blocks * C(k, t) = 50388 * 35 = ~1.76M ops, each a small tuple.
    block_masks = np.zeros((len(all_blocks), nt), dtype=np.uint16)
    block_list = []
    for bi, blk in enumerate(all_blocks):
        for tsub in itertools.combinations(blk, t):
            idx = t_to_idx[tsub]
            block_masks[bi, idx] = 1
        block_list.append(blk)

    return all_blocks, block_masks, point_t_sets, pair_t_sets, all_t, t_to_idx, nt


def _get_unique_indices(membership):
    """Given membership array [N, nt] uint16, return for each block i the list of
    subset indices where membership[:, idx] == 1 and membership[i, idx] == 1."""
    N, nt = membership.shape
    total = membership.sum(axis=0)
    uniques = []
    for i in range(N):
        m = membership[i]
        # subsets where this block is present (m==1) and total==1
        idxs = np.where((m == 1) & (total == 1))[0]
        uniques.append(idxs)
    return uniques


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s

    # Precompute (fixed for this task)
    all_blocks, block_masks, point_t_sets, pair_t_sets, all_t, t_to_idx, nt = \
        _build_pre(v, k, t, seed=12345)

    num_blocks = len(all_blocks)

    best_blocks = None
    best_count = float('inf')
    best_membership = None

    def elapsed_ok():
        return time.time() < deadline

    def compute_membership(blocks):
        """blocks: list of tuples (indices into all_blocks). Returns membership array."""
        if not blocks:
            return np.zeros((0, nt), dtype=np.uint16)
        return block_masks[np.array(blocks, dtype=np.int32)].sum(axis=0).astype(np.uint16)

    def is_valid(membership):
        return bool((membership > 0).all())

    def removal_pass(blocks, membership):
        """Try to remove redundant blocks. Returns (new_blocks, new_membership, removed_any)."""
        # unique indices per block
        uniques = _get_unique_indices(membership)
        n = len(blocks)
        order = list(range(n))
        random.shuffle(order)
        removed = []
        for i in order:
            if i in removed:
                continue
            if len(uniques[i]) == 0:
                # Removing i does not break any subset
                # Remove it
                membership -= block_masks[blocks[i]]
                blocks.pop(i)
                # reindex uniques for all j > i (shift)
                for j in range(i + 1, len(blocks)):
                    # uniques[j] still valid (indices unchanged)
                    pass
                removed.append(i)
                # recompute uniques lazily? For correctness of further removals,
                # we need updated uniques. Recompute for remaining blocks.
                if len(blocks) < n:
                    uniques = _get_unique_indices(membership)
                    order = list(range(len(blocks)))
                    random.shuffle(order)
                    removed = []
        return blocks, membership, True

    def swap_pass(blocks, membership, max_iters=200):
        """Try to swap one block for another to reduce unique load or enable removal.
        Returns (new_blocks, new_membership, improved)."""
        n = len(blocks)
        if n == 0:
            return blocks, membership, False
        uniques = _get_unique_indices(membership)
        # unique counts per block
        ucounts = np.array([len(u) for u in uniques])
        improved = False
        for _ in range(max_iters):
            if not elapsed_ok():
                break
            # pick the block with the most uniques
            i = int(np.argmax(ucounts))
            if ucounts[i] == 0:
                # try to remove
                membership -= block_masks[blocks[i]]
                blocks.pop(i)
                improved = True
                uniques = _get_unique_indices(membership)
                ucounts = np.array([len(u) for u in uniques])
                continue
            # Try to swap block i with some other block b (not in current set)
            # such that after swap, coverage remains valid AND ucounts decreases.
            current_set = set(blocks)
            candidates = [b for b in range(num_blocks) if b not in current_set]
            random.shuffle(candidates)
            best_b = None
            best_delta = 0
            for b in candidates[:200]:
                # simulate: remove i, add b
                new_membership = membership - block_masks[blocks[i]] + block_masks[b]
                if not (new_membership > 0).all():
                    continue
                # compute new unique count for b
                # subsets uniquely covered by b in new_membership
                m_b = block_masks[b]
                total_new = new_membership
                # unique for b = (m_b==1) & (total_new==1)
                new_u_b = int(((m_b == 1) & (total_new == 1)).sum())
                # new unique count for other blocks: a subset that was unique to some
                # other block j could be lost if b also covers it... actually removal of i
                # only affects subsets that i covered. Adding b covers more.
                # A subset unique to block j (j != i) remains unique unless b also covers it.
                # So new unique count for j = old unique count for j - (# of j's uniques that b covers)
                # Total unique count change:
                #   - remove i's uniques (they're now covered by nothing unless b covers them)
                #   Actually if a subset was unique to i, after removing i it's uncovered
                #   unless b covers it. If b covers it, it becomes unique to b (or shared).
                #   If b doesn't cover it, the swap is invalid (already checked via new_membership>0).
                #   So all of i's uniques are covered by b (since new_membership>0 and only
                #   i and b changed). They may now be unique to b or shared with others.
                # This is getting complex. Let's just recompute uniques for the whole set.
                # But that's O(n*nt) which is ~150*3876 ~ 580k, times 200 candidates = 116M per iter.
                # Too slow. Let's approximate: use new_u_b and old ucounts[i].
                # The swap is beneficial if new_u_b < old ucounts[i] (roughly).
                # Actually the total unique count may go up or down. Let's just use:
                # delta = new_u_b - ucounts[i]  (approximate, ignoring redistribution)
                delta = new_u_b - ucounts[i]
                if delta < -best_delta:
                    best_delta = -delta
                    best_b = b
                    if best_delta > 10:  # good enough
                        break
            if best_b is not None:
                # do the swap
                membership = membership - block_masks[blocks[i]] + block_masks[best_b]
                blocks[i] = best_b
                improved = True
                uniques = _get_unique_indices(membership)
                ucounts = np.array([len(u) for u in uniques])
            else:
                # No swap found for block i. Try removing i if possible (it has uniques, so no).
                # Mark i as "tried" by temporarily setting its ucount to -1 so we don't pick it again
                # within this pass iteration. Instead, just break to outer loop? 
                # Let's try another block.
                ucounts[i] = -1  # skip in this iteration
                if (ucounts < 0).all() or ucounts.max() == 0:
                    break
        return blocks, membership, improved

    def point_optimize(blocks, membership, max_rounds=50):
        """For each block, try replacing one point with another to reduce its unique load.
        Returns (new_blocks, new_membership, improved)."""
        improved = False
        for _ in range(max_rounds):
            if not elapsed_ok():
                break
            n = len(blocks)
            if n == 0:
                break
            uniques = _get_unique_indices(membership)
            ucounts = np.array([len(u) for u in uniques])
            if ucounts.max() == 0:
                break
            i = int(np.argmax(ucounts))
            if ucounts[i] == 0:
                break
            block = blocks[i]
            block_set = set(block)
            # For each point p in block, try replacing with point q not in block
            best_change = None
            best_delta = 0
            for p in block:
                for q in range(v):
                    if q in block_set:
                        continue
                    # new block = (block - {p}) + {q}
                    new_block = tuple(sorted((list(block) + [q])[:k] if False else
                                             tuple(x for x in block if x != p) + (q,)))
                    new_block = tuple(sorted(new_block))
                    # find its index in all_blocks
                    # We need a lookup. Let's build a dict mapping tuple->index.
                    # To avoid building it every time, build once outside. But for simplicity:
                    # Use bisect or a global dict. Let's build a dict in the precompute.
                    # Actually we don't have it. Let's create it here lazily.
                    # This is expensive. Let's skip point_optimize for now and rely on swap.
                    pass
            # Skip point optimization for now (too slow to implement efficiently)
            break
        return blocks, membership, improved

    # Build a lookup for block tuple -> index for point_optimize if needed
    block_to_idx = {blk: i for i, blk in enumerate(all_blocks)}

    def point_optimize2(blocks, membership, max_rounds=30):
        improved = False
        for _ in range(max_rounds):
            if not elapsed_ok():
                break
            n = len(blocks)
            if n == 0:
                break
            uniques = _get_unique_indices(membership)
            ucounts = np.array([len(u) for u in uniques])
            if ucounts.max() == 0:
                break
            i = int(np.argmax(ucounts))
            if ucounts[i] == 0:
                break
            block = blocks[i]
            block_set = set(block)
            best_change = None
            best_delta = 0
            for p in block:
                for q in range(v):
                    if q in block_set:
                        continue
                    new_block = tuple(sorted(tuple(x for x in block if x != p) + (q,)))
                    if new_block not in block_to_idx:
                        continue
                    bidx = block_to_idx[new_block]
                    # simulate swap
                    new_membership = membership - block_masks[blocks[i]] + block_masks[bidx]
                    if not (new_membership > 0).all():
                        continue
                    m_b = block_masks[bidx]
                    new_u_b = int(((m_b == 1) & (new_membership == 1)).sum())
                    delta = new_u_b - ucounts[i]
                    if delta < -best_delta:
                        best_delta = -delta
                        best_change = bidx
                        if best_delta > 5:
                            break
                if best_change is not None and best_delta > 5:
                    break
            if best_change is not None:
                membership = membership - block_masks[blocks[i]] + block_masks[best_change]
                blocks[i] = best_change
                improved = True
            else:
                break
        return blocks, membership, improved

    def greedy_build(rng):
        """Greedy construction from scratch using vectorized gain computation."""
        remaining = np.ones(nt, dtype=bool)
        # For each potential block, the count of currently-uncovered subsets it covers
        # We maintain gain[b] = number of uncovered subsets in block b.
        # Initially gain[b] = sum(block_masks[b] * remaining)
        gain = (block_masks * remaining.astype(np.uint16)).sum(axis=1).astype(np.int32)
        chosen = []
        chosen_set = set()
        while remaining.any() and elapsed_ok():
            # pick block with max gain
            b = int(np.argmax(gain))
            if gain[b] <= 0:
                # stuck? shouldn't happen since remaining has uncovered subsets
                # and every uncovered subset is covered by some block.
                # But if gain is 0 for all, break.
                break
            chosen.append(b)
            chosen_set.add(b)
            remaining[block_masks[b] == 1] = False
            # Update gain for all other blocks: subtract overlap with new block
            # gain[b'] -= number of subsets in b' that are now covered (i.e., in new block)
            # = (block_masks[b'] & block_masks[b]).sum(axis=1)
            overlap = (block_masks * block_masks[b].astype(np.uint16)).sum(axis=1)
            gain -= overlap
            # Set gain of chosen blocks to -1 to avoid re-selection
            gain[chosen] = -1

        # Verify
        membership = block_masks[np.array(chosen, dtype=np.int32)].sum(axis=0).astype(np.uint16) if chosen else np.zeros(nt, dtype=np.uint16)
        if not is_valid(membership):
            # Fallback: add blocks until valid
            for b in range(num_blocks):
                if b not in chosen_set:
                    membership += block_masks[b]
                    chosen.append(b)
                    chosen_set.add(b)
                    if is_valid(membership):
                        break
        return chosen, membership

    # Multiple restarts
    num_restarts = 0
    while elapsed_ok():
        rng = random.Random()
        # Greedy build
        blocks, membership = greedy_build(rng)
        if not is_valid(membership):
            # Should not happen, but ensure validity
            for b in range(num_blocks):
                if b not in set(blocks):
                    membership += block_masks[b]
                    blocks.append(b)
                    if is_valid(membership):
                        break

        # Local search phases
        # 1. Removal pass (repeated until no more removals)
        for _ in range(10):
            if not elapsed_ok():
                break
            blocks, membership, changed = removal_pass(blocks, membership)
            if not changed:
                break

        # 2. Swap pass
        if elapsed_ok():
            blocks, membership, _ = swap_pass(blocks, membership, max_iters=500)

        # 3. Point optimize
        if elapsed_ok():
            blocks, membership, _ = point_optimize2(blocks, membership, max_rounds=30)

        # 4. Another removal pass
        for _ in range(10):
            if not elapsed_ok():
                break
            blocks, membership, changed = removal_pass(blocks, membership)
            if not changed:
                break

        # 5. Final swap pass
        if elapsed_ok():
            blocks, membership, _ = swap_pass(blocks, membership, max_iters=300)

        # 6. Final removal
        for _ in range(5):
            if not elapsed_ok():
                break
            blocks, membership, changed = removal_pass(blocks, membership)
            if not changed:
                break

        count = len(blocks)
        if count < best_count:
            best_count = count
            best_blocks = blocks[:]
            best_membership = membership.copy()
            if best_count <= 131:
                break
        num_restarts += 1

    # Convert block indices back to point lists
    result = [list(all_blocks[i]) for i in best_blocks]
    return result
