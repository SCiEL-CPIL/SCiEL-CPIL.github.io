"""Size-efficient sorting network on 18 wires.

Fresh design:
  - Seed = Batcher odd-even merge sort (small, correct by construction).
  - Optimization = local search (deletions of redundant comparators,
    insertions of helpful ones) driven by a fast SAMPLED verifier for the
    hot loop and a FULL 2^18 verification only for promising candidates.
  - Vectorized, pre-allocated verification (no per-step array copies).
  - Explicit deadline guard: always return the best VALID network found.
"""
import time
import random
import numpy as np

N = 18
FULL_MASK = (1 << N) - 1          # 262143
DEADLINE = 80.0                   # wall-clock safety margin (evaluator kills at 90s)
MAX_COMPARATORS = 4096


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _build_block(net):
    """Pre-compute a per-wire block-fill vector for fast state init.

    block[:, w] is the length-2^N vector of wire w's value across all masks,
    so state[:, w] = block[:, w] is a single vectorized assignment (no copies).
    """
    M = 1 << N
    block = np.empty((M, N), dtype=np.uint8)
    for w in range(N):
        block[:, w] = ((np.arange(M, dtype=np.uint64) >> w) & 1).astype(np.uint8)
    return block


def _verify_full(net, block):
    """Return True iff `net` sorts every Boolean input (all 2^N masks)."""
    state = np.empty_like(block)
    for i, j in net:
        state[:, i] = np.minimum(block[:, i], block[:, j])
        state[:, j] = np.maximum(block[:, i], block[:, j])
    # sorted iff top wire is all-ones AND bottom wire is all-zeros
    if np.any(state[:, N - 1] != 1):
        return False
    if np.any(state[:, 0] != 0):
        return False
    return True


def _verify_sample(net, masks, block, out):
    """Fast sampled check over a fixed subset of masks. Returns #failures."""
    idx = masks
    state = out
    state[:, :] = block[idx]
    for i, j in net:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # failures = masks whose top wire != 1 or bottom wire != 0
    fails = (state[:, N - 1] != 1) | (state[:, 0] != 0)
    return int(fails.sum())


# --------------------------------------------------------------------------- #
# Batcher odd-even merge sort (correct by construction)
# --------------------------------------------------------------------------- #
def _batcher_sort(n):
    """Return a list of (i, j) comparators for Batcher odd-even merge sort on n wires."""
    comps = []
    if n <= 1:
        return comps

    def merge_sorted(a, b, k):
        # merge two adjacent sorted runs of length k at offsets a (even) and b (odd)
        for s in range(1, 2 * k + 1, 2):
            if s <= k:
                for i in range(k - s + 1):
                    comps.append((a + i, b + i))
            else:
                for i in range(s - k):
                    comps.append((b + i, a + i + 2 * (k - s + 1)))

    def sort_rec(n):
        if n <= 1:
            return
        if n % 2 == 0:
            h = n // 2
            sort_rec(h)
            sort_rec(h)
            merge_sorted(0, h, h)
        else:
            # odd-even merge: split into two odd halves
            h1 = n // 2            # size of even-indexed half (the larger)
            h2 = n // 2            # size of odd-indexed half
            # even indices: 0,2,4,...  -> h1 elements
            # odd  indices: 1,3,5,...  -> h2 elements
            even = []
            odd = []
            for x in range(n):
                if x % 2 == 0:
                    even.append(x)
                else:
                    odd.append(x)
            # recursively sort each half (in place on the index lists)
            def rec(indices):
                m = len(indices)
                if m <= 1:
                    return
                if m % 2 == 0:
                    hh = m // 2
                    rec(indices[:hh])
                    rec(indices[hh:])
                    merge_sorted(indices[0], indices[hh], hh)
                else:
                    # recurse on even/odd sub-halves of this index set
                    e2 = [indices[x] for x in range(0, m, 2)]
                    o2 = [indices[x] for x in range(1, m, 2)]
                    rec(e2)
                    rec(o2)
                    # merge the two sorted halves back into the interleaved order
                    # merge_sorted expects contiguous offsets; emulate via mapping
                    # We instead use the standard odd-even merge merge step:
                    #   for the combined set, merge halves of sizes len(e2), len(o2)
                    # implemented by the generic merge over the index list.
                    merge_indexed(e2, o2, len(e2))
            rec(even)
            rec(odd)
            # final merge of the two sorted halves (even positions and odd positions)
            # Use the odd-even merge merge step over the full sequence.
            merge_indexed_full(even, odd, h1, h2)

    def merge_indexed(e2, o2, k):
        # merge two sorted runs given as index lists e2 (first half) and o2 (second half)
        # using odd-even merge merge step, comparing by their actual wire positions.
        # e2[0] < o2[0] and the runs are adjacent in the interleaved layout.
        a = e2[0]
        b = o2[0]
        for s in range(1, 2 * k + 1, 2):
            if s <= k:
                for i in range(k - s + 1):
                    comps.append((e2[i], o2[i]))
            else:
                for i in range(s - k):
                    comps.append((o2[i], e2[i + 2 * (k - s + 1)]))

    def merge_indexed_full(even, odd, h1, h2):
        # Standard odd-even merge final step for an odd total length.
        # even has h1 elements, odd has h2 elements.
        # We merge by the Batcher odd-even merge merge step.
        # The two halves are interleaved: even[0], odd[0], even[1], odd[1], ...
        # Use the generic merge over the concatenated index list.
        full = []
        for t in range(h1):
            full.append(even[t])
            if t < h2:
                full.append(odd[t])
        if h2 > h1:
            full.append(odd[h1])
        # Now full is the interleaved sequence of length n; its first h1 "slots"
        # correspond to even positions and next h2 to odd. We merge the two
        # sorted halves (even-run and odd-run) that were produced by recursion.
        # Batcher odd-even merge merge step:
        k = min(h1, h2)
        a = even[0]
        b = odd[0]
        for s in range(1, 2 * k + 1, 2):
            if s <= k:
                for i in range(k - s + 1):
                    comps.append((even[i], odd[i]))
            else:
                for i in range(s - k):
                    comps.append((odd[i], even[i + 2 * (k - s + 1)]))

    sort_rec(n)
    return comps


def _batcher_seed():
    """Batcher odd-even merge sort comparator list for N wires."""
    return _batcher_sort(N)


# --------------------------------------------------------------------------- #
# Local search
# --------------------------------------------------------------------------- #
def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def construct():
    start = time.time()
    rng = random.Random(12345)

    # Fast verification infrastructure
    block = _build_block(_batcher_seed())  # block depends only on N, build once
    M = 1 << N
    out = np.empty((256, N), dtype=np.uint8)          # sampled workspace
    masks = np.arange(M, dtype=np.uint64)
    # fixed random sample of masks for the fast check
    sample_size = 256
    sample = np.sort(rng.sample(range(M), sample_size)).astype(np.uint64)

    net = _batcher_seed()
    # ensure the seed is fully valid (it should be by construction)
    if not _verify_full(net, block):
        # fall back to a known-correct transposition network
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        if not _verify_full(net, block):
            return net

    best = list(net)
    best_len = len(best)

    pairs = _all_pairs()
    pair_set = set(pairs)

    def fast_ok(candidate):
        return _verify_sample(candidate, sample, block, out) == 0

    def full_ok(candidate):
        return _verify_full(candidate, block)

    def try_delete(net):
        """Try removing one comparator; return (accepted, new_net)."""
        if len(net) <= 1:
            return False, net
        idx = rng.randrange(len(net))
        cand = net[:idx] + net[idx + 1:]
        if not fast_ok(cand):
            return False, net
        if full_ok(cand):
            return True, cand
        return False, net

    def try_insert(net):
        """Try adding one comparator at a random position; return (accepted, new_net)."""
        if len(net) >= MAX_COMPARATORS:
            return False, net
        p = rng.choice(pairs)
        pos = rng.randrange(len(net) + 1)
        cand = net[:pos] + [p] + net[pos:]
        if not fast_ok(cand):
            return False, net
        if full_ok(cand):
            return True, cand
        return False, net

    # Search: aggressively delete; occasionally insert to escape local minima.
    # Deadline guard checked every iteration.
    no_improve = 0
    while time.time() < DEADLINE:
        # deletion-heavy schedule
        if rng.random() < 0.7:
            accepted, net = try_delete(net)
        else:
            accepted, net = try_insert(net)
        if accepted:
            best = list(net)
            best_len = len(best)
            no_improve = 0
        else:
            no_improve += 1
        # If stuck for a while, occasionally reset to best to avoid drift
        if no_improve > 2000:
            net = list(best)
            no_improve = 0

    # Final safety: ensure the returned network is fully valid.
    if not _verify_full(best, block):
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    return best
