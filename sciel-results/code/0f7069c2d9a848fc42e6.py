import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    best_blocks = None
    best_len = float('inf')

    total_t = sum(1 for _ in itertools.combinations(range(v), t))

    def triple_index(tri):
        # Encode a sorted t-tuple as an integer for O(1) dict lookups
        idx = 0
        for x in tri:
            idx = idx * v + x
        return idx

    def build_coverage(blocks):
        """Return (count_dict, covered_count) for the given blocks."""
        cnt = defaultdict(int)
        for b in blocks:
            for tri in itertools.combinations(b, t):
                cnt[triple_index(tri)] += 1
        return cnt, len(cnt)

    def is_valid(blocks):
        _, covered = build_coverage(blocks)
        return covered == total_t

    def remove_redundant(blocks, time_limit=None):
        """Incremental redundancy removal using a coverage-count dict."""
        blocks = list(blocks)
        if not blocks:
            return blocks
        cnt, _ = build_coverage(blocks)
        changed = True
        while changed:
            changed = False
            if time_limit is not None and time.time() > time_limit:
                break
            for i in range(len(blocks)):
                # Decrement counts for block i
                for tri in itertools.combinations(blocks[i], t):
                    cnt[triple_index(tri)] -= 1
                    if cnt[triple_index(tri)] == 0:
                        # Would become uncovered -> not redundant
                        # Restore and break
                        cnt[triple_index(tri)] += 1
                        break
                else:
                    # All triples still covered; remove block
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def greedy_random_pivot():
        """Fast greedy: pick random uncovered triple, best block covering it."""
        # Use a dict for uncovered triples (index -> triple) for O(1) ops
        uncovered = {}
        for tri in itertools.combinations(range(v), t):
            uncovered[triple_index(tri)] = tri

        blocks = []
        # Precompute all t-subsets of each possible k-subset? Too large.
        # Instead, for each pivot, enumerate C(v-t, k-t) candidates.
        while uncovered:
            # Pick a random uncovered triple
            idx = random.choice(list(uncovered.keys()))
            pivot = uncovered[idx]
            pivot_set = set(pivot)
            other_points = [p for p in range(v) if p not in pivot_set]

            best_block = None
            best_gain = -1

            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_set | set(combo)))
                gain = 0
                for tri in itertools.combinations(block, t):
                    ti = triple_index(tri)
                    if ti in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block

            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    ti = triple_index(tri)
                    if ti in uncovered:
                        del uncovered[ti]

        return blocks

    def local_search(blocks, max_time):
        """
        Fast local search with incremental coverage tracking.
        Tries: (1) remove redundant blocks, (2) swap a block with a better candidate.
        """
        blocks = list(blocks)
        end_time = min(max_time, deadline)

        # Build coverage index
        cnt, _ = build_coverage(blocks)

        improved = True
        while improved and time.time() < end_time:
            improved = False
            n = len(blocks)

            # Phase 1: Try removing each block (in random order)
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if time.time() > end_time:
                    break
                # Decrement
                removable = True
                for tri in itertools.combinations(blocks[i], t):
                    ti = triple_index(tri)
                    cnt[ti] -= 1
                    if cnt[ti] == 0:
                        removable = False
                        break
                if removable:
                    # All still covered; remove
                    blocks.pop(i)
                    improved = True
                    n = len(blocks)
                    break
                else:
                    # Restore
                    for tri in itertools.combinations(blocks[i], t):
                        ti = triple_index(tri)
                        if cnt[ti] == 0:
                            cnt[ti] += 1
                            break
                        cnt[ti] += 1

            if improved or time.time() > end_time:
                continue

            # Phase 2: Try swapping a block with a better candidate
            order = list(range(len(blocks)))
            random.shuffle(order)
            for i in order:
                if time.time() > end_time:
                    break

                old_block = blocks[i]
                # Find triples only covered by this block
                only_triples = []
                for tri in itertools.combinations(old_block, t):
                    ti = triple_index(tri)
                    if cnt[ti] == 1:
                        only_triples.append(tri)

                if not only_triples:
                    # Redundant (shouldn't happen after Phase 1, but safety)
                    for tri in itertools.combinations(old_block, t):
                        cnt[triple_index(tri)] -= 1
                    blocks.pop(i)
                    improved = True
                    break

                # must_include = union of only_triples
                must_include = set()
                for tri in only_triples:
                    must_include.update(tri)

                if len(must_include) > k:
                    continue

                remaining_points = [p for p in range(v) if p not in must_include]
                need = k - len(must_include)

                # Enumerate candidate blocks containing must_include
                best_candidate = None
                best_new_uncovered = float('inf')

                for combo in itertools.combinations(remaining_points, need):
                    candidate = tuple(sorted(must_include | set(combo)))
                    # Score: after removing old_block and adding candidate,
                    # how many triples become uncovered?
                    # Equivalently: count triples in candidate that are NOT covered
                    # by any other block (i.e., would be new "only" or uncovered)
                    # Actually we want: the number of triples that would become
                    # uncovered if we replace old with candidate.
                    # A triple becomes uncovered if:
                    #   - it's in old_block but not in candidate, AND cnt[tri]==1
                    #   - it's in candidate but not in old_block, AND cnt[tri]==0
                    # But since candidate must cover all only_triples of old,
                    # the first case is handled. Let's compute:
                    # After swap: for each tri in candidate, new_cnt = cnt[tri] - (1 if tri in old else 0) + 1
                    # A tri is uncovered iff new_cnt == 0
                    # For tri in old but not candidate: new_cnt = cnt[tri] - 1, uncovered iff cnt[tri]==1
                    # For tri in candidate but not old: new_cnt = cnt[tri] + 1, never 0 (since cnt>=0)
                    # For tri in both: new_cnt = cnt[tri], uncovered iff cnt[tri]==0 (impossible since tri in old means cnt>=1)
                    # So uncovered after swap = #{tri in old, not in candidate, cnt[tri]==1}
                    # But we already know only_triples are covered by candidate.
                    # So uncovered = #{tri in old, not in candidate, cnt[tri]==1} - #{tri in only_triples that are in candidate}
                    # Wait, only_triples are all in candidate by construction.
                    # So uncovered_after = #{tri in old, cnt[tri]==1} - len(only_triples) + #{tri in old, not in candidate, cnt[tri]==1 and tri not in only_triples}
                    # Hmm, simpler: just count directly.
                    new_uncovered = 0
                    cand_set = set(candidate)
                    old_set = set(old_block)
                    # Triples in old but not in candidate
                    for tri in itertools.combinations(old_block, t):
                        if not (set(tri) <= cand_set):
                            ti = triple_index(tri)
                            if cnt[ti] == 1:
                                new_uncovered += 1
                    # Triples in candidate but not in old: can they be uncovered?
                    # new_cnt for such tri = cnt[tri] + 1 >= 1, so no.
                    # But wait: we need to check if the candidate covers all only_triples.
                    # By construction it does (must_include subset of candidate).
                    # Also need: no triple in candidate that was cnt==0 and not in old.
                    # new_cnt = cnt[tri] + 1 for tri in candidate not in old. If cnt[tri]==0, new_cnt=1. OK.
                    # So new_uncovered is correct.

                    if new_uncovered < best_new_uncovered:
                        best_new_uncovered = new_uncovered
                        best_candidate = candidate
                        if new_uncovered == 0:
                            break

                if best_candidate is not None and best_new_uncovered == 0:
                    # Apply swap
                    for tri in itertools.combinations(old_block, t):
                        cnt[triple_index(tri)] -= 1
                    for tri in itertools.combinations(best_candidate, t):
                        cnt[triple_index(tri)] += 1
                    blocks[i] = list(best_candidate)
                    improved = True
                    break

        return remove_redundant(blocks, time_limit=deadline)

    # Main loop: multiple randomized restarts
    while time.time() < deadline - 5:
        blocks = greedy_random_pivot()
        blocks = remove_redundant(blocks, time_limit=deadline - 5)

        if time.time() < deadline - 10:
            blocks = local_search(blocks, deadline - 5)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

        # Early exit if we hit the known lower bound (69)
        if best_len <= 69:
            break

    # Fallback: deterministic greedy if nothing was found
    if best_blocks is None:
        blocks = []
        uncovered = set(itertools.combinations(range(v), t))
        while uncovered:
            pivot_triple = min(uncovered)
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
        best_blocks = remove_redundant(blocks)

    return best_blocks
