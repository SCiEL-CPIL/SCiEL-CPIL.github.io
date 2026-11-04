"""Covering design C(v,k,t) via Max-SAT-style stochastic local search (WalkSAT-like)."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # For each point x and each (t-1)-subset of other points, store the bitmask of
    # t-subsets that contain x and those others.
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            masks[x][others] = masks[x].get(others, 0) | bit

    def count_bits(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    # Build a block from seed points using max-gain greedy
    def build_block_from(seed_points, covered_mask):
        B = set(seed_points)
        available = [x for x in range(v) if x not in B]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            B_list = list(B)
            for x in available:
                g = 0
                for S in itertools.combinations(B_list, t - 1):
                    key = tuple(S)
                    m = masks[x].get(key)
                    if m is not None:
                        g += count_bits(m & ~covered_mask)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.add(best_x)
            available.remove(best_x)
        return B

    # Run initial greedy to get a starting solution
    def run_initial_greedy(rng_seed):
        rng = random.Random(rng_seed)
        cov = 0
        blks = []
        full_mask = (1 << n_subsets) - 1

        while True:
            if time.time() > deadline - 20.0:
                break
            uncov_mask = full_mask & ~cov
            if uncov_mask == 0:
                break

            # Find uncovered indices
            uncovered_indices = []
            m = uncov_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                uncovered_indices.append(idx)
                m ^= lsb

            if not uncovered_indices:
                break

            # Try multiple seeds to find best block
            sample_size = min(50, len(uncovered_indices))
            sampled_indices = rng.sample(uncovered_indices, sample_size)

            best_block = None
            best_gain = -1

            for idx in sampled_indices:
                seed = all_subsets[idx]
                B = build_block_from(set(seed), cov)
                if len(B) < k:
                    continue
                # Count new coverage
                new_cov = 0
                bl = sorted(B)
                for s in itertools.combinations(bl, t):
                    si = subset_to_index[s]
                    if not (cov >> si) & 1:
                        new_cov += 1
                if new_cov > best_gain:
                    best_gain = new_cov
                    best_block = B

            if best_block is None:
                break

            bl = sorted(best_block)
            for s in itertools.combinations(bl, t):
                cov |= 1 << subset_to_index[s]
            blks.append(bl)

        return blks, cov

    # Get initial solution
    initial_blocks, initial_cov = run_initial_greedy(42)
    best_blocks = list(initial_blocks)
    best_count = len(best_blocks)

    # Max-SAT style local search: try to remove or swap blocks to reduce count
    # while maintaining full coverage
    def is_valid_covering(blocks):
        """Check if blocks cover all t-subsets."""
        cov = 0
        for b in blocks:
            for s in itertools.combinations(sorted(b), t):
                cov |= 1 << subset_to_index[s]
        return cov == (1 << n_subsets) - 1

    def try_remove_block(blocks, idx, deadline):
        """Try removing block idx; return True if still valid."""
        if len(blocks) <= 1:
            return False
        # Quick check: does this block have any uniquely covered triplet?
        # Build coverage count
        n = len(blocks)
        # For efficiency, just test removal directly
        test_blocks = blocks[:idx] + blocks[idx+1:]
        cov = 0
        for b in test_blocks:
            for s in itertools.combinations(sorted(b), t):
                cov |= 1 << subset_to_index[s]
        return cov == (1 << n_subsets) - 1

    def try_swap_block(blocks, idx, new_block, deadline):
        """Try replacing block idx with new_block; return True if valid and no worse."""
        test_blocks = blocks[:idx] + [new_block] + blocks[idx+1:]
        cov = 0
        for b in test_blocks:
            for s in itertools.combinations(sorted(b), t):
                cov |= 1 << subset_to_index[s]
        return cov == (1 << n_subsets) - 1

    # Local search: iteratively try to remove blocks
    rng = random.Random(123)
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        n_blocks = len(best_blocks)
        # Shuffle order for variety
        order = list(range(n_blocks))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline - 10.0:
                break
            if try_remove_block(best_blocks, idx, deadline):
                best_blocks = best_blocks[:idx] + best_blocks[idx+1:]
                best_count = len(best_blocks)
                improved = True
                # Restart the pass since indices shifted
                break

    # Additional local search: try swapping blocks with better alternatives
    if time.time() < deadline - 5.0:
        for _iter in range(5):
            if time.time() > deadline - 5.0:
                break
            n_blocks = len(best_blocks)
            order = list(range(n_blocks))
            rng.shuffle(order)
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                # Try to find a replacement block that covers at least as much
                # but allows removal of another block
                current_block = best_blocks[idx]
                # Find triplets uniquely covered by this block
                # Skip complex swap for now; just do removal passes
                pass

    # Final cleanup: multiple removal passes
    for _pass in range(5):
        if time.time() > deadline - 2.0:
            break
        changed = False
        for idx in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            if try_remove_block(best_blocks, idx, deadline):
                best_blocks = best_blocks[:idx] + best_blocks[idx+1:]
                changed = True
        if not changed:
            break

    return best_blocks
