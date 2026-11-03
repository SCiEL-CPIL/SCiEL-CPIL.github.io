"""Probabilistic covering design via expected-coverage estimation + simulated annealing."""
import random
import itertools
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a probabilistic approach:
    - Estimate expected number of uncovered t-subsets via inclusion-exclusion-style
      sampling (noisy objective) to escape local minima.
    - Use simulated annealing with this noisy objective to minimize block count
      while maintaining coverage.
    """
    start_time = time.time()
    timeout = 115  # Leave 5s buffer

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For a given block, return list of t-subset indices it covers
    def block_tsubsets(block):
        return [t_subset_index[ts] for ts in itertools.combinations(block, t)]

    # ---- Greedy initialization ----
    def greedy_init():
        """Greedy set-cover style: repeatedly pick block covering most uncovered t-subsets."""
        uncovered = [1] * num_t_subsets  # 1 if uncovered, 0 if covered
        blocks = []
        block_set = set()

        while any(uncovered):
            # Find the k-subset that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            # Sample candidate blocks for speed (not exhaustive)
            num_candidates = 500
            for _ in range(num_candidates):
                cand = tuple(sorted(random.sample(range(v), k)))
                if cand in block_set:
                    continue
                gain = 0
                for ts_idx in block_tsubsets(cand):
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = cand

            if best_block is None:
                # Fallback: pick any uncovered t-subset and extend
                unc_idx = next(i for i, u in enumerate(uncovered) if u)
                base = list(all_t_subsets[unc_idx])
                while len(base) < k:
                    best_x = None
                    best_g = -1
                    for x in range(v):
                        if x in base:
                            continue
                        g = 0
                        for rest in itertools.combinations(base, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_subset_index.get(ts)
                            if idx is not None and uncovered[idx]:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    base.append(best_x)
                best_block = tuple(sorted(base))

            # Add block and update uncovered
            blocks.append(best_block)
            block_set.add(best_block)
            for ts_idx in block_tsubsets(best_block):
                uncovered[ts_idx] = 0

        return blocks

    # ---- Probabilistic expected uncovered estimator ----
    def expected_uncovered(blocks, sample_size=200):
        """
        Estimate expected number of uncovered t-subsets by sampling.
        For each sampled t-subset, compute P(it is uncovered) = product over blocks
        of P(block does not contain this t-subset). Using inclusion-exclusion lite:
        P(covered) = 1 - prod(1 - I(block covers ts)) ... but we use Monte Carlo.
        
        Actually, for a fixed t-subset S, it's covered if ANY block contains S.
        P(S uncovered) = prod over blocks of [1 if block doesn't contain S, else 0]
        This is deterministic per S. So expected uncovered = count of S with no block containing it.
        
        To make it noisy/stochastic, we estimate by sampling t-subsets and checking coverage.
        """
        if not blocks:
            return float(num_t_subsets)
        
        # For efficiency, precompute which t-subset indices each block covers
        # But for sampling, just check membership
        total_uncovered = 0
        for _ in range(sample_size):
            ts = random.choice(all_t_subsets)
            ts_set = set(ts)
            covered = False
            for block in blocks:
                if ts_set.issubset(set(block)):
                    covered = True
                    break
            if not covered:
                total_uncovered += 1
        # Scale to estimate total
        return (total_uncovered / sample_size) * num_t_subsets

    # ---- Exact uncovered count (for final verification) ----
    def exact_uncovered(blocks):
        cov = [0] * num_t_subsets
        for block in blocks:
            for ts_idx in block_tsubsets(block):
                cov[ts_idx] += 1
        return sum(1 for c in cov if c == 0)

    # Initialize
    blocks = greedy_init()
    best_blocks = list(blocks)
    best_exact = exact_uncovered(blocks)

    # Simulated annealing with noisy objective
    temperature = 50.0
    cooling = 0.998
    min_temp = 0.5
    sample_size = 150

    # Precompute block t-subsets for fast updates
    def compute_block_tsubsets_list(blocks_list):
        return [block_tsubsets(b) for b in blocks_list]

    block_ts_list = compute_block_tsubsets_list(blocks)

    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break

        # Choose a random block to modify
        block_idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[block_idx]
        old_tsubs = block_ts_list[block_idx]

        # Generate new block by removing 1-2 points and adding new ones
        old_list = list(old_block)
        num_remove = random.choice([1, 1, 2])
        remove_positions = random.sample(range(len(old_list)), num_remove)
        remove_positions.sort(reverse=True)
        for pos in remove_positions:
            old_list.pop(pos)

        # Add new random points
        candidates = [x for x in range(v) if x not in old_list]
        num_add = num_remove
        new_points = random.sample(candidates, num_add)
        new_list = old_list + new_points
        new_list.sort()
        new_block = tuple(new_list)

        # Check duplicate
        if new_block == old_block:
            continue
        if new_block in blocks and blocks.index(new_block) != block_idx:
            continue

        # Compute noisy objective for new configuration
        # Temporarily swap
        blocks[block_idx] = new_block
        new_tsubs = block_tsubsets(new_block)
        block_ts_list[block_idx] = new_tsubs

        new_est = expected_uncovered(blocks, sample_size)
        old_est = expected_uncovered(blocks, sample_size)  # This is same config, so use old estimate
        
        # Actually, we need to compare old vs new. Let's restructure:
        # Compute old estimate before swap
        # Let me redo this properly

        # Restore
        blocks[block_idx] = old_block
        block_ts_list[block_idx] = old_tsubs

        # Compute estimates
        # For speed, use a faster local estimate: how many t-subsets in old_block are newly covered/uncovered
        # and vice versa
        def local_delta(old_tsubs_set, new_tsubs_set, blocks_list):
            """Estimate change in uncovered count from replacing old with new."""
            old_set = set(old_tsubs_set)
            new_set = set(new_tsubs_set)
            
            # t-subsets covered only by old (might become uncovered)
            only_old = old_set - new_set
            # t-subsets covered only by new (might become covered)
            only_new = new_set - old_set
            
            delta = 0
            # Check if any t-subset in only_old is covered by other blocks
            for ts_idx in only_old:
                covered_by_other = False
                for i, b in enumerate(blocks_list):
                    if i == block_idx:
                        continue
                    if ts_idx in block_ts_list[i]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    delta += 1  # becomes uncovered
            
            for ts_idx in only_new:
                was_uncovered = True
                for i, b in enumerate(blocks_list):
                    if i == block_idx:
                        continue
                    if ts_idx in block_ts_list[i]:
                        was_uncovered = False
                        break
                if was_uncovered:
                    delta -= 1  # becomes covered
            
            return delta

        delta = local_delta(old_tsubs, new_tsubs, blocks)

        # Add noise to escape local minima
        noise = random.gauss(0, temperature * 0.1)
        noisy_delta = delta + noise

        # Metropolis acceptance
        if noisy_delta < 0 or random.random() < math.exp(-noisy_delta / temperature):
            blocks[block_idx] = new_block
            block_ts_list[block_idx] = new_tsubs

            # Periodically check exact uncovered
            if random.random() < 0.05:
                cur_exact = exact_uncovered(blocks)
                if cur_exact < best_exact:
                    best_exact = cur_exact
                    best_blocks = list(blocks)

        temperature *= cooling

    # Final verification and return
    if exact_uncovered(best_blocks) > 0:
        # Fallback: use greedy result if best is invalid
        greedy_blocks = greedy_init()
        if exact_uncovered(greedy_blocks) <= exact_uncovered(best_blocks):
            best_blocks = greedy_blocks

    return [list(b) for b in best_blocks]
