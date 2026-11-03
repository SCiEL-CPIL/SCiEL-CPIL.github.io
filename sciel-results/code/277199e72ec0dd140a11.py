"""Probabilistic covering design using inclusion-exclusion-inspired noisy objective + SA for C(19,8,5)."""
import random
import itertools
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using:
    1. Greedy initialization
    2. Simulated annealing with a noisy probabilistic objective (inclusion-exclusion inspired)
    3. Final repair phase
    """
    start_time = time.time()
    timeout = 115

    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    def block_tsubsets(block):
        return frozenset(t_subset_index[ts] for ts in itertools.combinations(block, t))

    # --- Greedy initialization ---
    def greedy_init():
        cov = [0] * num_t_subsets
        blocks = []
        uncovered = set(range(num_t_subsets))

        while uncovered:
            best_block = None
            best_gain = -1

            for _ in range(250):
                if not uncovered:
                    break
                unc_idx = random.choice(tuple(uncovered))
                seed = list(all_t_subsets[unc_idx])
                block = list(seed)
                random.shuffle(block)

                while len(block) < k:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    best_x = None
                    best_local = -1
                    for x in candidates[:12]:
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_subset_index.get(ts)
                            if idx is not None and idx in uncovered:
                                gain += 1
                        if gain > best_local:
                            best_local = gain
                            best_x = x
                    if best_x is not None:
                        block.append(best_x)

                block.sort()
                bt = tuple(block)

                gain = sum(1 for ts in itertools.combinations(bt, t)
                          if t_subset_index.get(ts) in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = bt

            if best_block is None:
                break

            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                idx = t_subset_index[ts]
                cov[idx] += 1
                if cov[idx] == 1:
                    uncovered.discard(idx)

        return blocks

    blocks = greedy_init()
    cov = [0] * num_t_subsets
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov[t_subset_index[ts]] += 1

    current_uncovered = sum(1 for c in cov if c == 0)
    best_blocks = list(blocks)
    best_uncovered = current_uncovered

    # --- Noisy probabilistic objective ---
    # For each t-subset i, let c_i be its coverage count.
    # P(i is uncovered) = 0 if c_i > 0, else 1. But we add noise:
    # E[uncovered] estimated by sampling: for each sampled subset, indicator(c_i == 0)
    # This is unbiased but noisy, helping SA escape local minima.
    def noisy_objective(cov, sample_frac=0.2):
        n_sample = max(1, int(num_t_subsets * sample_frac))
        count = 0
        for _ in range(n_sample):
            idx = random.randint(0, num_t_subsets - 1)
            if cov[idx] == 0:
                count += 1
        return count / sample_frac

    def exact_uncovered(cov):
        return sum(1 for c in cov if c == 0)

    # --- Simulated Annealing ---
    temperature = 25.0
    cooling = 0.9992
    min_temp = 0.3
    use_noisy = True
    noisy_frac = 0.15

    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break

        block_idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[block_idx]
        old_tsubs = block_tsubsets(old_block)

        # Remove old block
        for idx in old_tsubs:
            cov[idx] -= 1

        # Generate new block
        uncovered_list = [i for i, c in enumerate(cov) if c == 0]
        if uncovered_list:
            target_idx = random.choice(uncovered_list)
            seed = list(all_t_subsets[target_idx])
            block = list(seed)
            random.shuffle(block)
        else:
            block = list(old_block)
            random.shuffle(block)

        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            best_x = None
            best_local = -1
            for x in candidates[:10]:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = t_subset_index.get(ts)
                    if idx is not None and cov[idx] == 0:
                        gain += 1
                if gain > best_local:
                    best_local = gain
                    best_x = x
            if best_x is not None:
                block.append(best_x)

        block.sort()
        new_block = tuple(block)

        # Check duplicate
        if new_block != old_block:
            dup = any(i != block_idx and b == new_block for i, b in enumerate(blocks))
            if dup:
                for idx in old_tsubs:
                    cov[idx] += 1
                continue

        new_tsubs = block_tsubsets(new_block)
        for idx in new_tsubs:
            cov[idx] += 1

        # Compute delta: use noisy objective for acceptance decision
        if use_noisy:
            new_obj = noisy_objective(cov, noisy_frac)
            # We need old_obj, but we already modified cov. 
            # Instead, compute delta in exact terms for reliability,
            # but add noise to the acceptance probability.
            exact_new = exact_uncovered(cov)
            delta = exact_new - current_uncovered
            # Add stochastic perturbation to delta to create noise
            delta_noisy = delta + random.gauss(0, 0.5)
            if delta_noisy <= 0 or random.random() < math.exp(-delta_noisy / temperature):
                blocks[block_idx] = new_block
                current_uncovered = exact_new
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = list(blocks)
            else:
                for idx in new_tsubs:
                    cov[idx] -= 1
                for idx in old_tsubs:
                    cov[idx] += 1
        else:
            new_uncovered = exact_uncovered(cov)
            delta = new_uncovered - current_uncovered
            if delta <= 0 or random.random() < math.exp(-delta / temperature):
                blocks[block_idx] = new_block
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = list(blocks)
            else:
                for idx in new_tsubs:
                    cov[idx] -= 1
                for idx in old_tsubs:
                    cov[idx] += 1

        temperature *= cooling

    # --- Final repair ---
    if best_uncovered > 0 and time.time() - start_time < timeout - 2:
        cov_final = [0] * num_t_subsets
        for b in best_blocks:
            for ts in itertools.combinations(b, t):
                cov_final[t_subset_index[ts]] += 1

        uncovered_final = [i for i, c in enumerate(cov_final) if c == 0]
        for _ in range(150):
            if time.time() - start_time > timeout - 1 or not uncovered_final:
                break
            target = random.choice(uncovered_final)
            seed = list(all_t_subsets[target])
            block = list(seed)
            random.shuffle(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                best_x = None
                best_local = -1
                for x in candidates[:8]:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_index.get(ts)
                        if idx is not None and cov_final[idx] == 0:
                            gain += 1
                    if gain > best_local:
                        best_local = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
            block.sort()
            nb = tuple(block)
            if nb not in best_blocks:
                best_blocks.append(nb)
                for ts in itertools.combinations(nb, t):
                    cov_final[t_subset_index[ts]] += 1
                uncovered_final = [i for i, c in enumerate(cov_final) if c == 0]

    return [list(b) for b in best_blocks]
