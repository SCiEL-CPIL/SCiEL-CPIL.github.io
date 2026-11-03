import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)

    # Precompute: for each block, which t-subsets it covers (as bitmask)
    block_cover_mask = []
    for b in all_blocks:
        mask = 0
        for s in itertools.combinations(b, t):
            mask |= (1 << t_to_idx[s])
        block_cover_mask.append(mask)

    # For each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, mask in enumerate(block_cover_mask):
        m = mask
        while m:
            lb = m & (-m)
            ti = lb.bit_length() - 1
            t_to_blocks[ti].append(bi)
            m ^= lb

    full_mask = (1 << N_t) - 1

    def fast_greedy(rng):
        """
        Greedy set cover with randomization using bitmask coverage.
        """
        uncovered = full_mask
        selected = []

        while uncovered:
            # Pick a random uncovered t-subset
            bits = uncovered
            ti_target = (bits & (-bits)).bit_length() - 1
            # Actually pick a random set bit
            # Use a random walk: pick a random position and check
            while True:
                pos = rng.randrange(N_t)
                if (uncovered >> pos) & 1:
                    ti_target = pos
                    break

            candidates = t_to_blocks[ti_target]
            if not candidates:
                break

            # Score each candidate by gain (number of newly covered t-subsets)
            best_block = -1
            best_gain = -1

            # For speed, sample if too many candidates
            if len(candidates) > 150:
                sample = rng.sample(candidates, 150)
            else:
                sample = candidates

            for bi in sample:
                gain = bin(uncovered & block_cover_mask[bi]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi

            if best_block == -1:
                best_block = candidates[0]

            selected.append(best_block)
            uncovered &= ~block_cover_mask[best_block]

        return selected

    def prune(blocks):
        """Remove redundant blocks using bitmask coverage."""
        if not blocks:
            return []

        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Compute coverage counts per t-subset
            cov_counts = [0] * N_t
            for bi in current_blocks:
                m = block_cover_mask[bi]
                while m:
                    lb = m & (-m)
                    ti = lb.bit_length() - 1
                    cov_counts[ti] += 1
                    m ^= lb

            new_blocks = []
            for bi in current_blocks:
                # Check if any t-subset is uniquely covered by this block
                m = block_cover_mask[bi]
                unique = False
                while m:
                    lb = m & (-m)
                    ti = lb.bit_length() - 1
                    if cov_counts[ti] == 1:
                        unique = True
                        break
                    m ^= lb
                if unique:
                    new_blocks.append(bi)
                else:
                    # Remove this block's contribution
                    m = block_cover_mask[bi]
                    while m:
                        lb = m & (-m)
                        ti = lb.bit_length() - 1
                        cov_counts[ti] -= 1
                        m ^= lb
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    def local_search(blocks, rng, time_limit):
        """
        Local search: try to remove or swap blocks using bitmask.
        """
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks

        # Build coverage counts
        cov_counts = [0] * N_t
        for bi in current_blocks:
            m = block_cover_mask[bi]
            while m:
                lb = m & (-m)
                ti = lb.bit_length() - 1
                cov_counts[ti] += 1
                m ^= lb

        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        no_improve_count = 0
        max_no_improve = 50

        def is_redundant(bi):
            m = block_cover_mask[bi]
            while m:
                lb = m & (-m)
                ti = lb.bit_length() - 1
                if cov_counts[ti] == 1:
                    return False
                m ^= lb
            return True

        def remove_block(bi):
            m = block_cover_mask[bi]
            while m:
                lb = m & (-m)
                ti = lb.bit_length() - 1
                cov_counts[ti] -= 1
                m ^= lb

        def add_block(bi):
            m = block_cover_mask[bi]
            while m:
                lb = m & (-m)
                ti = lb.bit_length() - 1
                cov_counts[ti] += 1
                m ^= lb

        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                break

            idx = rng.randrange(len(current_blocks))
            bi = current_blocks[idx]

            # Try to remove block bi if it's redundant
            if is_redundant(bi):
                current_blocks.pop(idx)
                remove_block(bi)
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue

            # Try to swap block bi with another block
            # Find t-subsets uniquely covered by bi
            m = block_cover_mask[bi]
            unique_ts = []
            while m:
                lb = m & (-m)
                ti = lb.bit_length() - 1
                if cov_counts[ti] == 1:
                    unique_ts.append(ti)
                m ^= lb

            if not unique_ts:
                no_improve_count += 1
                continue

            # Find blocks that cover all unique_ts
            candidates = set(t_to_blocks[unique_ts[0]])
            for ti in unique_ts[1:]:
                candidates &= set(t_to_blocks[ti])
            candidates.discard(bi)

            if not candidates:
                no_improve_count += 1
                continue

            cand = rng.choice(list(candidates))

            # Check if swap is valid: after removing bi and adding cand, no t-subset uncovered
            # Temporarily remove bi
            m_bi = block_cover_mask[bi]
            m_cand = block_cover_mask[cand]
            # New coverage for each t-subset in bi's coverage:
            # old_count - 1 + (1 if in cand else 0)
            # We need old_count - 1 + (1 if in cand) >= 1 for all ti in bi's coverage
            # i.e., old_count + (1 if in cand) >= 2
            # i.e., old_count >= 1 and if old_count == 1 then ti must be in cand
            valid = True
            m_check = m_bi
            while m_check:
                lb = m_check & (-m_check)
                ti = lb.bit_length() - 1
                if cov_counts[ti] == 1:
                    # Must be covered by cand
                    if not ((m_cand >> ti) & 1):
                        valid = False
                        break
                m_check ^= lb

            if not valid:
                no_improve_count += 1
                continue

            # Perform swap
            remove_block(bi)
            add_block(cand)
            current_blocks[idx] = cand

            # Try to remove any newly redundant blocks
            new_blocks = []
            for b in current_blocks:
                if is_redundant(b):
                    remove_block(b)
                else:
                    new_blocks.append(b)

            current_blocks = new_blocks
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
                no_improve_count = 0
            else:
                no_improve_count += 1

        return best_blocks

    def repair_cover(blocks, rng, time_limit):
        """
        Repair a partial cover by adding blocks to cover all t-subsets.
        """
        # Compute uncovered
        covered_mask = 0
        for bi in blocks:
            covered_mask |= block_cover_mask[bi]
        uncovered = full_mask & ~covered_mask

        current_blocks = list(blocks)
        while uncovered and time.time() < time_limit:
            # Pick a random uncovered t-subset
            while True:
                pos = rng.randrange(N_t)
                if (uncovered >> pos) & 1:
                    ti_target = pos
                    break

            candidates = t_to_blocks[ti_target]
            if not candidates:
                break

            best_block = -1
            best_gain = -1
            if len(candidates) > 100:
                sample = rng.sample(candidates, 100)
            else:
                sample = candidates

            for bi in sample:
                gain = bin(uncovered & block_cover_mask[bi]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi

            if best_block == -1:
                break

            current_blocks.append(best_block)
            uncovered &= ~block_cover_mask[best_block]

        return current_blocks

    def genetic_algorithm(rng, time_limit):
        """
        Genetic algorithm with population of covering sets.
        """
        pop_size = 6
        best_blocks = None
        best_len = float('inf')

        # Initialize population with greedy solutions
        population = []
        for i in range(pop_size):
            blocks = fast_greedy(rng)
            blocks = prune(blocks)
            population.append(blocks)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)

        generation = 0
        while time.time() < time_limit:
            generation += 1

            # Selection: keep top half
            population.sort(key=lambda x: len(x))
            survivors = population[:pop_size // 2]

            # Crossover and mutation to fill population
            new_gen = list(survivors)

            while len(new_gen) < pop_size and time.time() < time_limit:
                # Crossover: pick two parents, combine blocks
                p1 = rng.choice(survivors)
                p2 = rng.choice(survivors)

                # Combine: take blocks from both parents, repair
                combined = list(p1)
                # Add some blocks from p2 that might help
                for bi in p2:
                    if rng.random() < 0.3 and bi not in combined:
                        combined.append(bi)

                # Repair coverage
                combined = repair_cover(combined, rng, time_limit)
                combined = prune(combined)

                if len(combined) < best_len:
                    best_len = len(combined)
                    best_blocks = list(combined)

                new_gen.append(combined)

            # Mutation: remove random block and repair
            if time.time() < time_limit:
                parent = rng.choice(new_gen)
                mutated = list(parent)
                if len(mutated) > 1:
                    idx = rng.randrange(len(mutated))
                    mutated.pop(idx)
                    mutated = repair_cover(mutated, rng, time_limit)
                    mutated = prune(mutated)
                    if len(mutated) < best_len:
                        best_len = len(mutated)
                        best_blocks = list(mutated)
                    new_gen[-1] = mutated

            population = new_gen

            if generation % 5 == 0:
                # Occasionally inject fresh greedy solutions
                if time.time() < time_limit - 10:
                    fresh = fast_greedy(rng)
                    fresh = prune(fresh)
                    if len(fresh) < best_len:
                        best_len = len(fresh)
                        best_blocks = list(fresh)
                    population[-1] = fresh

        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')

    # Phase 1: Multi-start Greedy
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 50.0:
            break
        rng = random.Random(123 + i * 7)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = list(blocks)

    if not best_solution:
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(best_solution)
        best_len = len(best_solution)

    # Phase 2: Local Search
    if best_solution and time.time() < deadline - 20.0:
        rng = random.Random(999)
        optimized = local_search(best_solution, rng, deadline - 15.0)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized

    # Phase 3: Genetic Algorithm for further improvement
    if best_solution and time.time() < deadline - 10.0:
        rng = random.Random(42)
        ga_result = genetic_algorithm(rng, deadline - 5.0)
        if ga_result and len(ga_result) < best_len:
            best_len = len(ga_result)
            best_solution = ga_result

    # Final local search on best solution
    if best_solution and time.time() < deadline - 3.0:
        rng = random.Random(777)
        final_opt = local_search(best_solution, rng, deadline - 1.0)
        if len(final_opt) < best_len:
            best_len = len(final_opt)
            best_solution = final_opt

    return [list(all_blocks[bi]) for bi in best_solution]
