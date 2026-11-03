import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build incidence structure:
    # block_to_tsubsets[block_idx] = list of t-subset indices covered by this block
    # tsub_to_blocks[tsub_idx] = list of block indices that cover this t-subset
    block_to_tsubsets = []
    tsub_to_blocks = [[] for _ in range(N)]

    for bi, b in enumerate(all_blocks):
        covers = []
        for s in itertools.combinations(b, t):
            ti = t_idx[s]
            covers.append(ti)
            tsub_to_blocks[ti].append(bi)
        block_to_tsubsets.append(np.array(covers, dtype=np.int32))

    # Convert to numpy arrays for faster operations
    # For each block, store the set of t-subsets it covers as a boolean mask
    # But that's too large. Instead, we'll use sparse representation.

    def greedy_initial_solution(seed=0):
        """Fast greedy set cover to get an initial feasible solution."""
        random.seed(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        # Track which blocks have been selected
        used_blocks = set()

        while np.any(uncovered):
            if time.time() > deadline - 30:
                break

            # Pick a random uncovered t-subset to guide the search
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            target = random.choice(unc_idx)

            # Among blocks covering this target, pick the one covering most uncovered subsets
            candidates = tsub_to_blocks[target]
            best_bi = -1
            best_gain = 0

            for bi in candidates:
                if bi in used_blocks:
                    continue
                cov = block_to_tsubsets[bi]
                gain = np.sum(uncovered[cov])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if gain > 50:  # Early exit if good enough
                        break

            if best_bi == -1:
                break

            selected.append(best_bi)
            used_blocks.add(best_bi)
            uncovered[block_to_tsubsets[best_bi]] = False

        # Verify completeness
        if not np.all(~uncovered):
            # Fallback: brute force add remaining
            for ti in np.where(uncovered)[0]:
                for bi in tsub_to_blocks[ti]:
                    if bi not in used_blocks:
                        selected.append(bi)
                        used_blocks.add(bi)
                        break

        return selected

    def branch_and_bound(initial_solution, max_time=120.0):
        """
        Branch-and-bound ILP-style search for minimum set cover.
        Variables: x_b in {0,1} for each block b
        Minimize: sum(x_b)
        Subject to: for each t-subset s, sum_{b covers s} x_b >= 1
        """
        start_time = time.time()
        best_solution = list(initial_solution)
        best_size = len(best_solution)

        # Precompute: for each t-subset, the list of blocks that cover it
        # Already have tsub_to_blocks

        # Lower bound: use greedy fractional relaxation
        # For each uncovered t-subset, count how many available blocks cover it
        # LB = max over all t-subsets of (1 / max_blocks_covering_it) ... not tight
        # Better: solve a simple LP relaxation approximately

        def compute_lower_bound(selected_set):
            """
            Compute a lower bound on the number of additional blocks needed.
            Use: for each t-subset not yet covered, we need at least 1 block.
            But blocks can cover multiple. Use a simple packing bound.
            """
            # Count uncovered t-subsets
            # Build coverage count from selected blocks
            if not selected_set:
                return N  # Need at least N blocks if none selected (loose)

            # For each t-subset, count how many selected blocks cover it
            cover_count = np.zeros(N, dtype=np.int32)
            for bi in selected_set:
                cover_count[block_to_tsubsets[bi]] += 1

            uncovered_mask = cover_count == 0
            num_uncovered = np.sum(uncovered_mask)

            if num_uncovered == 0:
                return 0

            # Simple lower bound: each block can cover at most C(k,t) t-subsets
            # But we need a tighter bound. Use: for each uncovered t-subset,
            # the number of blocks that can cover it (excluding already selected)
            # LB >= max over uncovered s of (1 / num_available_blocks_covering_s)
            # This is weak. Use a better heuristic:
            # Greedy lower bound: simulate greedy and count steps

            # For now, use a simple bound: num_uncovered / max_coverage_per_block
            max_cov = max(len(block_to_tsubsets[bi]) for bi in range(num_blocks)) if num_blocks > 0 else 1
            return int(np.ceil(num_uncovered / max_cov))

        # Recursive branch-and-bound
        # State: selected_blocks (list), current_depth
        # We'll use a stack-based approach to avoid recursion depth issues

        # For efficiency, we'll do a limited-depth B&B with randomized variable ordering
        # Since the problem is large, we'll use a heuristic-guided search

        # Alternative: use a local search with simulated annealing on the block set
        # Start from greedy solution, try to remove blocks and repair

        def local_search_minimize(blocks, max_iter=10000):
            """
            Local search: iteratively try to remove a block and repair if needed.
            Or swap a block for another that allows further removal.
            """
            current = list(blocks)
            current_size = len(current)
            best = list(current)
            best_size = current_size

            # Build cover count
            cover_count = np.zeros(N, dtype=np.int32)
            for bi in current:
                cover_count[block_to_tsubsets[bi]] += 1

            for iteration in range(max_iter):
                if time.time() > start_time + max_time:
                    break

                # Try to remove a random block
                idx = random.randint(0, len(current) - 1)
                bi_to_remove = current[idx]

                # Check if removal breaks coverage
                cov = block_to_tsubsets[bi_to_remove]
                would_break = False
                for ti in cov:
                    if cover_count[ti] <= 1:
                        would_break = True
                        break

                if not would_break:
                    # Safe to remove
                    cover_count[cov] -= 1
                    current.pop(idx)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best = list(current)
                    continue

                # Try to find a replacement that allows removal
                # Find t-subsets that would become uncovered
                uncovered_by_removal = [ti for ti in cov if cover_count[ti] == 1]

                if not uncovered_by_removal:
                    continue

                # Find blocks that cover all of uncovered_by_removal
                # and are not already in current
                current_set = set(current)
                candidates = set(tsub_to_blocks[uncovered_by_removal[0]])
                for ti in uncovered_by_removal[1:]:
                    candidates &= set(tsub_to_blocks[ti])
                candidates -= current_set

                if not candidates:
                    continue

                # Pick the candidate that maximizes "flexibility"
                # i.e., covers the most t-subsets that are covered by exactly 1 block
                best_candidate = -1
                best_score = -1
                for cand in list(candidates)[:20]:  # Limit candidates for speed
                    new_cov = block_to_tsubsets[cand]
                    score = 0
                    for ti in new_cov:
                        if cover_count[ti] == 1 and ti != bi_to_remove:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                if best_candidate != -1:
                    # Perform swap: remove bi_to_remove, add best_candidate
                    cover_count[block_to_tsubsets[bi_to_remove]] -= 1
                    cover_count[block_to_tsubsets[best_candidate]] += 1
                    current[idx] = best_candidate
                    # After swap, try to remove again in next iteration

            return best, best_size

        # Run multiple local search instances with different starting points
        for attempt in range(5):
            if time.time() > start_time + max_time:
                break

            # Get a greedy solution with different seed
            greedy_sol = greedy_initial_solution(seed=attempt * 7 + 1)
            if len(greedy_sol) < best_size:
                best_size = len(greedy_sol)
                best_solution = list(greedy_sol)

            # Run local search on this solution
            improved, improved_size = local_search_minimize(greedy_sol, max_iter=5000)
            if improved_size < best_size:
                best_size = improved_size
                best_solution = improved

        return best_solution

    # Phase 1: Get initial greedy solution
    initial_sol = greedy_initial_solution(seed=42)
    if not initial_sol:
        # Fallback: return all blocks (should not happen)
        return [list(b) for b in all_blocks]

    # Phase 2: Branch-and-bound / local search optimization
    optimized_sol = branch_and_bound(initial_sol, max_time=110.0)

    # Convert back to point lists
    final_blocks = [list(all_blocks[bi]) for bi in optimized_sol]

    return final_blocks
