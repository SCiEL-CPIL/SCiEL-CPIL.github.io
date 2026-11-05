import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # For each k-block, store the t-subset indices it covers
    block_t_indices = []
    for block in all_k_blocks:
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)

    # Reverse mapping: for each t-subset, which k-blocks cover it
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx in range(num_k_blocks):
        for idx in block_t_indices[b_idx]:
            t_subset_to_blocks[idx].append(b_idx)

    # Precompute hardness: 1/num_covering_blocks for each t-subset
    hardness = np.array([1.0 / max(1, len(t_subset_to_blocks[i])) for i in range(num_t_subsets)], dtype=np.float64)

    best_blocks = None
    best_score = float('inf')

    # Use fewer restarts but more aggressive local search per restart
    num_restarts = 12

    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break

        random.seed(restart * 137 + 42)
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []

        # Probabilistic greedy phase with weighted sampling
        uncovered_count = num_t_subsets
        uncovered_set = set(range(num_t_subsets))

        # Temperature for sampling: lower = more deterministic, higher = more random
        temperature = 1.5

        while uncovered_count > 0:
            if time.time() > deadline - 15.0:
                break

            # Pick a random uncovered t-subset
            if len(uncovered_set) < 500:
                u_idx = random.choice(list(uncovered_set))
            else:
                while True:
                    candidate = random.randint(0, num_t_subsets - 1)
                    if coverage[candidate] == 0:
                        u_idx = candidate
                        break

            # Get candidate blocks that cover u_idx
            candidates = t_subset_to_blocks[u_idx]

            # Compute weights for probabilistic sampling
            weights = []
            for b_idx in candidates:
                idxs = block_t_indices[b_idx]
                mask = coverage[idxs] == 0
                gain = int(np.count_nonzero(mask))
                if gain == 0:
                    weights.append(0.0)
                    continue
                # Hardness bonus: prefer blocks covering hard-to-cover subsets
                hard_bonus = float(np.sum(hardness[idxs[mask]]))
                # Weight proportional to gain^alpha * hardness_factor
                alpha = 1.2
                w = (gain ** alpha) * (1.0 + 0.3 * hard_bonus)
                weights.append(w)

            # Filter out zero-weight candidates
            valid_candidates = [(b_idx, w) for b_idx, w in zip(candidates, weights) if w > 0]
            if not valid_candidates:
                # Fallback: pick first candidate
                best_block_idx = candidates[0]
            else:
                # Weighted sampling with temperature
                total_w = sum(w for _, w in valid_candidates)
                if total_w == 0:
                    best_block_idx = valid_candidates[0][0]
                else:
                    # Apply temperature: w' = w^(1/temperature)
                    probs = [(w ** (1.0 / temperature)) / total_w for _, w in valid_candidates]
                    # Normalize
                    total_p = sum(probs)
                    probs = [p / total_p for p in probs]
                    # Sample
                    r = random.random()
                    cumsum = 0.0
                    best_block_idx = valid_candidates[-1][0]
                    for (b_idx, _), p in zip(valid_candidates, probs):
                        cumsum += p
                        if r <= cumsum:
                            best_block_idx = b_idx
                            break

            # Add the selected block
            block_indices.append(best_block_idx)

            # Update coverage
            idxs = block_t_indices[best_block_idx]
            newly_covered = idxs[coverage[idxs] == 0]
            coverage[idxs] += 1
            uncovered_count -= len(newly_covered)

            for idx in newly_covered:
                uncovered_set.discard(idx)

        # Dynamic repair phase: identify most frequently uncovered 5-subsets
        # and force blocks containing them
        if uncovered_count > 0 and time.time() < deadline - 10.0:
            # Count how many times each uncovered t-subset has been "missed"
            # by counting how many blocks were added that COULD have covered it
            miss_count = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in block_indices:
                idxs = block_t_indices[b_idx]
                for idx in idxs:
                    if coverage[idx] == 0:
                        miss_count[idx] += 1

            # Sort uncovered subsets by miss count (descending)
            uncovered_list = list(uncovered_set)
            uncovered_list.sort(key=lambda x: -miss_count[x])

            # Process most-missed subsets first
            for u_idx in uncovered_list:
                if time.time() > deadline - 10.0:
                    break
                if coverage[u_idx] > 0:
                    continue  # Already covered

                candidates = t_subset_to_blocks[u_idx]
                # Pick block with maximum gain among candidates
                best_gain = -1
                best_block_idx = candidates[0]
                for b_idx in candidates:
                    idxs = block_t_indices[b_idx]
                    gain = int(np.count_nonzero(coverage[idxs] == 0))
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx

                block_indices.append(best_block_idx)
                idxs = block_t_indices[best_block_idx]
                newly_covered = idxs[coverage[idxs] == 0]
                coverage[idxs] += 1
                uncovered_count -= len(newly_covered)
                for idx in newly_covered:
                    uncovered_set.discard(idx)

                if uncovered_count == 0:
                    break

        # Aggressive reduction phase with local search
        # Phase 1: Remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 12.0:
            changed = False
            for i in range(len(block_indices) - 1, -1, -1):
                if time.time() > deadline - 12.0:
                    break
                b_idx = block_indices[i]
                idxs = block_t_indices[b_idx]
                if np.any(coverage[idxs] == 1):
                    continue
                coverage[idxs] -= 1
                block_indices.pop(i)
                changed = True

        # Phase 2: Local search - try to replace pairs of blocks with one
        # or swap blocks to reduce total count
        improved = True
        max_ls_iterations = 500
        ls_iter = 0
        while improved and ls_iter < max_ls_iterations and time.time() < deadline - 10.0:
            improved = False
            ls_iter += 1
            random.shuffle(block_indices)
            for i in range(len(block_indices)):
                if time.time() > deadline - 10.0:
                    break
                b_idx = block_indices[i]
                idxs = block_t_indices[b_idx]
                
                # Check if this block is essential
                if np.any(coverage[idxs] == 1):
                    continue
                
                # Try removing this block and see if we can add one block that covers all its unique t-subsets
                coverage[idxs] -= 1
                block_indices.pop(i)
                
                # Find all t-subsets that are now uncovered
                newly_uncovered = idxs[coverage[idxs] == 0]
                
                if len(newly_uncovered) == 0:
                    # Block was redundant, keep it removed
                    improved = True
                    continue
                
                # Try to find a single block that covers all newly_uncovered
                if len(newly_uncovered) <= 20:
                    # Check if any block covers all these
                    found = False
                    newly_uncovered_set = set(newly_uncovered.tolist())
                    for u_idx in newly_uncovered:
                        for cand in t_subset_to_blocks[u_idx]:
                            cand_idxs = block_t_indices[cand]
                            cand_idxs_set = set(cand_idxs.tolist())
                            if newly_uncovered_set.issubset(cand_idxs_set):
                                coverage[cand_idxs] += 1
                                block_indices.append(cand)
                                found = True
                                improved = True
                                break
                        if found:
                            break
                    
                    if not found:
                        # Restore the removed block
                        coverage[idxs] += 1
                        block_indices.insert(i, b_idx)
                else:
                    # Too many to cover with one block, restore
                    coverage[idxs] += 1
                    block_indices.insert(i, b_idx)

        # Phase 3: Final cleanup - remove any remaining redundant blocks
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            for i in range(len(block_indices) - 1, -1, -1):
                if time.time() > deadline - 8.0:
                    break
                b_idx = block_indices[i]
                idxs = block_t_indices[b_idx]
                if np.any(coverage[idxs] == 1):
                    continue
                coverage[idxs] -= 1
                block_indices.pop(i)
                changed = True

        if len(block_indices) < best_score:
            best_score = len(block_indices)
            best_blocks = [list(all_k_blocks[bi]) for bi in block_indices]

    return best_blocks
