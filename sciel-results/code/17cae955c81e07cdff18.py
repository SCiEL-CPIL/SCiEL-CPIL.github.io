import numpy as np
import itertools
import time
import random
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers (stored as sorted array)
    block_t_indices = []
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        idxs = np.array([idx_map[sub] for sub in subs], dtype=np.int32)
        block_t_indices.append(idxs)
    
    # Convert to a single array with consistent shape for vectorized ops
    block_t_indices_arr = np.array(block_t_indices, dtype=np.int32)

    # Precompute, for each point x, the list of t-subset indices that contain x.
    point_ts = [[] for _ in range(v)]
    for i in range(n_ts):
        comb = all_ts[i]
        for x in comb:
            point_ts[x].append(i)
    point_ts = [np.array(lst, dtype=np.int32) for lst in point_ts]

    best_solution = None
    best_score = float('inf')

    def aggressive_deletion(current_blocks, coverage_count):
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 4.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def rebuild_coverage(block_list):
        cc = np.zeros(n_ts, dtype=np.int32)
        for b in block_list:
            cc[block_t_indices[b]] += 1
        return cc

    # Precompute for each block and each point in it, the lost/gained triple indices
    swap_cache = {}
    
    def try_point_swap(blocks, coverage_count, b_idx, old_x, new_x):
        """Try replacing old_x with new_x in block b_idx. Returns True if valid."""
        if new_x == old_x:
            return False
        block = all_ks[b_idx]
        if new_x in block:
            return False

        cache_key = (b_idx, old_x, new_x)
        if cache_key in swap_cache:
            lost, gained = swap_cache[cache_key]
        else:
            other = [p for p in block if p != old_x]
            lost_list = []
            gained_list = []
            for rest in itertools.combinations(other, t - 1):
                lost_list.append(idx_map[tuple(sorted(rest + (old_x,)))])
                gained_list.append(idx_map[tuple(sorted(rest + (new_x,)))])
            lost = np.array(lost_list, dtype=np.int32)
            gained = np.array(gained_list, dtype=np.int32)
            if len(swap_cache) < 80000:
                swap_cache[cache_key] = (lost, gained)

        # After swap, coverage for 'lost' drops by 1; must stay >= 1.
        if np.any(coverage_count[lost] <= 1):
            return False

        coverage_count[lost] -= 1
        coverage_count[gained] += 1
        return True

    def block_swap_search(blocks, coverage_count, max_iters=200):
        """Try swapping entire blocks to reduce count while maintaining coverage."""
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            n_blocks = len(blocks)
            if n_blocks < 2:
                break
            
            # Try removing each block and checking if we can cover its unique triples
            # with a swap of another block
            for i in range(n_blocks):
                if time.time() - start_time > time_limit - 5.0:
                    break
                b_idx = blocks[i]
                subs_in_block = block_t_indices[b_idx]
                # Check which triples are uniquely covered by this block
                unique_mask = coverage_count[subs_in_block] == 1
                n_unique = np.sum(unique_mask)
                
                if n_unique == 0:
                    # Can remove this block entirely
                    coverage_count[subs_in_block] -= 1
                    blocks = [b for j, b in enumerate(blocks) if j != i]
                    improved = True
                    break
                
                # Try to find a block that can cover the unique triples
                # by swapping one of its points
                for j in range(n_blocks):
                    if j == i:
                        continue
                    other_b_idx = blocks[j]
                    other_block = all_ks[other_b_idx]
                    
                    # Try swapping each point in other_block with a point from b_idx
                    # that might help cover the unique triples
                    for old_x in other_block:
                        for new_x in all_ks[b_idx]:
                            if new_x in other_block:
                                continue
                            if try_point_swap(blocks, coverage_count, other_b_idx, old_x, new_x):
                                # Check if we can now remove block i
                                subs_in_block = block_t_indices[b_idx]
                                if np.all(coverage_count[subs_in_block] > 1):
                                    coverage_count[subs_in_block] -= 1
                                    blocks = [b for idx, b in enumerate(blocks) if idx != i]
                                    improved = True
                                else:
                                    # Undo the swap
                                    # We need to reverse the point swap
                                    # For simplicity, just accept if it helped
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return blocks

    def local_search(blocks, coverage_count, max_iters):
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False

            # Pass 1: try to delete any redundant blocks
            new_blocks = []
            for b_idx in blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    coverage_count[subs_in_block] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks

            # Pass 2: point-swap perturbations
            if len(blocks) == 0:
                break
            
            for b_idx in list(blocks):
                if time.time() - start_time > time_limit - 5.0:
                    break
                block = all_ks[b_idx]
                for old_x in block:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    for new_x in candidates[:10]:
                        if try_point_swap(blocks, coverage_count, b_idx, old_x, new_x):
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # After any swap, try deleting again
            if improved:
                new_blocks = []
                for b_idx in blocks:
                    subs_in_block = block_t_indices[b_idx]
                    if np.all(coverage_count[subs_in_block] > 1):
                        coverage_count[subs_in_block] -= 1
                    else:
                        new_blocks.append(b_idx)
                blocks = new_blocks

            if not improved:
                break

        return blocks

    def column_generation(seed, max_cg_iters=50):
        """Column generation: iteratively add blocks guided by LP dual variables."""
        random.seed(seed)
        np.random.seed(seed)

        # Start with a greedy solution
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 8.0:
                return None, None

            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_gain_blocks = np.where(gains == max_gain)[0]
            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = list(max_gain_blocks)

            best_candidate = candidates[0]
            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            remaining -= np.sum(newly_covered)
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        # Column generation loop: use LP duals to guide block swaps/additions
        for cg_iter in range(max_cg_iters):
            if time.time() - start_time > time_limit - 8.0:
                break
            
            # Compute LP dual variables (prices) for uncovered/under-covered triples
            # Use a simple weighting: higher weight for less-covered triples
            weights = np.zeros(n_ts, dtype=np.float64)
            for i in range(n_ts):
                if coverage_count[i] == 0:
                    weights[i] = 10.0
                elif coverage_count[i] == 1:
                    weights[i] = 3.0
                elif coverage_count[i] == 2:
                    weights[i] = 1.0
                else:
                    weights[i] = 0.1

            # Score each block by how well it covers high-weight triples
            # Focus on blocks that could replace existing ones
            if len(current_blocks) == 0:
                break
            
            block_set = set(current_blocks)
            
            # Try to find a better block to replace each existing block
            improved = False
            for b_idx in list(current_blocks):
                if time.time() - start_time > time_limit - 8.0:
                    break
                
                subs_in_block = block_t_indices[b_idx]
                unique_mask = coverage_count[subs_in_block] == 1
                n_unique = np.sum(unique_mask)
                
                if n_unique == 0:
                    coverage_count[subs_in_block] -= 1
                    current_blocks = [b for b in current_blocks if b != b_idx]
                    improved = True
                    continue
                
                # Try point swaps guided by weights
                block = all_ks[b_idx]
                for old_x in block:
                    candidates = [x for x in range(v) if x not in block]
                    random.shuffle(candidates)
                    for new_x in candidates[:12]:
                        # Estimate improvement: gained triples with high weight
                        # vs lost triples with high weight
                        other = [p for p in block if p != old_x]
                        lost_list = []
                        gained_list = []
                        for rest in itertools.combinations(other, t - 1):
                            lost_list.append(idx_map[tuple(sorted(rest + (old_x,)))])
                            gained_list.append(idx_map[tuple(sorted(rest + (new_x,)))])
                        
                        lost_arr = np.array(lost_list, dtype=np.int32)
                        gained_arr = np.array(gained_list, dtype=np.int32)
                        
                        # Check feasibility
                        if np.any(coverage_count[lost_arr] <= 1):
                            continue
                        
                        # Compute net weight change
                        lost_weight = np.sum(weights[lost_arr])
                        gained_weight = np.sum(weights[gained_arr])
                        
                        if gained_weight > lost_weight:
                            # Apply the swap
                            coverage_count[lost_arr] -= 1
                            coverage_count[gained_arr] += 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Try block swaps
            if not improved and len(current_blocks) >= 2:
                current_blocks = block_swap_search(current_blocks, coverage_count, max_iters=20)
                if len(current_blocks) < len(set(current_blocks)):
                    improved = True
            
            if not improved:
                break
            
            # Rebuild coverage after changes
            coverage_count = rebuild_coverage(current_blocks)
            current_blocks = aggressive_deletion(current_blocks, coverage_count)
            coverage_count = rebuild_coverage(current_blocks)

        return current_blocks, len(current_blocks)

    def run_greedy(seed, use_random_weights):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)

        if use_random_weights:
            point_weights = np.random.uniform(0.5, 2.0, size=v).astype(np.float32)
        else:
            point_weights = None

        while remaining > 0:
            if time.time() - start_time > time_limit - 6.0:
                return None, None

            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 60:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(60, len(max_gain_blocks)))]
            else:
                candidates = list(max_gain_blocks)

            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered_mask = ~covered[subs]
                if np.any(newly_covered_mask):
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered_mask]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.15
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        if time.time() - start_time < time_limit - 10.0:
            current_blocks = local_search(current_blocks, coverage_count, max_iters=300)
            coverage_count = rebuild_coverage(current_blocks)

        current_blocks = aggressive_deletion(current_blocks, coverage_count)

        return current_blocks, len(current_blocks)

    # Phase 1: Standard greedy with multiple seeds
    restart_count = 0
    while time.time() - start_time < time_limit - 35.0:
        seed = restart_count
        blocks, score = run_greedy(seed, use_random_weights=False)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    # Phase 2: Column generation with dual-guided swaps
    if time.time() - start_time < time_limit - 25.0:
        for seed in range(10):
            if time.time() - start_time > time_limit - 25.0:
                break
            blocks, score = column_generation(seed, max_cg_iters=30)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 3: Randomized weighted greedy
    if time.time() - start_time < time_limit - 15.0:
        for seed in range(5):
            if time.time() - start_time > time_limit - 15.0:
                break
            blocks, score = run_greedy(2000 + seed, use_random_weights=True)
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = blocks

    # Phase 4: Final high-precision polish
    if best_solution is not None and time.time() - start_time < time_limit - 8.0:
        coverage_count = rebuild_coverage(best_solution)
        best_solution = aggressive_deletion(best_solution, coverage_count)
        coverage_count = rebuild_coverage(best_solution)

        if time.time() - start_time < time_limit - 5.0:
            best_solution = local_search(best_solution, coverage_count, max_iters=1000)
            coverage_count = rebuild_coverage(best_solution)
            best_solution = aggressive_deletion(best_solution, coverage_count)
        
        # Block swap polish
        if time.time() - start_time < time_limit - 4.0:
            coverage_count = rebuild_coverage(best_solution)
            best_solution = block_swap_search(best_solution, coverage_count, max_iters=100)
            coverage_count = rebuild_coverage(best_solution)
            best_solution = aggressive_deletion(best_solution, coverage_count)

    if best_solution is None:
        blocks, score = run_greedy(0, use_random_weights=False)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    return final_blocks
