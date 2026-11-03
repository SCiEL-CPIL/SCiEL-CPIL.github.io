import itertools
import time
import numpy as np
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    def time_left():
        return deadline - time.time()

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each block, compute which t-subsets it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = [t_index[combo] for combo in itertools.combinations(block, t)]
        block_t_subsets.append(covered)

    # Build reverse mapping
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)
    block_t_sets = [set(lst) for lst in block_t_subsets]
    block_t_arr = [np.array(lst, dtype=np.int32) for lst in block_t_subsets]

    # Precompute for fast gain updates: for each block, store its t-subset indices as numpy array
    # Also precompute a sparse representation for vectorized operations
    max_t_per_block = max(len(lst) for lst in block_t_subsets) if block_t_subsets else 0
    block_t_arr_padded = np.zeros((num_k, max_t_per_block), dtype=np.int32)
    block_t_lens = np.zeros(num_k, dtype=np.int32)
    for i, lst in enumerate(block_t_subsets):
        if lst:
            block_t_arr_padded[i, :len(lst)] = lst
            block_t_lens[i] = len(lst)

    def run_greedy(weight_mode='inverse', seed=42):
        rng = np.random.default_rng(seed)
        if weight_mode == 'inverse':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = 1.0 / safe_counts
        elif weight_mode == 'random':
            t_weights = 0.3 / np.maximum(t_cover_count, 1).astype(np.float64) + 0.7 * rng.random(num_t)
        elif weight_mode == 'exp':
            safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
            t_weights = np.exp(-safe_counts)
        else:
            t_weights = 1.0 / np.maximum(t_cover_count, 1).astype(np.float64)

        gain = np.zeros(num_k, dtype=np.float64)
        for i, covered in enumerate(block_t_subsets):
            if covered:
                gain[i] = np.sum(t_weights[covered])

        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        while uncovered_count > 0 and time_left() > 25:
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1e10
            best_idx = int(np.argmax(masked_gain))
            best_gain = gain[best_idx]
            if best_gain <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
        return selected

    def local_search(selected, max_time_frac=0.15):
        if not selected:
            return selected
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        selected_set = set(selected)

        # Phase 1: Repeated single-drop
        for _ in range(80):
            if time_left() < 12:
                break
            changed = False
            for si in range(len(selected) - 1, -1, -1):
                if time_left() < 12:
                    break
                block_idx = selected[si]
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(block_idx)
                    changed = True
            if not changed:
                break

        # Phase 2: 1-for-1 swaps
        max_rounds = 150
        for round_num in range(max_rounds):
            if time_left() < 10:
                break
            improved = False
            order = list(range(len(selected)))
            random.shuffle(order)
            for si in order:
                if time_left() < 10:
                    break
                if improved:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                critical = [idx for idx in drop_set if cover_count[idx] == 1]
                if not critical:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    selected.pop(si)
                    selected_set.discard(drop_block)
                    improved = True
                    continue
                
                # Find candidates that cover all critical t-subsets
                candidate_sets = [set(t_to_blocks[idx]) for idx in critical]
                if not candidate_sets:
                    continue
                candidates = candidate_sets[0]
                for cs in candidate_sets[1:]:
                    candidates.intersection_update(cs)
                
                candidates -= selected_set
                
                if not candidates:
                    continue
                    
                best_candidate = -1
                best_score = float('inf')
                
                for b in candidates:
                    b_set = block_t_sets[b]
                    lost = 0
                    for idx in drop_set:
                        if idx not in b_set and cover_count[idx] == 1:
                            lost += 1
                            if lost > best_score:
                                break
                    if lost > best_score:
                        continue
                            
                    if lost < best_score:
                        best_score = lost
                        best_candidate = b
                        if best_score == 0:
                            break
                
                if best_candidate != -1:
                    for idx in drop_set:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    selected[si] = best_candidate
                    selected_set.discard(drop_block)
                    selected_set.add(best_candidate)
                    improved = True

        # Phase 3: 2-for-1 swaps (drop 2 blocks, add 1)
        for _ in range(30):
            if time_left() < 8:
                break
            improved = False
            # Try all pairs of blocks to drop
            n_sel = len(selected)
            if n_sel < 2:
                break
            pair_order = [(i, j) for i in range(n_sel) for j in range(i+1, n_sel)]
            random.shuffle(pair_order)
            for i, j in pair_order:
                if time_left() < 8:
                    break
                if improved:
                    break
                drop1 = selected[i]
                drop2 = selected[j]
                drop_set1 = block_t_subsets[drop1]
                drop_set2 = block_t_subsets[drop2]
                
                # Find t-subsets that would become uncovered
                would_lose = set()
                for idx in drop_set1:
                    if cover_count[idx] <= 1:
                        would_lose.add(idx)
                for idx in drop_set2:
                    if cover_count[idx] <= 1:
                        would_lose.add(idx)
                
                if not would_lose:
                    # Can just remove both
                    for idx in drop_set1:
                        cover_count[idx] -= 1
                    for idx in drop_set2:
                        cover_count[idx] -= 1
                    # Remove in reverse order to keep indices valid
                    selected.pop(max(i, j))
                    selected.pop(min(i, j))
                    selected_set.discard(drop1)
                    selected_set.discard(drop2)
                    improved = True
                    continue
                
                # Find a single block that covers all would_lose t-subsets
                candidate_sets = [set(t_to_blocks[idx]) for idx in would_lose]
                candidates = candidate_sets[0]
                for cs in candidate_sets[1:]:
                    candidates.intersection_update(cs)
                candidates -= selected_set
                
                if not candidates:
                    continue
                
                # Pick candidate that maximizes coverage of other t-subsets
                best_candidate = -1
                best_gain = -1
                for b in candidates:
                    b_set = block_t_sets[b]
                    gain = 0
                    for idx in drop_set1:
                        if idx in b_set and cover_count[idx] <= 1:
                            gain += 1
                    for idx in drop_set2:
                        if idx in b_set and cover_count[idx] <= 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b
                    if best_gain >= len(would_lose):
                        break
                
                if best_candidate != -1 and best_gain >= len(would_lose):
                    # Perform the swap
                    for idx in drop_set1:
                        cover_count[idx] -= 1
                    for idx in drop_set2:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_candidate]:
                        cover_count[idx] += 1
                    # Remove both and add new
                    selected.pop(max(i, j))
                    selected.pop(min(i, j))
                    selected_set.discard(drop1)
                    selected_set.discard(drop2)
                    selected.append(best_candidate)
                    selected_set.add(best_candidate)
                    improved = True
        return selected

    def greedy_repair(selected):
        """Repair coverage by adding blocks to cover any uncovered t-subsets."""
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        
        selected_set = set(selected)
        result = list(selected)
        
        # Find uncovered t-subsets
        uncovered = [i for i in range(num_t) if cover_count[i] == 0]
        
        if not uncovered:
            return result
        
        # Greedily add blocks to cover uncovered t-subsets
        while uncovered and time_left() > 5:
            # Find block that covers most uncovered t-subsets
            best_block = -1
            best_count = 0
            for b in range(num_k):
                if b in selected_set:
                    continue
                count = 0
                for idx in block_t_subsets[b]:
                    if cover_count[idx] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b
            
            if best_block == -1 or best_count == 0:
                break
            
            # Add the block
            result.append(best_block)
            selected_set.add(best_block)
            for idx in block_t_subsets[best_block]:
                if cover_count[idx] == 0:
                    cover_count[idx] = 1
                    uncovered.remove(idx)
                else:
                    cover_count[idx] += 1
        
        return result

    def crossover(parent1, parent2):
        """Combine two valid covers by randomly selecting blocks from each parent."""
        if not parent1 or not parent2:
            return list(parent1) if parent1 else list(parent2)
        
        # Shuffle both parents
        p1 = list(parent1)
        p2 = list(parent2)
        random.shuffle(p1)
        random.shuffle(p2)
        
        # Randomly select blocks from each parent
        child = []
        used = set()
        
        for b in p1:
            if b not in used and random.random() < 0.5:
                child.append(b)
                used.add(b)
        
        for b in p2:
            if b not in used and random.random() < 0.5:
                child.append(b)
                used.add(b)
        
        # Repair coverage
        child = greedy_repair(child)
        
        return child

    def mutation(selected):
        """Remove a random block and re-add the minimum-cost set of blocks to restore validity."""
        if not selected or time_left() < 5:
            return list(selected)
        
        result = list(selected)
        
        # Remove a random block
        if random.random() < 0.7:  # 70% chance to remove a block
            idx = random.randint(0, len(result) - 1)
            removed_block = result.pop(idx)
        
        # Repair coverage
        result = greedy_repair(result)
        
        return result

    def simulated_annealing(initial_solution, max_time_frac=0.3):
        """Run simulated annealing on a single covering solution."""
        if not initial_solution:
            return []
        
        selected = list(initial_solution)
        best_solution = list(selected)
        best_len = len(selected)
        
        # Build cover counts
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        selected_set = set(selected)
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.01
        cooling = 0.98
        
        T = T_start
        
        while time_left() > 15 and T > T_end:
            # Generate a neighbor
            if random.random() < 0.6:
                # Move type 1: remove a random block and repair
                if not selected:
                    break
                idx = random.randint(0, len(selected) - 1)
                removed_block = selected.pop(idx)
                selected_set.discard(removed_block)
                for idx2 in block_t_subsets[removed_block]:
                    cover_count[idx2] -= 1
                
                # Repair: add blocks to cover uncovered t-subsets
                uncovered = [i for i in range(num_t) if cover_count[i] == 0]
                added_blocks = []
                while uncovered and time_left() > 15:
                    best_b = -1
                    best_c = 0
                    # Sample a subset of blocks for speed
                    candidates = random.sample(range(num_k), min(500, num_k))
                    for b in candidates:
                        if b in selected_set:
                            continue
                        c = 0
                        for idx2 in block_t_subsets[b]:
                            if cover_count[idx2] == 0:
                                c += 1
                        if c > best_c:
                            best_c = c
                            best_b = b
                    if best_b == -1 or best_c == 0:
                        break
                    selected.append(best_b)
                    selected_set.add(best_b)
                    for idx2 in block_t_subsets[best_b]:
                        cover_count[idx2] += 1
                    added_blocks.append(best_b)
                
                new_len = len(selected)
                delta = new_len - best_len
                if new_len < best_len:
                    best_solution = list(selected)
                    best_len = new_len
                elif random.random() < np.exp(-delta / T):
                    # Accept worse solution
                    pass
                else:
                    # Reject: undo
                    for b in added_blocks:
                        selected.remove(b)
                        selected_set.discard(b)
                        for idx2 in block_t_subsets[b]:
                            cover_count[idx2] -= 1
                    selected.append(removed_block)
                    selected_set.add(removed_block)
                    for idx2 in block_t_subsets[removed_block]:
                        cover_count[idx2] += 1
            else:
                # Move type 2: 1-for-1 swap
                if not selected:
                    break
                idx = random.randint(0, len(selected) - 1)
                drop_block = selected[idx]
                drop_set = block_t_subsets[drop_block]
                critical = [i for i in drop_set if cover_count[i] == 1]
                
                if not critical:
                    # Can just remove
                    selected.pop(idx)
                    selected_set.discard(drop_block)
                    for i in drop_set:
                        cover_count[i] -= 1
                    new_len = len(selected)
                    if new_len < best_len:
                        best_solution = list(selected)
                        best_len = new_len
                else:
                    # Find a block that covers all critical t-subsets
                    candidate_sets = [set(t_to_blocks[i]) for i in critical]
                    candidates = candidate_sets[0]
                    for cs in candidate_sets[1:]:
                        candidates.intersection_update(cs)
                    candidates -= selected_set
                    
                    if candidates:
                        # Sample from candidates
                        cand_list = list(candidates)
                        if len(cand_list) > 20:
                            cand_list = random.sample(cand_list, 20)
                        best_swap = -1
                        best_swap_cost = float('inf')
                        for b in cand_list:
                            b_set = block_t_sets[b]
                            lost = 0
                            for i in drop_set:
                                if i not in b_set and cover_count[i] == 1:
                                    lost += 1
                                    if lost >= best_swap_cost:
                                        break
                            if lost < best_swap_cost:
                                best_swap_cost = lost
                                best_swap = b
                            if best_swap_cost == 0:
                                break
                        
                        if best_swap != -1:
                            # Perform swap
                            for i in drop_set:
                                cover_count[i] -= 1
                            for i in block_t_subsets[best_swap]:
                                cover_count[i] += 1
                            selected[idx] = best_swap
                            selected_set.discard(drop_block)
                            selected_set.add(best_swap)
                            new_len = len(selected)
                            if new_len < best_len:
                                best_solution = list(selected)
                                best_len = new_len
                        else:
                            # Try adding a block that covers most critical
                            best_add = -1
                            best_add_cover = 0
                            candidates2 = random.sample(range(num_k), min(200, num_k))
                            for b in candidates2:
                                if b in selected_set:
                                    continue
                                c = 0
                                for i in critical:
                                    if i in block_t_sets[b]:
                                        c += 1
                                if c > best_add_cover:
                                    best_add_cover = c
                                    best_add = b
                            if best_add != -1 and best_add_cover == len(critical):
                                # Add the block
                                selected.append(best_add)
                                selected_set.add(best_add)
                                for i in block_t_subsets[best_add]:
                                    cover_count[i] += 1
                                new_len = len(selected)
                                if new_len < best_len:
                                    best_solution = list(selected)
                                    best_len = new_len
        
        return best_solution

    # Run multiple greedy restarts
    best_selected = None
    best_len = float('inf')

    configs = [
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('random', 2024),
        ('exp', 314),
        ('random', 4242),
        ('inverse', 808),
    ]

    initial_solutions = []
    for i, (mode, seed) in enumerate(configs):
        if time_left() < 15:
            break
        sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]
        initial_solutions.append(sel)

    # Run simulated annealing on the best initial solution
    if time_left() > 25 and best_selected:
        sa_result = simulated_annealing(best_selected, 0.3)
        if sa_result and len(sa_result) < best_len:
            best_len = len(sa_result)
            best_selected = sa_result

    # Final polish with local search (extended)
    if best_selected and time_left() > 12:
        best_selected = local_search(best_selected, 0.25)
        # Additional SA pass if time remains
        if time_left() > 18:
            sa_result = simulated_annealing(best_selected, 0.15)
            if sa_result and len(sa_result) < len(best_selected):
                best_selected = sa_result
        # One more local search pass
        if time_left() > 8:
            best_selected = local_search(best_selected, 0.1)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 2:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_subsets[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]
    return result
