import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    
    # Convert to numpy arrays
    block_subsets_np = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_blocks_np = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy()
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subsets_np[best_idx]
            newly_covered_mask = uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            for s in newly_covered:
                uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_blocks_np[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subsets_np[best_idx]
            newly_covered_mask = current_uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            for s in newly_covered:
                current_uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None, max_passes=3):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
                
        for _ in range(max_passes):
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                is_redundant = True
                for s_idx in block_subsets_np[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            for remove_pos in sorted(redundant_blocks, reverse=True):
                b_idx_to_remove = current_blocks[remove_pos]
                for s_idx in block_subsets_np[b_idx_to_remove]:
                    cover_count[s_idx] -= 1
                current_blocks.pop(remove_pos)
                changed = True
            
            if not changed:
                break
            
        return current_blocks

    def gradient_ascent_search(initial_blocks, random_seed=None, max_time=15.0):
        """
        Stochastic gradient ascent on a smooth surrogate objective.
        
        Formulation: Treat block selection as latent variables x_i in [0,1].
        Coverage of subset s: c_s = sum_i x_i * A[i,s] where A is incidence matrix.
        Soft coverage penalty: sum_s exp(-alpha * c_s) (smooth, differentiable)
        Objective: minimize F(x) = sum_s exp(-alpha * c_s) + lambda * sum_i x_i
        
        Use SGD with adaptive step size to escape local minima.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        start_time = time.time()
        
        # Initialize from initial blocks
        x = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in initial_blocks:
            x[b_idx] = 1.0
        
        alpha = 2.5
        lam = 0.015
        
        def compute_obj_and_grad(x_vec):
            """Compute objective and gradient of smooth surrogate."""
            # Coverage: c_s = sum_i x_i * A[i,s]
            # For efficiency, compute via block contributions
            coverage = np.zeros(num_subsets, dtype=np.float64)
            for i in range(num_blocks):
                if x_vec[i] > 1e-10:
                    coverage[block_subsets_np[i]] += x_vec[i]
            
            # Soft uncovered penalty: sum exp(-alpha * c_s)
            # Gradient w.r.t. x_i: -alpha * sum_{s in block_i} exp(-alpha * c_s)
            soft_uncovered = np.sum(np.exp(-alpha * coverage))
            
            # Regularizer: lambda * sum x_i
            reg = lam * np.sum(x_vec)
            
            obj = soft_uncovered + reg
            
            # Gradient
            grad = np.zeros(num_blocks, dtype=np.float64)
            exp_neg_alpha_cov = np.exp(-alpha * coverage)
            for i in range(num_blocks):
                if x_vec[i] > 1e-10 or rng.random() < 0.1:
                    b_subs = block_subsets_np[i]
                    grad[i] = -alpha * np.sum(exp_neg_alpha_cov[b_subs]) + lam
            
            return obj, grad, coverage
        
        current_obj, current_grad, current_cov = compute_obj_and_grad(x)
        
        best_blocks = list(initial_blocks)
        best_obj = current_obj
        best_valid_len = len(initial_blocks)
        
        # Check validity
        if np.all(current_cov > 0):
            best_valid_len = len(initial_blocks)
        
        lr = 0.3
        min_lr = 0.01
        max_lr = 1.0
        patience = 0
        max_patience = 200
        
        step = 0
        while time.time() - start_time < max_time:
            # Adaptive step size
            grad_norm = np.linalg.norm(current_grad)
            if grad_norm > 1e-10:
                step_size = lr / (1.0 + grad_norm * 0.1)
            else:
                step_size = lr
            
            # SGD update with momentum-like noise
            x_new = x.copy()
            x_new -= step_size * current_grad
            
            # Add noise to escape local minima
            noise_scale = 0.05 * (1.0 - step / 5000.0)
            x_new += rng.gauss(0, noise_scale, size=num_blocks)
            
            # Clip to [0, 1]
            x_new = np.clip(x_new, 0.0, 1.0)
            
            # Compute new objective
            new_obj, new_grad, new_cov = compute_obj_and_grad(x_new)
            
            # Accept if better or with probability (simulated annealing style)
            delta = new_obj - current_obj
            if delta < 0 or (patience > 100 and rng.random() < 0.3):
                x = x_new
                current_obj = new_obj
                current_grad = new_grad
                current_cov = new_cov
                patience = 0
                
                # Extract hard blocks (where x > 0.5)
                hard_blocks = [i for i in range(num_blocks) if x[i] > 0.5]
                
                # Check validity and count
                if len(hard_blocks) > 0:
                    cov_check = np.zeros(num_subsets, dtype=np.int32)
                    for b_idx in hard_blocks:
                        cov_check[block_subsets_np[b_idx]] += 1
                    if np.all(cov_check > 0) and len(hard_blocks) < best_valid_len:
                        best_valid_len = len(hard_blocks)
                        best_blocks = hard_blocks
                        best_obj = new_obj
            else:
                patience += 1
                # Reduce learning rate when stuck
                lr = max(min_lr, lr * 0.98)
            
            # Occasionally add random blocks to explore
            if step % 50 == 0 and rng.random() < 0.3:
                n_add = rng.randint(1, 3)
                for _ in range(n_add):
                    b_rand = rng.randint(0, num_blocks - 1)
                    x[b_rand] = 1.0
            
            # Occasionally remove blocks
            if step % 50 == 0 and rng.random() < 0.2:
                n_remove = rng.randint(1, 3)
                active = [i for i in range(num_blocks) if x[i] > 0.5]
                if len(active) > n_remove + 10:
                    to_remove = rng.sample(active, n_remove)
                    for b in to_remove:
                        x[b] = 0.0
            
            step += 1
            
            if patience > max_patience:
                # Restart with perturbation
                x = np.zeros(num_blocks, dtype=np.float64)
                for b_idx in best_blocks:
                    x[b_idx] = 1.0
                lr = 0.3
                patience = 0
        
        return best_blocks

    def local_search_swap(blocks_indices, random_seed=None, max_iters=50):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        current_set = set(current_blocks)
        
        if len(current_blocks) < 2:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
        
        best = list(current_blocks)
        best_len = len(current_blocks)
        
        for _ in range(max_iters):
            if not current_blocks:
                break
            
            swap_idx = rng.randint(0, len(current_blocks) - 1)
            b_swap = current_blocks[swap_idx]
            
            for _ in range(20):
                new_b = rng.randint(0, num_blocks - 1)
                if new_b == b_swap or new_b in current_set:
                    continue
                    
                test_cover = cover_count.copy()
                for s_idx in block_subsets_np[b_swap]:
                    test_cover[s_idx] -= 1
                for s_idx in block_subsets_np[new_b]:
                    test_cover[s_idx] += 1
                    
                if np.all(test_cover > 0):
                    current_blocks[swap_idx] = new_b
                    current_set.discard(b_swap)
                    current_set.add(new_b)
                    cover_count = test_cover
                    
                    redundant = []
                    for i, b_idx in enumerate(current_blocks):
                        is_redundant = True
                        for s_idx in block_subsets_np[b_idx]:
                            if cover_count[s_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            redundant.append(i)
                    for remove_pos in sorted(redundant, reverse=True):
                        b_to_remove = current_blocks[remove_pos]
                        for s_idx in block_subsets_np[b_to_remove]:
                            cover_count[s_idx] -= 1
                        current_blocks.pop(remove_pos)
                        current_set.discard(b_to_remove)
                        
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best = list(current_blocks)
                    break
            
        return best

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
        
        # Strategy 2: Gradient Ascent on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 20:
            ga_res = gradient_ascent_search(best_blocks, trial + 5000, max_time=12.0)
            min_res_ga = minimize(ga_res, trial + 6000)
            
            if len(min_res_ga) < best_len:
                best_len = len(min_res_ga)
                best_blocks = min_res_ga
        
        # Strategy 3: Local Search on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 15:
            ls_res = local_search_swap(best_blocks, trial + 7000, max_iters=50)
            min_res_ls = minimize(ls_res, trial + 8000)
            
            if len(min_res_ls) < best_len:
                best_len = len(min_res_ls)
                best_blocks = min_res_ls
        
        # Strategy 4: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 15:
            perturbed = list(best_blocks)
            num_remove = min(5, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subsets_np[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    # Final polish pass
    if best_blocks is not None and time.time() < deadline - 5:
        final_ls = local_search_swap(best_blocks, 99999, max_iters=80)
        final_min = minimize(final_ls, 99998)
        if len(final_min) < best_len:
            best_len = len(final_min)
            best_blocks = final_min
    
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
