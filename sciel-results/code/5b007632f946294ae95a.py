import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert to arrays for fast access
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]
    
    # Precompute numpy arrays for block_tsubs for vectorized operations
    block_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_tsubs]

    def run_greedy_indices(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            
            if randomize:
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                best_block_idx = int(np.argmax(block_gain))
                
            selected_indices.append(best_block_idx)
            
            # Vectorized update: find newly covered t-subsets
            subs_np = block_tsubs_np[best_block_idx]
            newly_covered_mask = is_covered[subs_np] == 0
            newly_covered = subs_np[newly_covered_mask]
            
            if len(newly_covered) > 0:
                is_covered[newly_covered] = 1
                uncovered_count -= len(newly_covered)
                
                # Vectorized gain decrement: for each newly covered t-subset,
                # decrement gain of all blocks covering it
                # Build a flat array of block indices to decrement
                all_block_indices = []
                for si in newly_covered:
                    all_block_indices.extend(tsub_blocks_np[si].tolist())
                if all_block_indices:
                    all_block_indices_np = np.array(all_block_indices, dtype=np.int32)
                    np.subtract.at(block_gain, all_block_indices_np, 1)
                    
        return selected_indices

    def anneal_solution(initial_indices, time_limit=5.0):
        """
        Simulated Annealing on the set of blocks.
        Cost = len(blocks) + M * uncovered_count
        """
        if not initial_indices:
            return initial_indices
            
        M = 1000 # Penalty for uncovered t-subsets
        T = 2.0
        alpha = 0.97
        T_min = 1e-3
        
        # State
        selected = list(initial_indices)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1
            
        uncovered_count = int(np.sum(cov == 0))
        current_cost = len(selected) + M * uncovered_count
        
        best_cost = current_cost
        best_selected = list(selected)
        
        start = time.time()
        
        # Precompute: for each t-subset, list of blocks covering it (already have tsub_blocks)
        
        while T > T_min and (time.time() - start) < time_limit and time.time() < DEADLINE:
            # Generate Move: 35% Remove, 25% Add, 40% Swap
            move_type = random.random()
            
            if move_type < 0.35 and selected:
                # Remove a block
                b_idx = random.choice(selected)
                # Vectorized delta computation
                subs_np = block_tsubs_np[b_idx]
                delta_uncovered = int(np.sum(cov[subs_np] == 1))
                delta_cost = -1 + M * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
                        
            elif move_type < 0.60:
                # Add a block: prioritize blocks covering uncovered t-subsets
                if uncovered_count > 0:
                    # Pick a random uncovered t-subset, then a random block covering it
                    uncovered_mask = (cov == 0).nonzero()[0]
                    if len(uncovered_mask) > 0:
                        target_si = int(uncovered_mask[random.randint(0, len(uncovered_mask)-1)])
                        candidate_blocks = tsub_blocks[target_si]
                        if candidate_blocks:
                            b_idx = random.choice(candidate_blocks)
                            if b_idx not in selected_set:
                                # Vectorized delta computation
                                subs_np = block_tsubs_np[b_idx]
                                newly_covered = int(np.sum(cov[subs_np] == 0))
                                
                                delta_cost = 1 - M * newly_covered
                                
                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    selected.append(b_idx)
                                    selected_set.add(b_idx)
                                    cov[subs_np] += 1
                                    uncovered_count -= newly_covered
                                    current_cost += delta_cost
                                    
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)
                else:
                    # No uncovered, just add random block (will likely be rejected)
                    b_idx = random.randint(0, N_blocks - 1)
                    if b_idx not in selected_set:
                        delta_cost = 1
                        if random.random() < np.exp(-delta_cost / T):
                            selected.append(b_idx)
                            selected_set.add(b_idx)
                            cov[block_tsubs_np[b_idx]] += 1
                            current_cost += delta_cost
                            
            else:
                # Swap: remove one block, add another
                if len(selected) >= 2:
                    b_remove = random.choice(selected)
                    subs_remove = block_tsubs_np[b_remove]
                    
                    # Vectorized: find t-subsets that would become uncovered
                    would_uncover_mask = cov[subs_remove] == 1
                    would_uncover_count = int(np.sum(would_uncover_mask))
                    
                    if would_uncover_count > 0:
                        would_uncover = subs_remove[would_uncover_mask]
                        # Pick a random t-subset that would become uncovered, find a block covering it
                        target_si = int(random.choice(would_uncover))
                        candidate_blocks = tsub_blocks[target_si]
                        candidates = [b for b in candidate_blocks if b != b_remove and b not in selected_set]
                        if not candidates:
                            candidates = [b for b in candidate_blocks if b != b_remove]
                        if candidates:
                            # Score candidates by how many additional t-subsets they cover
                            best_b_add = -1
                            best_score = -1
                            # Sample up to 10 candidates for speed
                            sample = candidates if len(candidates) <= 10 else random.sample(candidates, 10)
                            for b_add in sample:
                                subs_add = block_tsubs_np[b_add]
                                # Count: t-subsets with cov==0 OR (in would_uncover AND cov==1)
                                score = int(np.sum((cov[subs_add] == 0) | (np.isin(subs_add, would_uncover) & (cov[subs_add] == 1))))
                                if score > best_score:
                                    best_score = score
                                    best_b_add = b_add
                            
                            b_add = best_b_add
                            
                            # Calculate delta: remove b_remove, add b_add
                            delta_uncover_remove = would_uncover_count
                            
                            # Uncovered change from adding b_add (considering b_remove is gone)
                            subs_add = block_tsubs_np[b_add]
                            delta_cover_add = int(np.sum((cov[subs_add] == 0) | (np.isin(subs_add, would_uncover) & (cov[subs_add] == 1))))
                            
                            # Net uncovered change: +delta_uncover_remove - delta_cover_add
                            net_uncover_change = delta_uncover_remove - delta_cover_add
                            delta_cost = net_uncover_change * M  # Block count stays same
                            
                            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                # Perform swap
                                selected.remove(b_remove)
                                selected_set.discard(b_remove)
                                cov[subs_remove] -= 1
                                uncovered_count += delta_uncover_remove
                                
                                selected.append(b_add)
                                selected_set.add(b_add)
                                cov[subs_add] += 1
                                uncovered_count -= delta_cover_add
                                current_cost += delta_cost
                                
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_selected = list(selected)
            
            T *= alpha
        
        # Repair: Ensure validity - use vectorized operations for speed
        uncovered_mask = (cov == 0).nonzero()[0]
        while len(uncovered_mask) > 0:
            best_b = -1
            max_cov = 0
            candidate_blocks = set()
            # Use up to 30 uncovered t-subsets for candidate generation
            for si in uncovered_mask[:30]:
                candidate_blocks.update(tsub_blocks[si])
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                # Vectorized count of newly covered t-subsets
                count = int(np.sum(cov[block_tsubs_np[b_idx]] == 0))
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx
            
            if best_b == -1:
                break
                
            selected.append(best_b)
            selected_set.add(best_b)
            cov[block_tsubs_np[best_b]] += 1
            uncovered_mask = (cov == 0).nonzero()[0]
            
        # Final Local Search: Reduction with multiple passes - more aggressive
        # Pass 1: Random order
        improved = True
        max_passes = 15
        for _ in range(max_passes):
            improved = False
            for b_idx in random.sample(list(selected), len(selected)):
                subs_np = block_tsubs_np[b_idx]
                if int(np.sum(cov[subs_np] == 1)) == 0:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    improved = True
                    break
            if not improved:
                break
        
        # Pass 2: Reverse order
        improved = True
        for _ in range(10):
            improved = False
            for b_idx in reversed(list(selected)):
                subs_np = block_tsubs_np[b_idx]
                if int(np.sum(cov[subs_np] == 1)) == 0:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    improved = True
                    break
            if not improved:
                break
        
        # Pass 3: Forward order
        improved = True
        for _ in range(10):
            improved = False
            for b_idx in list(selected):
                subs_np = block_tsubs_np[b_idx]
                if int(np.sum(cov[subs_np] == 1)) == 0:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    improved = True
                    break
            if not improved:
                break
        
        return selected

    # Main Loop
    best_sol_idx = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = anneal_solution(sol_idx, time_limit=15.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)
    
    # 2. Randomized Restarts with varied parameters
    restart_count = 0
    while time.time() < DEADLINE - 15:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        # Limit time per restart
        remaining = DEADLINE - time.time()
        if remaining < 10:
            break
        time_limit = min(10.0, remaining - 10.0)
        if time_limit < 1.0:
            break
            
        cand_idx = anneal_solution(cand_idx, time_limit=time_limit)
        
        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)
            
        restart_count += 1

    # 3. Final polish: try to reduce the best solution further
    if best_sol_idx is not None and time.time() < DEADLINE - 5:
        polished = anneal_solution(best_sol_idx, time_limit=min(8.0, DEADLINE - time.time() - 2.0))
        if len(polished) < best_len:
            best_sol_idx = polished
            best_len = len(polished)

    return [all_blocks[i] for i in best_sol_idx]
