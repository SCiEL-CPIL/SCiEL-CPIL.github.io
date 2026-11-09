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
    
    # Precompute intersection sizes between all pairs of blocks (for 2-for-1 exchanges)
    # block_intersection[i][j] = number of t-subsets covered by both block i and block j
    # This is expensive for all pairs, so we compute lazily or use a different approach
    # For now, we'll compute on-the-fly in the local search

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

    def aggressive_reduction(selected, selected_set, cov, time_limit=3.0):
        """
        Intensive local search to reduce block count.
        Uses multiple strategies: 1-for-0, 1-for-1, and 2-for-1 exchanges.
        """
        start = time.time()
        max_iterations = 2000
        
        for iteration in range(max_iterations):
            if time.time() - start > time_limit or time.time() > DEADLINE:
                break
            
            # Strategy 1: Remove redundant blocks (1-for-0)
            improved = False
            # Try random order multiple times
            for _ in range(3):
                for b_idx in random.sample(list(selected), len(selected)):
                    subs_np = block_tsubs_np[b_idx]
                    if int(np.sum(cov[subs_np] == 1)) == 0:
                        selected.remove(b_idx)
                        selected_set.discard(b_idx)
                        cov[subs_np] -= 1
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Strategy 2: 1-for-1 swap (replace one block with another that covers more)
            if len(selected) >= 1:
                b_remove = random.choice(selected)
                subs_remove = block_tsubs_np[b_remove]
                would_uncover_mask = cov[subs_remove] == 1
                would_uncover_count = int(np.sum(would_uncover_mask))
                
                if would_uncover_count > 0:
                    would_uncover = subs_remove[would_uncover_mask]
                    # Find best replacement block
                    best_b_add = -1
                    best_score = -1
                    
                    # Sample from blocks covering the would-be-uncovered t-subsets
                    candidate_set = set()
                    for si in would_uncover[:20]:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b_remove)
                    candidate_set -= selected_set
                    
                    if candidate_set:
                        candidates = list(candidate_set)
                        sample = candidates if len(candidates) <= 15 else random.sample(candidates, 15)
                        for b_add in sample:
                            subs_add = block_tsubs_np[b_add]
                            # Score: how many t-subsets does b_add cover that b_remove covers?
                            # We want b_add to cover all of would_uncover and ideally more
                            score = int(np.sum(np.isin(subs_add, would_uncover)))
                            # Bonus for covering additional uncovered t-subsets
                            score += int(np.sum(cov[subs_add] == 0))
                            if score > best_score:
                                best_score = score
                                best_b_add = b_add
                        
                        if best_b_add != -1 and best_score >= would_uncover_count:
                            # Perform swap
                            subs_add = block_tsubs_np[best_b_add]
                            selected.remove(b_remove)
                            selected_set.discard(b_remove)
                            cov[subs_remove] -= 1
                            
                            selected.append(best_b_add)
                            selected_set.add(best_b_add)
                            cov[subs_add] += 1
                            improved = True
            
            if improved:
                continue
            
            # Strategy 3: 2-for-1 exchange (remove two blocks, add one)
            if len(selected) >= 2 and time.time() - start < time_limit * 0.7:
                b1, b2 = random.sample(selected, 2)
                subs1 = block_tsubs_np[b1]
                subs2 = block_tsubs_np[b2]
                
                # Find t-subsets that would become uncovered
                cov_temp = cov.copy()
                cov_temp[subs1] -= 1
                cov_temp[subs2] -= 1
                would_uncover_mask = cov_temp == 0
                would_uncover_count = int(np.sum(would_uncover_mask))
                
                if would_uncover_count > 0:
                    would_uncover = np.where(would_uncover_mask)[0]
                    
                    # Find a single block that covers all would_uncover t-subsets
                    candidate_set = set()
                    for si in would_uncover[:30]:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b1)
                    candidate_set.discard(b2)
                    candidate_set -= selected_set
                    
                    if candidate_set:
                        candidates = list(candidate_set)
                        sample = candidates if len(candidates) <= 20 else random.sample(candidates, 20)
                        best_b_add = -1
                        best_coverage = -1
                        
                        for b_add in sample:
                            subs_add = block_tsubs_np[b_add]
                            coverage = int(np.sum(np.isin(subs_add, would_uncover)))
                            if coverage > best_coverage:
                                best_coverage = coverage
                                best_b_add = b_add
                        
                        if best_b_add != -1 and best_coverage >= would_uncover_count:
                            # Perform 2-for-1 exchange
                            subs_add = block_tsubs_np[best_b_add]
                            selected.remove(b1)
                            selected_set.discard(b1)
                            selected.remove(b2)
                            selected_set.discard(b2)
                            cov[subs1] -= 1
                            cov[subs2] -= 1
                            
                            selected.append(best_b_add)
                            selected_set.add(best_b_add)
                            cov[subs_add] += 1
                            improved = True
            
            if not improved:
                # Try a few more random removals before giving up
                for _ in range(5):
                    if not selected:
                        break
                    b_idx = random.choice(selected)
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
        
        # Repair: Ensure validity
        uncovered_mask = (cov == 0).nonzero()[0]
        while len(uncovered_mask) > 0:
            best_b = -1
            max_cov = 0
            candidate_blocks = set()
            for si in uncovered_mask[:30]:
                candidate_blocks.update(tsub_blocks[si])
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
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
        
        # Aggressive reduction
        remaining_time = min(3.0, DEADLINE - time.time() - 2.0)
        if remaining_time > 0.5:
            selected = aggressive_reduction(selected, selected_set, cov, time_limit=remaining_time)
        
        return selected

    # Main Loop
    best_sol_idx = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = anneal_solution(sol_idx, time_limit=12.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)
    
    # 2. Randomized Restarts with varied parameters
    restart_count = 0
    while time.time() < DEADLINE - 20:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        # Limit time per restart
        remaining = DEADLINE - time.time()
        if remaining < 15:
            break
        time_limit = min(8.0, remaining - 15.0)
        if time_limit < 1.0:
            break
            
        cand_idx = anneal_solution(cand_idx, time_limit=time_limit)
        
        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)
            
        restart_count += 1

    # 3. Final intensive polish: try to reduce the best solution further
    if best_sol_idx is not None and time.time() < DEADLINE - 3:
        # Multiple rounds of aggressive reduction
        selected = list(best_sol_idx)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1
        
        polish_time = min(10.0, DEADLINE - time.time() - 2.0)
        if polish_time > 1.0:
            # Round 1: aggressive reduction
            selected = aggressive_reduction(selected, selected_set, cov, time_limit=polish_time * 0.4)
            
            # Round 2: more SA + reduction
            if time.time() < DEADLINE - 3:
                remaining = DEADLINE - time.time() - 3.0
                if remaining > 2.0:
                    # Rebuild state for SA
                    selected_set = set(selected)
                    cov = np.zeros(N_t, dtype=np.int32)
                    for b_idx in selected:
                        cov[block_tsubs_np[b_idx]] += 1
                    
                    polished = anneal_solution(selected, time_limit=min(5.0, remaining))
                    if len(polished) < len(selected):
                        selected = polished
                        selected_set = set(selected)
                        cov = np.zeros(N_t, dtype=np.int32)
                        for b_idx in selected:
                            cov[block_tsubs_np[b_idx]] += 1
            
            # Round 3: final reduction
            if time.time() < DEADLINE - 2:
                remaining = DEADLINE - time.time() - 2.0
                if remaining > 0.5:
                    selected_set = set(selected)
                    cov = np.zeros(N_t, dtype=np.int32)
                    for b_idx in selected:
                        cov[block_tsubs_np[b_idx]] += 1
                    selected = aggressive_reduction(selected, selected_set, cov, time_limit=remaining)
            
            if len(selected) < best_len:
                best_sol_idx = selected
                best_len = len(selected)

    return [all_blocks[i] for i in best_sol_idx]
