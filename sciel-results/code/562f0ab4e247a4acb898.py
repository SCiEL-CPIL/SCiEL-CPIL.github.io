import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Precompute block overlap counts for faster swap evaluation
    # overlap[i][j] = number of t-subsets shared by block i and block j
    # This is expensive to fully precompute, so we'll compute on-demand with caching
    # Instead, precompute for each block a set of t-subsets for O(1) membership
    block_tsub_sets = [set(subs) for subs in block_tsubs]
    
    def run_greedy(seed):
        random.seed(seed)
        # Initial gains: number of t-subsets covered by each block
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        # For speed, we can use a heap, but with vectorized updates, 
        # np.argmax is O(N) which is fine for ~75k blocks.
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            # Find all blocks with max gain
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        global_to_local = {g: i for i, g in enumerate(selected_indices)}
        
        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)
        
        # Compute total_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1
                
        # Compute crit_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = True
            g = selected_indices[i]
            
            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    # si is now critical. Update crit_count for all other blocks containing si.
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in global_to_local:
                            i_other = global_to_local[g_other]
                            if not removed[i_other]:
                                crit_count[i_other] += 1
                
                elif old_c == 1:
                    pass
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def is_valid_cover(selected_indices):
        """Check if selected_indices form a valid cover."""
        covered = np.zeros(N_t, dtype=np.int8)
        for g in selected_indices:
            for si in block_tsubs[g]:
                covered[si] = 1
        return np.all(covered == 1)

    def local_search(selected_indices, max_iter=100):
        """
        Local search: try to remove blocks or swap blocks to improve the solution.
        Returns an improved list of global block indices.
        """
        if not selected_indices:
            return []
        
        current = list(selected_indices)
        current_set = set(current)
        n_sel = len(current)
        
        # Compute coverage counts
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Try removals first
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            for i in range(len(current)):
                g = current[i]
                # Check if removing g keeps the cover valid
                removable = True
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    # Remove g
                    current.pop(i)
                    current_set.discard(g)
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                    improved = True
                    break  # Restart loop after removal
        
        # Try swaps: replace one block with another that covers more uncovered t-subsets
        # Use a smarter candidate selection: prefer blocks that share many t-subsets with the removed block
        swap_improved = True
        swap_iter = 0
        while swap_improved and swap_iter < max_iter and time.time() < DEADLINE - 5.0:
            swap_improved = False
            swap_iter += 1
            random.shuffle(current)
            for i in range(len(current)):
                g_old = current[i]
                old_tsubs = block_tsub_sets[g_old]
                
                # Find candidate blocks: those that cover all critical t-subsets of g_old
                # A t-subset is critical for g_old if total_count[si] == 1
                critical_tsubs = [si for si in block_tsubs[g_old] if total_count[si] == 1]
                
                # Build candidate list: blocks that cover all critical t-subsets
                candidates = []
                # Sample from blocks that cover at least one critical t-subset
                critical_block_candidates = set()
                for si in critical_tsubs:
                    for g_cand in tsub_blocks[si]:
                        if g_cand not in current_set:
                            critical_block_candidates.add(g_cand)
                
                # If no critical t-subsets, any block works; sample randomly
                if not critical_tsubs:
                    for _ in range(30):
                        g_new = random.randint(0, N_blocks - 1)
                        if g_new not in current_set:
                            candidates.append(g_new)
                else:
                    # Filter candidates that cover ALL critical t-subsets
                    for g_cand in critical_block_candidates:
                        # Quick check: does g_cand cover all critical t-subsets?
                        if all(si in block_tsub_sets[g_cand] for si in critical_tsubs):
                            candidates.append(g_cand)
                        if len(candidates) >= 20:
                            break
                
                if not candidates:
                    continue
                
                random.shuffle(candidates)
                for g_new in candidates:
                    # Check if replacing g_old with g_new keeps the cover valid
                    # Remove g_old, add g_new, check coverage
                    temp_count = total_count.copy()
                    for si in block_tsubs[g_old]:
                        temp_count[si] -= 1
                    for si in block_tsubs[g_new]:
                        temp_count[si] += 1
                    
                    if np.all(temp_count >= 1):
                        # Valid replacement
                        current[i] = g_new
                        current_set.discard(g_old)
                        current_set.add(g_new)
                        total_count = temp_count
                        swap_improved = True
                        break
                
                if swap_improved:
                    break  # Restart after swap
        
        return current

    def crossover(sol1, sol2):
        """
        Crossover: combine two solutions by taking blocks from both.
        Returns a new solution that is a valid cover (hopefully smaller).
        Uses a greedy subset selection from the union to minimize size.
        """
        if not sol1 or not sol2:
            return sol1 if sol1 else sol2
        
        # Union of blocks
        union = list(set(sol1 + sol2))
        
        # Use a greedy approach to select a minimal subset from the union
        # that still covers all t-subsets
        if not union:
            return []
        
        # Start with the smaller solution and try to add blocks from the other
        # to see if we can get a smaller valid cover
        smaller = sol1 if len(sol1) <= len(sol2) else sol2
        larger = sol2 if len(sol1) <= len(sol2) else sol1
        
        # Try starting from smaller and see if we can reduce by swapping in blocks from larger
        candidate = list(smaller)
        candidate_set = set(candidate)
        
        # Compute coverage
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in candidate:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Try to replace blocks in candidate with blocks from larger
        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            for i in range(len(candidate)):
                g_old = candidate[i]
                # Find blocks from larger that could replace g_old
                for g_new in larger:
                    if g_new in candidate_set:
                        continue
                    # Check if replacement is valid
                    temp_count = total_count.copy()
                    for si in block_tsubs[g_old]:
                        temp_count[si] -= 1
                    for si in block_tsubs[g_new]:
                        temp_count[si] += 1
                    if np.all(temp_count >= 1):
                        # Valid replacement - this doesn't reduce size, but might enable future reductions
                        # Skip for now; focus on reduction
                        pass
                
                # Check if g_old can be removed
                removable = True
                for si in block_tsubs[g_old]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    candidate.pop(i)
                    candidate_set.discard(g_old)
                    for si in block_tsubs[g_old]:
                        total_count[si] -= 1
                    improved = True
                    break
        
        # Reduce the result
        reduced = reduce_solution_fast(candidate)
        return reduced

    def mutation(sol, rate=0.1):
        """
        Mutation: randomly swap blocks, then reduce.
        Focuses on swaps that maintain validity.
        """
        if not sol:
            return sol
        
        mutated = list(sol)
        mutated_set = set(mutated)
        n_mut = max(1, int(len(mutated) * rate))
        
        # Compute coverage for quick validity checks
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in mutated:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        for _ in range(n_mut):
            if time.time() > DEADLINE - 5.0:
                break
            op = random.choice(['swap', 'remove'])
            if op == 'swap' and len(mutated) > 0:
                # Pick a random block to potentially replace
                i = random.randint(0, len(mutated) - 1)
                g_old = mutated[i]
                # Try a few random replacements
                for attempt in range(10):
                    g_new = random.randint(0, N_blocks - 1)
                    if g_new in mutated_set:
                        continue
                    # Check if swap maintains validity
                    temp_count = total_count.copy()
                    for si in block_tsubs[g_old]:
                        temp_count[si] -= 1
                    for si in block_tsubs[g_new]:
                        temp_count[si] += 1
                    if np.all(temp_count >= 1):
                        # Valid swap
                        mutated[i] = g_new
                        mutated_set.discard(g_old)
                        mutated_set.add(g_new)
                        total_count = temp_count
                        break
            elif op == 'remove' and len(mutated) > 1:
                # Try to remove a block
                i = random.randint(0, len(mutated) - 1)
                g = mutated[i]
                # Check if removable
                removable = True
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    mutated.pop(i)
                    mutated_set.discard(g)
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
        
        # Reduce to ensure validity
        reduced = reduce_solution_fast(mutated)
        
        # Validate
        if is_valid_cover(reduced):
            return reduced
        else:
            return sol  # Revert if invalid

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Population for genetic algorithm - store as lists of indices
    population = [sol_idx]
    pop_set = [tuple(sol_idx)]
    
    # Randomized restarts with genetic improvement
    while time.time() < DEADLINE - 10.0:
        # Generate new candidate via greedy restart
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)
        
        # Apply local search
        cand_idx = local_search(cand_idx, max_iter=50)
        
        cand_tuple = tuple(cand_idx)
        
        # If better, update best
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
            population = [cand_idx]
            pop_set = [cand_tuple]
        else:
            # Add to population if not duplicate
            if cand_tuple not in pop_set:
                population.append(cand_idx)
                pop_set.append(cand_tuple)
                # Keep population size manageable
                if len(population) > 12:
                    population = population[:12]
                    pop_set = pop_set[:12]
        
        # Genetic operations: crossover and mutation
        if len(population) >= 2 and time.time() < DEADLINE - 15.0:
            # Crossover
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            if child and is_valid_cover(child):
                child = local_search(child, max_iter=30)
                child_tuple = tuple(child)
                if len(child) < best_len:
                    best_len = len(child)
                    best_sol = [all_blocks[i] for i in child]
                    population = [child]
                    pop_set = [child_tuple]
                elif child_tuple not in pop_set:
                    population.append(child)
                    pop_set.append(child_tuple)
                    if len(population) > 12:
                        population = population[:12]
                        pop_set = pop_set[:12]
            
            # Mutation
            parent = random.choice(population)
            child = mutation(parent, rate=0.15)
            if child and is_valid_cover(child):
                child = local_search(child, max_iter=30)
                child_tuple = tuple(child)
                if len(child) < best_len:
                    best_len = len(child)
                    best_sol = [all_blocks[i] for i in child]
                    population = [child]
                    pop_set = [child_tuple]
                elif child_tuple not in pop_set:
                    population.append(child)
                    pop_set.append(child_tuple)
                    if len(population) > 12:
                        population = population[:12]
                        pop_set = pop_set[:12]
        
        # Periodically apply local search to the best solution
        if time.time() < DEADLINE - 25.0 and random.random() < 0.3:
            best_idx = [all_blocks.index(b) for b in best_sol] if best_sol else []
            improved = local_search(best_idx, max_iter=50)
            if len(improved) < best_len:
                best_len = len(improved)
                best_sol = [all_blocks[i] for i in improved]
    
    # Final polish: apply local search to the best solution
    if best_sol and time.time() < DEADLINE - 5.0:
        # Convert back to indices
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        best_idx = [block_to_idx[b] for b in best_sol]
        improved = local_search(best_idx, max_iter=200)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol = [all_blocks[i] for i in improved]
    
    return best_sol
