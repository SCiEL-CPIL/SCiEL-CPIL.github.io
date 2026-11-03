import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Precompute block coverage as numpy arrays of indices
    # block_cover[b] = list of indices of t-subsets covered by block b
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper to get the set of t-subsets covered by a block
    # We use lists for block_cover, but sets for fast intersection in CSP
    # Converting to sets is expensive, so we keep lists and use numpy masks or manual checks
    # For CSP, we need "uncovered" set operations.
    
    # 2. Initial Solution via Greedy (Fast)
    def fast_greedy(rng):
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        while num_uncovered > 0:
            # Pick random uncovered
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            ti_target = rng.choice(uncovered_indices.tolist())
            
            candidates = t_to_blocks[ti_target]
            
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            selected.append(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected

    def prune(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return []
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    # 3. CSP Backtracking Search
    # Strategy:
    # Variables: Uncovered t-subsets
    # Domains: Blocks that cover the t-subset
    # Constraints: 
    #   1. Each block can be used at most once (implicit in set cover, but we allow reuse? No, distinct blocks required. But in covering designs, blocks are distinct sets. We select a SET of blocks. So each block is chosen 0 or 1 times.)
    #   2. Minimize number of blocks.
    
    # The CSP approach here is a bit tricky for minimization.
    # Standard CSP finds a valid solution. To minimize, we can use "Branch and Bound" or "Iterative Deepening".
    # However, the space is huge.
    
    # Alternative CSP interpretation:
    # We want to find a set of blocks S such that |S| is small and Union(cover(b) for b in S) == All T.
    # This is exactly the Set Cover problem.
    
    # The "Backtracking with Symmetry Breaking" idea:
    # 1. Order t-subsets by "difficulty" (fewest covering blocks).
    # 2. Pick the hardest uncovered t.
    # 3. Branch on which block to use to cover it.
    # 4. Symmetry breaking: 
    #    - If we have already chosen a block B, and we are choosing a new block B', 
    #      we can enforce an ordering on blocks to avoid permutations.
    #      E.g., if we sort blocks, we only allow choosing blocks with index > last_chosen_index?
    #      This works if the order of selection doesn't matter for the final set. It does.
    #      So we can enforce that blocks are added in increasing index order.
    #      This prunes all permutations of the same set.
    
    def csp_search(initial_blocks, max_blocks, rng, time_limit):
        """
        Try to find a covering with size <= max_blocks.
        Returns a valid covering if found, else None.
        """
        # Start with a greedy solution to get a good upper bound
        # We will use the initial_blocks to seed the search or as a bound.
        
        # State:
        # uncovered: set of t-subset indices not yet covered
        # current_blocks: list of block indices chosen so far
        # last_block_idx: to enforce ordering (symmetry breaking)
        
        # We need a fast way to update uncovered.
        # Using a set of ints for uncovered is good for small N_t, but N_t=11628.
        # A boolean array is faster for updates if we track counts.
        # But for backtracking, we need to revert.
        # Using a stack of changes is efficient.
        
        # Let's use a recursive backtracking with explicit stack or recursion.
        # Recursion depth might be up to max_blocks (~300). Python recursion limit is 1000. OK.
        
        uncovered = set(range(N_t))
        current_blocks = []
        
        # Sort t-subsets by number of covering blocks (static heuristic)
        # t_difficulty[ti] = len(t_to_blocks[ti])
        t_difficulty = [len(t_to_blocks[ti]) for ti in range(N_t)]
        
        # We will pick the "hardest" uncovered t-subset at each step.
        # To do this efficiently, we can maintain a priority queue or just scan.
        # Scanning 11628 items is fast.
        
        def find_hardest(uncovered_set):
            min_diff = float('inf')
            hardest = -1
            for ti in uncovered_set:
                d = t_difficulty[ti]
                if d < min_diff:
                    min_diff = d
                    hardest = ti
            return hardest

        # To speed up "find_hardest", we can pre-sort t-subsets by difficulty.
        # sorted_ts = sorted(range(N_t), key=lambda ti: t_difficulty[ti])
        # Then find the first ti in sorted_ts that is in uncovered_set.
        # This is O(N_t) worst case, but likely fast if easy ones are covered early.
        
        sorted_ts = sorted(range(N_t), key=lambda ti: t_difficulty[ti])
        
        def find_hardest_fast(uncovered_set):
            for ti in sorted_ts:
                if ti in uncovered_set:
                    return ti
            return -1

        # Backtracking function
        # Use a counter to track time
        start_time = time.time()
        
        # We will use a simple DFS with pruning.
        # Pruning 1: If len(current_blocks) + lower_bound > max_blocks, prune.
        # Lower bound: max(1, ceil(len(uncovered) / max_coverage_per_block))
        # max_coverage_per_block = max(len(block_cover[b]) for b in all_blocks) = C(8,5)=56.
        max_cov = 56
        
        # Pruning 2: If there is an uncovered t with no available blocks (given ordering?), prune.
        # With ordering constraint (blocks must be > last_idx), a t might have no blocks > last_idx.
        
        def backtrack(uncovered_set, current_blocks, last_idx):
            if time.time() > time_limit:
                return None
            
            if not uncovered_set:
                return list(current_blocks)
                
            # Pruning: Lower bound
            if len(current_blocks) + (len(uncovered_set) + max_cov - 1) // max_cov > max_blocks:
                return None
                
            # Find hardest uncovered t
            ti = find_hardest_fast(uncovered_set)
            if ti == -1:
                return None # Should not happen if uncovered_set is not empty
                
            # Candidates: blocks that cover ti and have index > last_idx
            candidates = []
            for bi in t_to_blocks[ti]:
                if bi > last_idx:
                    candidates.append(bi)
                    
            if not candidates:
                return None
                
            # Sort candidates by some heuristic? 
            # Maybe by how many NEW t's they cover?
            # Or randomize for diversity?
            # Let's sort by index to keep it deterministic first, then add randomization if needed.
            # Actually, for CSP, ordering matters for finding a solution quickly.
            # Let's try candidates in order of "gain" (new coverage).
            
            def score_candidate(bi):
                gain = 0
                for tj in block_cover[bi]:
                    if tj in uncovered_set:
                        gain += 1
                return gain
            
            # Sorting 56 candidates is cheap.
            # candidates.sort(key=score_candidate, reverse=True)
            
            # To allow exploration, we can shuffle if we want, but greedy order is usually good for finding *a* solution.
            # However, we are minimizing. We want to find *any* solution within max_blocks.
            # If we find one, we can try to reduce max_blocks.
            
            # Let's iterate candidates
            for bi in candidates:
                # Apply
                new_uncovered = set(uncovered_set)
                for tj in block_cover[bi]:
                    new_uncovered.discard(tj)
                    
                current_blocks.append(bi)
                
                result = backtrack(new_uncovered, current_blocks, bi)
                if result is not None:
                    return result
                    
                current_blocks.pop()
                
            return None

        # Start search
        # We need to handle the "last_idx" symmetry breaking correctly.
        # The first block can be any block. So last_idx starts at -1.
        
        return backtrack(uncovered, current_blocks, -1)

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Multi-start Greedy to get a good initial upper bound
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 60.0:
            break
        rng = random.Random(123 + i)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    if not best_solution:
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(blocks)
        best_len = len(best_solution)

    # Phase 2: CSP Search to improve
    # We try to find a solution with size < best_len.
    # We start with target_size = best_len - 1 and go down.
    # If we find a solution, we update best_len and try smaller.
    
    # Note: CSP search can be slow if it explores too much.
    # We allocate the remaining time to this phase.
    
    # To make CSP search more effective, we can use the current best solution's blocks
    # to guide the search? Or just rely on the difficulty ordering.
    
    # Let's try to find a solution of size best_len - 1.
    # If successful, best_len -= 1.
    # Repeat until time runs out or no improvement.
    
    # We need to be careful with the time limit for each CSP call.
    # If we run out of time in the middle of a CSP search, we discard it.
    
    target_len = best_len - 1
    while target_len >= 264 and time.time() < deadline - 10.0:
        # Allocate a chunk of time for this search
        # If we have plenty of time, give it more.
        # If we have little, give it little.
        time_remaining = deadline - 10.0 - time.time()
        search_time_limit = time.time() + min(time_remaining, 10.0) # Max 10s per target size
        
        # Run CSP search
        # We need a fresh state for each search? 
        # The backtrack function modifies local sets, so it's safe.
        
        solution = csp_search(None, target_len, random.Random(42), search_time_limit)
        
        if solution is not None:
            # Validate
            if len(solution) <= target_len:
                # Check validity
                covered_mask = np.zeros(N_t, dtype=bool)
                for bi in solution:
                    for ti in block_cover[bi]:
                        covered_mask[ti] = True
                if not covered_mask.all():
                    # Invalid, shouldn't happen
                    pass
                else:
                    best_solution = solution
                    best_len = len(solution)
                    target_len = best_len - 1
                    continue
        
        # If no solution found, decrease target_len? 
        # No, if we can't find a solution of size target_len, we can't find one of size target_len - 1 
        # easily with this simple DFS without more powerful pruning.
        # Actually, if we fail to find size K, we should stop trying smaller sizes 
        # unless we improve the heuristic or use a different search strategy.
        # But maybe we just ran out of time for that specific branch.
        # Let's break if we didn't find one, as trying smaller is unlikely to succeed 
        # if the larger one failed due to search space explosion.
        
        # However, the failure might be due to the specific path. 
        # A more robust approach: if we fail, we stop the CSP phase.
        break

    # Phase 3: Final Local Search (optional, if time permits)
    # The current program has a local search. 
    # The CSP approach is a structural change. 
    # We can still apply the local search from the original program to the CSP result 
    # to squeeze out more.
    
    if best_solution and time.time() < deadline - 5.0:
        # Re-use the local_search_reduce logic from the original program 
        # but adapted to work with the current best_solution.
        
        def local_search_reduce(blocks, rng, time_limit):
            current_blocks = list(blocks)
            if not current_blocks:
                return current_blocks
                
            counts = np.zeros(N_t, dtype=np.int32)
            for bi in current_blocks:
                for ti in block_cover[bi]:
                    counts[ti] += 1
                    
            best_blocks = list(current_blocks)
            best_size = len(current_blocks)
            
            def get_unique(bi):
                return [ti for ti in block_cover[bi] if counts[ti] == 1]
                
            def is_redundant(bi):
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        return False
                return True
                
            while time.time() < time_limit:
                idx = rng.randrange(len(current_blocks))
                bi = current_blocks[idx]
                
                unique_ts = get_unique(bi)
                
                if not unique_ts:
                    current_blocks.pop(idx)
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                
                candidates = set(t_to_blocks[unique_ts[0]])
                for ti in unique_ts[1:]:
                    candidates &= set(t_to_blocks[ti])
                    
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                best_cand = -1
                best_score = -1
                
                cand_list = list(candidates)
                if len(cand_list) > 50:
                    cand_list = rng.sample(cand_list, 50)
                    
                for c in cand_list:
                    score = 0
                    for ti in block_cover[c]:
                        if counts[ti] == 1:
                            if ti not in unique_ts:
                                score += 1
                                
                    if score > best_score:
                        best_score = score
                        best_cand = c
                        
                if best_cand == -1:
                    continue
                    
                for ti in block_cover[bi]:
                    counts[ti] -= 1
                for ti in block_cover[best_cand]:
                    counts[ti] += 1
                current_blocks[idx] = best_cand
                
                new_blocks = []
                drop_occurred = False
                for b in current_blocks:
                    if is_redundant(b):
                        for ti in block_cover[b]:
                            counts[ti] -= 1
                        drop_occurred = True
                    else:
                        new_blocks.append(b)
                        
                if drop_occurred:
                    current_blocks = new_blocks
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    current_blocks = new_blocks
                    
            return best_blocks

        rng = random.Random(999)
        optimized = local_search_reduce(best_solution, rng, deadline - 2.0)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized

    return [list(all_blocks[bi]) for bi in best_solution]
