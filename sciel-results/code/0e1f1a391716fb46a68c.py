import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # For each block, list of subset indices it covers
    block_to_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_to_subsets.append(subs)
        
    # For each subset, list of block indices that cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
            
    best_solution = None
    best_len = float('inf')
    
    def greedy_lagrangian(dual_weights=None, penalty=0.0):
        """Greedy set cover with Lagrangian dual weights on uncovered subsets."""
        coverage = [0] * num_subsets
        solution = []
        uncovered_count = num_subsets
        
        # Initialize dual weights if not provided
        if dual_weights is None:
            dual_weights = [1.0] * num_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
                
            # Calculate score for each block: weighted sum of uncovered subsets it covers
            # minus penalty for adding a block
            scores = [0.0] * num_blocks
            
            # Vote method: for each uncovered subset, add its dual weight to its blocks
            for s_idx in range(num_subsets):
                if coverage[s_idx] == 0:
                    w = dual_weights[s_idx]
                    for b_idx in subset_to_blocks[s_idx]:
                        scores[b_idx] += w
            
            # Subtract penalty for adding a block (Lagrangian relaxation)
            if penalty > 0:
                for b_idx in range(num_blocks):
                    scores[b_idx] -= penalty
            
            # Find max score with random tie-breaking
            max_score = -float('inf')
            best_b = -1
            candidates = []
            for b_idx in range(num_blocks):
                if scores[b_idx] > max_score:
                    max_score = scores[b_idx]
                    candidates = [b_idx]
                elif scores[b_idx] == max_score and scores[b_idx] > 0:
                    candidates.append(b_idx)
            
            if not candidates:
                # Fallback: pick any block that covers uncovered subsets
                for b_idx in range(num_blocks):
                    for s_idx in block_to_subsets[b_idx]:
                        if coverage[s_idx] == 0:
                            candidates = [b_idx]
                            break
                    if candidates:
                        break
                if not candidates:
                    break
                    
            best_b = random.choice(candidates)
            solution.append(best_b)
            
            # Update coverage
            for s_idx in block_to_subsets[best_b]:
                if coverage[s_idx] == 0:
                    uncovered_count -= 1
                coverage[s_idx] += 1
                
        return solution

    def local_search(solution):
        """Local search with removal and swap operations."""
        coverage = [0] * num_subsets
        for b in solution:
            for s in block_to_subsets[b]:
                coverage[s] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            n = len(solution)
            for i in range(n):
                if time.time() > deadline:
                    break
                    
                b = solution[i]
                # Check if redundant
                critical_subs = []
                for s in block_to_subsets[b]:
                    if coverage[s] == 1:
                        critical_subs.append(s)
                        
                if not critical_subs:
                    # Redundant, remove
                    for s in block_to_subsets[b]:
                        coverage[s] -= 1
                    solution.pop(i)
                    improved = True
                    break
                    
                else:
                    # Critical, try to find a replacement
                    if len(critical_subs) > 0:
                        # Start with the first critical sub's blocks
                        candidates = set(subset_to_blocks[critical_subs[0]])
                        for s in critical_subs[1:]:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                                
                        if candidates:
                            # Filter out the current block and any block already in solution
                            sol_set = set(solution)
                            valid_candidates = [c for c in candidates if c not in sol_set]
                            
                            if valid_candidates:
                                # Pick one that maximizes additional coverage
                                best_new_b = None
                                best_additional = -1
                                for new_b in valid_candidates:
                                    # Count how many subsets this block covers that aren't fully covered yet
                                    additional = 0
                                    for s in block_to_subsets[new_b]:
                                        if coverage[s] == 1:  # Only this block covers it
                                            additional += 1
                                    if additional > best_additional:
                                        best_additional = additional
                                        best_new_b = new_b
                                        
                                if best_new_b is not None and best_additional > 0:
                                    # Perform swap
                                    for s in block_to_subsets[b]:
                                        coverage[s] -= 1
                                    for s in block_to_subsets[best_new_b]:
                                        coverage[s] += 1
                                        
                                    solution[i] = best_new_b
                                    improved = True
                                    break
                                
        return solution

    def lagrangian_relaxation(num_iterations=20):
        """
        Lagrangian relaxation: penalize uncovered 4-subsets, update dual weights
        via subgradient descent to guide greedy block selection.
        """
        best_sol = None
        best_len_local = float('inf')
        
        # Initialize dual weights
        dual_weights = [1.0] * num_subsets
        step_size = 0.5
        penalty = 0.0  # Start with no block penalty
        
        for it in range(num_iterations):
            if time.time() > deadline:
                break
                
            # Run greedy with current dual weights and penalty
            sol = greedy_lagrangian(dual_weights, penalty)
            if sol is None:
                break
            
            sol = local_search(sol)
            
            if len(sol) < best_len_local:
                best_len_local = len(sol)
                best_sol = sol.copy()
                
            # Check if we have a valid covering
            coverage = [0] * num_subsets
            valid = True
            for b in sol:
                for s in block_to_subsets[b]:
                    coverage[s] += 1
            
            uncovered = sum(1 for c in coverage if c == 0)
            if uncovered == 0:
                # Valid covering - update dual weights to emphasize hard subsets
                # Subgradient: increase weight of uncovered subsets (none here)
                # Decrease weight of well-covered subsets to encourage diversity
                for s_idx in range(num_subsets):
                    if coverage[s_idx] > 1:
                        dual_weights[s_idx] = max(0.1, dual_weights[s_idx] - step_size * 0.1)
                    else:
                        dual_weights[s_idx] = min(5.0, dual_weights[s_idx] + step_size * 0.05)
            else:
                # Not valid - increase penalty and weights for uncovered
                for s_idx in range(num_subsets):
                    if coverage[s_idx] == 0:
                        dual_weights[s_idx] = min(10.0, dual_weights[s_idx] + step_size)
                    else:
                        dual_weights[s_idx] = max(0.1, dual_weights[s_idx] - step_size * 0.05)
                
                # Increase penalty to discourage adding too many blocks
                penalty = min(2.0, penalty + step_size * 0.1)
            
            # Decay step size
            step_size *= 0.95
        
        return best_sol, best_len_local

    # Phase 1: Lagrangian relaxation to find good solutions
    lag_sol, lag_len = lagrangian_relaxation(num_iterations=25)
    if lag_sol is not None and lag_len < best_len:
        best_len = lag_len
        best_solution = lag_sol.copy()
    
    # Phase 2: Additional multi-start with greedy + local search
    for run in range(10):
        if time.time() > deadline:
            break
        
        # Use Lagrangian-inspired weights for diversity
        dual_weights = [random.uniform(0.5, 2.0) for _ in range(num_subsets)]
        sol = greedy_lagrangian(dual_weights, penalty=random.uniform(0, 0.5))
        if sol is None:
            continue
        
        sol = local_search(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
    
    # Phase 3: Final polish with more local search iterations on best solution
    if best_solution is not None and time.time() < deadline:
        # Do a few more passes of local search
        for _ in range(3):
            if time.time() > deadline:
                break
            best_solution = local_search(best_solution.copy())
    
    if best_solution is None:
        # Fallback: just return a valid covering, even if large
        sol = greedy_lagrangian(None, 0.0)
        return [list(all_blocks[b]) for b in sol]
        
    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    # Test
    blocks = construct(19, 7, 4)
    print(len(blocks))
