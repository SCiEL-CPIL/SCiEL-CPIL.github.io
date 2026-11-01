import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    
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
    
    def greedy_randomized(seed_offset=0):
        """Greedy set cover with randomized tie-breaking."""
        rng = random.Random(seed_offset)
        coverage = [0] * num_subsets
        solution = []
        uncovered_count = num_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
                
            # Calculate score for each block: number of currently uncovered subsets it covers
            scores = [0] * num_blocks
            
            # Vote method: for each uncovered subset, increment score of its blocks
            for s_idx in range(num_subsets):
                if coverage[s_idx] == 0:
                    for b_idx in subset_to_blocks[s_idx]:
                        scores[b_idx] += 1
            
            # Find max score with random tie-breaking
            max_score = -1
            candidates = []
            for b_idx in range(num_blocks):
                if scores[b_idx] > max_score:
                    max_score = scores[b_idx]
                    candidates = [b_idx]
                elif scores[b_idx] == max_score and scores[b_idx] > 0:
                    candidates.append(b_idx)
            
            if not candidates:
                break
                
            best_b = rng.choice(candidates)
            solution.append(best_b)
            
            # Update coverage
            for s_idx in block_to_subsets[best_b]:
                if coverage[s_idx] == 0:
                    uncovered_count -= 1
                coverage[s_idx] += 1
                
        return solution

    def local_search(solution, max_iter=5):
        """Local search with removal and swap operations."""
        coverage = [0] * num_subsets
        for b in solution:
            for s in block_to_subsets[b]:
                coverage[s] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            
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

    def simulated_annealing(solution, iterations=500):
        """Simulated annealing with swap moves."""
        if time.time() > deadline:
            return solution
            
        coverage = [0] * num_subsets
        for b in solution:
            for s in block_to_subsets[b]:
                coverage[s] += 1
        
        current_len = len(solution)
        best_solution_local = solution.copy()
        best_len_local = current_len
        
        temperature = 1.0
        cooling = 0.995
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            if temperature < 0.01:
                break
                
            # Random swap: pick a block in solution, replace with random block
            i = random.randint(0, len(solution) - 1)
            old_b = solution[i]
            
            # Find blocks that cover at least one critical subset of old_b
            critical_subs = [s for s in block_to_subsets[old_b] if coverage[s] == 1]
            
            if critical_subs:
                # Need a block that covers all critical subsets
                candidates = set(subset_to_blocks[critical_subs[0]])
                for s in critical_subs[1:]:
                    candidates.intersection_update(subset_to_blocks[s])
                    if not candidates:
                        break
                
                if not candidates:
                    temperature *= cooling
                    continue
                    
                candidates.discard(old_b)
                sol_set = set(solution)
                candidates -= sol_set
                
                if not candidates:
                    temperature *= cooling
                    continue
                    
                new_b = random.choice(list(candidates))
            else:
                # No critical subsets, can swap with any block
                new_b = random.randint(0, num_blocks - 1)
                if new_b == old_b or new_b in set(solution):
                    temperature *= cooling
                    continue
            
            # Calculate energy change
            # Remove old_b, add new_b
            old_only = 0
            for s in block_to_subsets[old_b]:
                if coverage[s] == 1:
                    old_only += 1
            
            new_extra = 0
            for s in block_to_subsets[new_b]:
                if coverage[s] == 0:
                    new_extra += 1
            
            # Energy change: -new_extra + old_only (we want to minimize)
            delta_e = old_only - new_extra
            
            if delta_e <= 0 or random.random() < pow(2.718281828, -delta_e / max(temperature, 0.01)):
                # Accept
                for s in block_to_subsets[old_b]:
                    coverage[s] -= 1
                for s in block_to_subsets[new_b]:
                    coverage[s] += 1
                solution[i] = new_b
                
                if len(solution) < best_len_local:
                    best_len_local = len(solution)
                    best_solution_local = solution.copy()
            
            temperature *= cooling
        
        return best_solution_local

    def multi_start_local_search():
        """Run multiple greedy starts with local search."""
        nonlocal best_solution, best_len
        
        for run in range(40):
            if time.time() > deadline:
                break
                
            sol = greedy_randomized(seed_offset=run * 12345)
            if sol is None:
                continue
            
            sol = local_search(sol, max_iter=8)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol.copy()
                
    def constrained_swap_search(solution):
        """Try 1-for-2 swaps: remove 1 block, add 2 blocks that cover all lost subsets."""
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
                    # Try to replace with 2 blocks that cover all critical subsets
                    # This is expensive, so only try for small critical sets
                    if len(critical_subs) <= 4:
                        # Find all blocks that cover at least one critical subset
                        candidate_blocks = set()
                        for s in critical_subs:
                            candidate_blocks.update(subset_to_blocks[s])
                        candidate_blocks.discard(b)
                        candidate_blocks -= set(solution)
                        
                        # Try pairs of candidates
                        candidates = list(candidate_blocks)
                        random.shuffle(candidates)
                        limit = min(len(candidates), 80)  # Limit search
                        for j in range(limit):
                            for l in range(j+1, limit):
                                b1, b2 = candidates[j], candidates[l]
                                # Check if b1 and b2 together cover all critical subsets
                                covered = set()
                                for s in block_to_subsets[b1]:
                                    if s in critical_subs or coverage[s] == 1:
                                        covered.add(s)
                                for s in block_to_subsets[b2]:
                                    if s in critical_subs or coverage[s] == 1:
                                        covered.add(s)
                                
                                # Check if all critical subsets are covered
                                if all(cs in covered for cs in critical_subs):
                                    # Accept this swap
                                    for s in block_to_subsets[b]:
                                        coverage[s] -= 1
                                    for s in block_to_subsets[b1]:
                                        coverage[s] += 1
                                    for s in block_to_subsets[b2]:
                                        coverage[s] += 1
                                    
                                    solution.pop(i)
                                    solution.extend([b1, b2])
                                    improved = True
                                    break
                            if improved:
                                break
                        
        return solution

    # Run multiple strategies
    for run in range(20):
        if time.time() > deadline:
            break
        
        sol = greedy_randomized(seed_offset=run * 9999)
        if sol is None:
            continue
        
        sol = local_search(sol, max_iter=10)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
    
    # Try simulated annealing on best solution
    if best_solution is not None and time.time() < deadline:
        sa_sol = simulated_annealing(best_solution.copy(), iterations=800)
        if len(sa_sol) < best_len:
            best_len = len(sa_sol)
            best_solution = sa_sol.copy()
    
    # Try additional constrained swaps on best solution
    if best_solution is not None and time.time() < deadline:
        sol = constrained_swap_search(best_solution.copy())
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
    
    # Final local search pass
    if best_solution is not None and time.time() < deadline:
        sol = local_search(best_solution.copy(), max_iter=15)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
    
    if best_solution is None:
        # Fallback: just return a valid covering, even if large
        sol = greedy_randomized()
        return [list(all_blocks[b]) for b in sol]
        
    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    # Test
    blocks = construct(19, 7, 4)
    print(len(blocks))
