import itertools
import numpy as np
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block and blocks_of_subset
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
    
    # Convert to numpy arrays for faster indexing
    subsets_of_block_arr = [np.array(subs, dtype=np.int32) for subs in subsets_of_block]
    blocks_of_subset_arr = [np.array(subs, dtype=np.int32) for subs in blocks_of_subset]
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(uncovered_mask, initial_solution=None):
        if initial_solution is None:
            initial_solution = []
            
        if not np.any(uncovered_mask):
            return initial_solution
        
        candidate_blocks = set()
        for s_idx in np.where(uncovered_mask)[0]:
            for b_idx in blocks_of_subset_arr[s_idx]:
                candidate_blocks.add(b_idx)
        
        if not candidate_blocks:
            return initial_solution
        
        weights = np.zeros(n_k, dtype=np.int32)
        for b_idx in candidate_blocks:
            subs = subsets_of_block_arr[b_idx]
            weights[b_idx] = np.sum(uncovered_mask[subs])
        
        heap = [(-weights[b_idx], b_idx) for b_idx in candidate_blocks if weights[b_idx] > 0]
        heapq.heapify(heap)
        
        new_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while np.any(current_uncovered):
            if time.time() > deadline - 2:
                break # Timeout, return partial (invalid)
                
            neg_w, b_idx = heapq.heappop(heap)
            w = -neg_w
            
            if w == 0:
                break
            
            subs = subsets_of_block_arr[b_idx]
            if not np.any(current_uncovered[subs]):
                continue
                
            new_blocks.append(b_idx)
            
            newly_covered = current_uncovered[subs]
            current_uncovered[subs] = False
            
            for s_idx in subs[newly_covered]:
                for b_prime in blocks_of_subset_arr[s_idx]:
                    if b_prime != b_idx and b_prime in candidate_blocks:
                        weights[b_prime] -= 1
                        if weights[b_prime] > 0:
                            heapq.heappush(heap, (-weights[b_prime], b_prime))
                        
        return initial_solution + new_blocks

    def local_search_deletion(solution_blocks):
        if not solution_blocks:
            return solution_blocks
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in solution_blocks:
            cover_count[subsets_of_block_arr[b_idx]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_sol = []
            for b_idx in solution_blocks:
                subs = subsets_of_block_arr[b_idx]
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            solution_blocks = new_sol
            
        return solution_blocks

    # Initial Greedy with Restart and ILS
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        uncovered = np.ones(n_t, dtype=bool)
        
        weights = np.zeros(n_k, dtype=np.int32)
        for j in range(n_k):
            weights[j] = len(subsets_of_block[j])
            
        if restart > 0:
            noise_scale = 0.1 * (restart / num_restarts)
            weights += np.random.uniform(0, noise_scale, n_k)
        
        heap = [(-weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        current_solution = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if w <= 0:
                break
                
            subs = subsets_of_block_arr[j]
            if not np.any(uncovered[subs]):
                continue
                
            current_solution.append(j)
            
            newly_covered = uncovered[subs]
            uncovered[subs] = False
            
            for s_idx in subs[newly_covered]:
                for b_prime in blocks_of_subset_arr[s_idx]:
                    if b_prime != j:
                        weights[b_prime] -= 1
                        if weights[b_prime] > 0:
                            heapq.heappush(heap, (-weights[b_prime], b_prime))
                            
        if not np.all(uncovered == False):
            continue
            
        current_solution = local_search_deletion(current_solution)
        
        if len(current_solution) < best_size:
            best_size = len(current_solution)
            best_solution = current_solution.copy()
            
        # ILS
        if len(current_solution) <= best_size + 5:
            for ils_step in range(5):
                if time.time() > deadline - 2:
                    break
                    
                k_remove = random.randint(1, 5)
                if len(current_solution) <= k_remove:
                    continue
                    
                blocks_to_remove = random.sample(current_solution, k_remove)
                remove_set = set(blocks_to_remove)
                
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in current_solution:
                    cover_count[subsets_of_block_arr[b_idx]] += 1
                
                for b_idx in blocks_to_remove:
                    cover_count[subsets_of_block_arr[b_idx]] -= 1
                    
                uncovered_mask = (cover_count == 0).astype(bool)
                
                if not np.any(uncovered_mask):
                    new_solution = [b for b in current_solution if b not in remove_set]
                    new_solution = local_search_deletion(new_solution)
                    if len(new_solution) < len(current_solution):
                        current_solution = new_solution
                        if len(current_solution) < best_size:
                            best_size = len(current_solution)
                            best_solution = current_solution.copy()
                    continue
                    
                remaining_solution = [b for b in current_solution if b not in remove_set]
                repaired_solution = greedy_cover(uncovered_mask, remaining_solution)
                
                if len(repaired_solution) < len(current_solution):
                    current_solution = repaired_solution
                    current_solution = local_search_deletion(current_solution)
                    if len(current_solution) < best_size:
                        best_size = len(current_solution)
                        best_solution = current_solution.copy()
                        
    if best_solution is None:
        # Fallback: Simple Greedy
        uncovered = np.ones(n_t, dtype=bool)
        current_solution = []
        # Simple max coverage greedy
        while np.any(uncovered):
            max_new = -1
            best_b = -1
            for j in range(n_k):
                subs = subsets_of_block_arr[j]
                new_cov = np.sum(uncovered[subs])
                if new_cov > max_new:
                    max_new = new_cov
                    best_b = j
            if best_b == -1:
                break
            current_solution.append(best_b)
            uncovered[subsets_of_block_arr[best_b]] = False
        best_solution = current_solution
        best_solution = local_search_deletion(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
