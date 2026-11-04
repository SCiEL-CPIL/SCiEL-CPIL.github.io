import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for radius 1
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    neighbor_array = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val # Restore
        neighbors.append(neigh)
        neighbor_array[i, :] = neigh
        
    # Helper: Calculate coverage counts for a given code list
    def get_counts(code_list):
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        # Flatten all neighbors into a single array and use bincount
        flat_neighbors = np.concatenate([neighbors[i] for i in code_list])
        np.add.at(counts, flat_neighbors, 1)
        return counts

    def prune_code(code_list):
        counts = get_counts(code_list)
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to safely remove
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                # Check if all neighbors have count > 1
                if np.min(counts[neighbors[c]]) > 1:
                    code_list.pop(i)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        return code_list

    def improve_by_swaps(code_list, max_iterations=100):
        counts = get_counts(code_list)
        code_set = set(code_list)
        improved = True
        iterations = 0
        
        # Precompute neighbor sets for faster membership testing
        neighbor_sets = [set(neigh) for neigh in neighbors]
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Try to remove redundant words first
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            # Try swaps
            # Shuffle to avoid bias
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                
                # Identify critical points for c: points covered only by c
                crit_points = []
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        crit_points.append(nb)
                        
                if not crit_points:
                    # c is redundant, prune_code should have removed it? 
                    # Maybe counts changed? Re-check
                    if np.min(counts[neighbors[c]]) > 1:
                        code_list.remove(c)
                        code_set.remove(c)
                        for nb in neighbors[c]:
                            counts[nb] -= 1
                        improved = True
                        break # Restart loop as counts changed
                    continue
                
                # Find a replacement w not in code that covers all crit_points
                # Candidates: neighbors of the first critical point
                p = crit_points[0]
                candidates = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    # Check if w covers ALL crit_points
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: How many points with count==2 does w cover? (Creates redundancy)
                    score = 0
                    for nb in neighbors[w]:
                        if counts[nb] == 2:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    # Perform swap
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                        
                    code_list.append(best_w)
                    code_set.add(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                        
                    improved = True
                    break # Restart loop as counts changed
                    
        return code_list

    def run_greedy_local_search(seed):
        random.seed(seed)
        code = []
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        
        # Greedy Set Cover
        while uncovered_count > 0 and time.time() < deadline - 20:
            # Vectorized gain calculation
            # gains[i] = number of uncovered points in neighbors[i]
            gains = uncovered[neighbor_array].sum(axis=1)
            
            # Find index with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to introduce variation
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_idx)
            
            # Update uncovered
            new_uncovered = neighbors[best_idx]
            newly_covered = 0
            for nb in new_uncovered:
                if uncovered[nb]:
                    uncovered[nb] = False
                    newly_covered += 1
            uncovered_count -= newly_covered
            
        # Local Search
        code = prune_code(code)
        code = improve_by_swaps(code)
        
        # Final validity check
        counts = get_counts(code)
        if np.min(counts) < 1:
            return None
            
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Estimate time per restart: Greedy ~0.1s, Local Search ~0.5-2s
    # We have ~100s. Let's try up to 20 restarts or until time runs out.
    
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 1000 + 42
        current_code = run_greedy_local_search(seed)
        num_restarts += 1
        
        if current_code is not None:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                # If we reach a very good size, we might spend more time polishing
                # But for now, just keep going.
                
        # If we've done a few restarts, check time more strictly
        if num_restarts >= 10:
             if time.time() > deadline - 30:
                 break

    if best_code is None:
        # Fallback: Return a valid code from the first run if possible, or generate one
        # The first run should always succeed.
        best_code = run_greedy_local_search(0)
        if best_code is None:
             # Emergency fallback: just return the first greedy result without local search
             # This shouldn't happen
             return []

    # Final Prune on the best code
    best_code = prune_code(best_code)
    
    # Return as list of lists
    return [list(words[i]) for i in best_code]
