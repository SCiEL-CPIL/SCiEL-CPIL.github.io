import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    total_space = q ** n

    # Precompute all words as a 2D numpy array: (total_space, n)
    # Use int32 to be safe against any overflow issues, though values are small
    words = np.zeros((total_space, n), dtype=np.int32)
    for i in range(total_space):
        idx = i
        for j in range(n):
            words[i, j] = idx % q
            idx //= q

    # Precompute neighbors for each word (including self)
    # neighbors[i] is a list of indices of words within Hamming distance <= radius
    neighbors = []
    for i in range(total_space):
        neigh = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = words[i, pos]
                # words[i, pos] is the digit with weight q^pos
                weight = q ** pos
                for val in range(q):
                    if val == orig_val:
                        continue
                    delta = (val - orig_val) * weight
                    new_idx = i + delta
                    neigh.append(new_idx)
        neighbors.append(neigh)

    # Greedy set cover (Optimized: pick random uncovered, check its neighbors)
    def greedy_cover():
        uncovered = [True] * total_space
        uncovered_count = total_space
        code = []
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered node to start
            # We need a list of uncovered indices to pick from efficiently
            # For simplicity, we can just pick a random index and check if it's uncovered,
            # or maintain a list. Maintaining a list is O(N) update.
            # Given N=2187, we can just scan for an uncovered node quickly.
            
            # Find an uncovered node
            target = -1
            for i in range(total_space):
                if uncovered[i]:
                    target = i
                    break
            
            if target == -1:
                break

            # Candidates are neighbors of target
            # We want the one that covers the most uncovered nodes
            best_gain = 0
            best_idx = -1
            
            for cand in neighbors[target]:
                # Calculate gain for cand
                gain = 0
                for nb in neighbors[cand]:
                    if uncovered[nb]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = cand
            
            if best_gain == 0:
                # Should not happen if target is uncovered and neighbors[target] is not empty
                break
                
            code.append(best_idx)
            
            # Update uncovered
            for nb in neighbors[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    uncovered_count -= 1
                    
        return code

    # Local search with neighbor lists
    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        if max_time is None:
            max_time = deadline

        current_code = code[:]
        code_set = set(current_code)
        
        # Build cover_count: cover_count[w] = number of codewords covering word w
        # Use int32 to be safe
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            for nb in neighbors[idx]:
                cover_count[nb] += 1

        improved = True
        while improved:
            improved = False
            if time.time() > max_time:
                break
            
            random.shuffle(current_code)
            code_set = set(current_code)
            n_cw = len(current_code)

            for i in range(n_cw):
                if time.time() > max_time:
                    return current_code

                idx = current_code[i]
                
                # Check if removable: all neighbors of idx have cover_count > 1
                removable = True
                unique_neighbors = []
                for nb in neighbors[idx]:
                    if cover_count[nb] == 1:
                        removable = False
                        unique_neighbors.append(nb)
                    elif cover_count[nb] < 1:
                        # Should not happen if valid
                        removable = False
                        break

                if removable:
                    # Remove it
                    for nb in neighbors[idx]:
                        cover_count[nb] -= 1
                    current_code.pop(i)
                    code_set.remove(idx)
                    improved = True
                    break
                else:
                    # Try to swap
                    # We have unique_neighbors. We need a candidate j not in code_set
                    # such that j covers all unique_neighbors.
                    candidate_set = set()
                    for nb in unique_neighbors:
                        candidate_set.update(neighbors[nb])
                    
                    # Filter candidates
                    valid_candidates = []
                    for j in candidate_set:
                        if j in code_set:
                            continue
                        # Check if j covers all unique_neighbors
                        covers_all = True
                        for nb in unique_neighbors:
                            if nb not in neighbors[j]:
                                covers_all = False
                                break
                        if covers_all:
                            valid_candidates.append(j)
                    
                    if valid_candidates:
                        new_idx = random.choice(valid_candidates)
                        # Swap
                        for nb in neighbors[idx]:
                            cover_count[nb] -= 1
                        for nb in neighbors[new_idx]:
                            cover_count[nb] += 1
                        code_set.remove(idx)
                        code_set.add(new_idx)
                        current_code[i] = new_idx
                        improved = True
                        break
        return current_code

    # Initial greedy
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code, max_time=deadline - 5.0)
    best_len = len(best_code)

    # Randomized restarts
    while time.time() < deadline - 5.0:
        cand_code = greedy_cover()
        cand_code = local_search_reduce(cand_code, max_time=deadline - 5.0)
        if len(cand_code) < best_len:
            best_code = cand_code
            best_len = len(cand_code)

    # Convert indices back to words
    return [words[i].tolist() for i in best_code]
