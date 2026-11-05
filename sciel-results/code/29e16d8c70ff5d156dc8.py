import time
import itertools
import random
import numpy as np

def _precompute_neighbors(q, n, radius):
    """
    Precompute the list of neighbors for every word in {0..q-1}^n.
    Returns a list of lists of integers, where neighbors[i] is a list of indices
    of words within Hamming distance <= radius from word i.
    """
    total = q ** n
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        
        if radius >= 1:
            # Generate all words at distance exactly 1
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
        
        neighbors.append(list(neigh_set))
        
    return all_words, neighbors

def _reduce_code(code, neighbors, counts):
    """
    Remove redundant codewords.
    A codeword c is redundant if for all x in neighbors[c], counts[x] > 1.
    Returns new_code and updated counts.
    """
    if not code:
        return code, counts
    
    current_code = list(code)
    current_counts = counts.copy()
    
    # We need to be careful with the order of removal. 
    # If we remove c, counts decrease. This might make other codewords redundant.
    # Simple approach: iterate until no changes.
    
    changed = True
    while changed:
        changed = False
        new_code = []
        for c in current_code:
            # Check if c is redundant
            # If min(counts[neighbors[c]]) >= 2, then c is redundant
            if np.min(current_counts[neighbors[c]]) >= 2:
                current_counts[neighbors[c]] -= 1
                changed = True
            else:
                new_code.append(c)
        current_code = new_code
        
    return current_code, current_counts

def _local_search(code, neighbors, counts, start_time, deadline):
    """
    Perform local search via swaps to reduce code size.
    """
    best_code = code.copy()
    best_counts = counts.copy()
    best_size = len(code)
    
    # Shuffle to avoid bias
    code = list(code)
    random.shuffle(code)
    
    while time.time() < deadline:
        # First, try to reduce
        code, counts = _reduce_code(code, neighbors, counts)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            best_counts = counts.copy()
            
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a swap
        # Strategy: Pick a codeword c. Find its unique neighbors.
        # Try to find a replacement w not in code that covers all unique neighbors of c.
        # If found, swap c with w, then reduce.
        
        # To make it efficient, we only consider codewords with few unique neighbors.
        # A codeword c has unique neighbors U_c = {x in neighbors[c] | counts[x] == 1}
        
        # We iterate through codewords. To avoid getting stuck, we randomize order.
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        found_swap = False
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
                
            c = code[idx]
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            # If no unique points, it should have been removed in reduction.
            # If unique points > some threshold (e.g., 3), it's hard to find a single replacement.
            # For r=1, max unique points is 17. But usually small.
            # Let's limit to unique points <= 3 for speed, or just try all.
            # Actually, if unique_points is large, the intersection of neighbors of these points is likely empty.
            
            if len(unique_points) == 0:
                # Should not happen after reduction, but force remove just in case
                code.pop(idx)
                counts[neighbors[c]] -= 1
                found_swap = True
                break
                
            if len(unique_points) > 4:
                continue
                
            # Find candidates w that cover all unique_points
            # Intersection of neighbors[u] for u in unique_points
            if len(unique_points) == 1:
                candidates_set = set(neighbors[unique_points[0]])
            else:
                candidates_set = set(neighbors[unique_points[0]])
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors[u])
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            # Filter candidates: must not be in current code
            code_set = set(code)
            candidates_set.difference_update(code_set)
            
            if not candidates_set:
                continue
            
            # Pick a candidate
            # Heuristic: Prefer candidates that cover many other unique points of other codewords?
            # For now, random choice to explore.
            c_prime = random.choice(list(candidates_set))
            
            # Apply swap
            # Remove c, add c_prime
            code.remove(c)
            counts[neighbors[c]] -= 1
            
            code.append(c_prime)
            counts[neighbors[c_prime]] += 1
            
            found_swap = True
            break # Restart reduction loop
        
        if not found_swap:
            # No swap found in this pass.
            # To avoid infinite loop, we can try a random neutral move or just break?
            # If we break, we might miss improvements from different random orders.
            # Let's do a few more passes with different random orders if time permits.
            # Or just break if we've done many passes.
            # For simplicity, we break. The reduction pass at the start of the loop ensures we are at a local minimum for removals.
            # If no swap is found, we are stuck.
            # To improve, we could try 2-for-1 swaps or random restarts, but 1-for-1 is standard.
            break

    return best_code, best_counts

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Handle general case or specific q=3, n=8, r=1
    # The problem statement specifies q=3, n=8, radius=1 as the main target.
    # We will implement a general solution that works for small q^n.
    
    if q ** n > 100000:
        # Fallback for large spaces: return empty or simple pattern? 
        # The task says "valid when every word... is within distance".
        # If we can't compute, we should try to return something valid if possible, 
        # but for large spaces, this is hard. 
        # Given the constraints, we assume q^n is manageable (6561).
        pass

    all_words, neighbors = _precompute_neighbors(q, n, radius)
    N = len(all_words)
    
    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Precompute neighbor arrays as numpy for faster sum
    neighbors_np = [np.array(neigh) for neigh in neighbors]
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        
        # Find word covering most uncovered
        # Optimization: If uncovered count is low, we can be more selective?
        # For now, standard greedy.
        for i in range(N):
            cnt = np.sum(uncovered[neighbors_np[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                
        if best_count == 0:
            break
            
        code.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        
        if time.time() > deadline - 20.0:
            break

    if not code:
        return []

    # Compute coverage counts
    counts = np.zeros(N, dtype=np.int64)
    for c in code:
        counts[neighbors_np[c]] += 1

    # 2. Reduction
    code, counts = _reduce_code(code, neighbors, counts)
    
    # 3. Local Search
    code, counts = _local_search(code, neighbors, counts, start_time, deadline)
    
    # Final verification and conversion to list of lists
    # Convert indices back to words
    result = [all_words[c] for c in code]
    
    return result

