import itertools
import time
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But the function signature is generic.
    # We assume the evaluator passes q=3, n=7, radius=1.
    # If parameters differ, we fall back to a simple greedy or seed.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple Greedy Set Cover
        # This is a generic implementation
        total_words = q ** n
        # Map word to index
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * q + w[i]
            return idx
        
        # Precompute neighbors
        # This might be slow for large q,n, but for 3,7 it's fine.
        # For generic fallback, we might just return seed if n is small.
        # Given the task contract, it's specifically for this instance.
        # I will implement the optimized solver for 3,7,1 and fallback to seed for others.
        
        if n <= 10:
             # Generic Greedy
             all_words = list(itertools.product(range(q), repeat=n))
             idx_map = {w: i for i, w in enumerate(all_words)}
             
             def get_neighbors(w):
                 neigh = set()
                 base_idx = idx_map[w]
                 neigh.add(base_idx)
                 for pos in range(n):
                     for val in range(q):
                         if val == w[pos]: continue
                         new_w = list(w)
                         new_w[pos] = val
                         neigh.add(idx_map[tuple(new_w)])
                 return neigh
             
             masks = [0] * len(all_words)
             coverers = [[] for _ in range(len(all_words))]
             
             for i, w in enumerate(all_words):
                 m = 0
                 for nb in get_neighbors(w):
                     m |= (1 << nb)
                 masks[i] = m
                 for nb in get_neighbors(w):
                     coverers[nb].append(i)
             
             uncovered = (1 << len(all_words)) - 1
             code = []
             
             while uncovered:
                 best_idx = -1
                 best_count = -1
                 # Optimization: only check candidates that cover at least one uncovered bit
                 # Simple loop
                 for i in range(len(all_words)):
                     if not (masks[i] & uncovered):
                         continue
                     count = bin(masks[i] & uncovered).count('1')
                     if count > best_count:
                         best_count = count
                         best_idx = i
                 
                 if best_idx == -1:
                     break
                     
                 code.append(best_idx)
                 uncovered &= ~masks[best_idx]
                 
                 # Simple local search: remove redundant
                 changed = True
                 while changed:
                     changed = False
                     for i in range(len(code)-1, -1, -1):
                         ci = code[i]
                         # Check if ci is redundant
                         redundant = True
                         m_ci = masks[ci]
                         # Elements covered only by ci
                         # We need to check if any x in m_ci has cover_count == 1
                         # We don't have cover_count here easily in this simple loop.
                         # We'll skip local search in fallback for simplicity/speed, 
                         # or implement a quick check.
                         # Actually, just returning greedy is okay for fallback.
                         pass
                         
             return [list(all_words[i]) for i in code]
        else:
             # Fallback to seed logic if not the specific instance
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific Optimized Solver for q=3, n=7, radius=1
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    # Total words: 3^7 = 2187
    N = 2187
    # Generate all words and map to index
    all_words = list(itertools.product(range(3), repeat=7))
    # idx = w[0]*3^6 + ... + w[6]
    # Since it's base 3, we can compute index directly
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx
        
    # Precompute masks
    # mask[i] is a bitmask of neighbors of word i
    masks = [0] * N
    # coverers[x] is list of indices i such that x is in mask[i]
    coverers = [[] for _ in range(N)]
    
    for i, w in enumerate(all_words):
        m = 0
        # Neighbor: self
        m |= (1 << i)
        # Neighbors: change 1 position
        for pos in range(7):
            old_val = w[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = word_to_idx(tuple(new_w))
                m |= (1 << j)
        masks[i] = m
        # Update coverers
        # Iterate through bits set in m? 
        # Or just iterate neighbors again?
        # Iterating bits in a 2187-bit int is slow in Python.
        # Better to store neighbor lists too?
        # Or just compute coverers by iterating all i and setting coverers for neighbors.
        # We already have m.
        # Let's just do a second pass or store neighbor lists.
        # Storing neighbor lists (size 15) is memory efficient (2187 * 15 ints).
        pass
        
    # Re-do precomputation to fill coverers efficiently
    # We can just iterate i, and for each neighbor j, append i to coverers[j]
    # We need the list of neighbors for each i.
    neigh_lists = []
    for i, w in enumerate(all_words):
        neighs = [i]
        for pos in range(7):
            old_val = w[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = word_to_idx(tuple(new_w))
                neighs.append(j)
        neigh_lists.append(neighs)
        # Update coverers
        for j in neighs:
            coverers[j].append(i)
            
    # 2. Greedy Initialization
    def greedy_init():
        uncovered = (1 << N) - 1
        code = []
        # To avoid bias, we can shuffle the order of checks if counts are equal?
        # Standard greedy: pick max coverage.
        while uncovered:
            best_idx = -1
            best_count = -1
            # Optimization: If uncovered is small, maybe check only candidates covering uncovered bits?
            # All candidates cover some bits.
            for i in range(N):
                # Quick check: if no overlap, skip
                if not (masks[i] & uncovered):
                    continue
                count = bin(masks[i] & uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~masks[best_idx]
        return code

    best_code = greedy_init()
    best_size = len(best_code)
    
    # 3. Local Search
    def local_search(code, time_limit):
        # Convert to list of indices
        # Maintain cover_count
        cover_count = [0] * N
        for c in code:
            for x in neigh_lists[c]:
                cover_count[x] += 1
                
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Pass 1: Remove Redundant
            # Check each codeword
            for k in range(len(code) - 1, -1, -1):
                c = code[k]
                # Check if any x in neigh_lists[c] has cover_count[x] == 1
                is_redundant = True
                for x in neigh_lists[c]:
                    if cover_count[x] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    for x in neigh_lists[c]:
                        cover_count[x] -= 1
                    code.pop(k)
                    improved = True
                    # Restart loop to check others with new counts
                    break
                    
            if time.time() >= time_limit:
                break
                
            # Pass 2: 2-Swap (Replace 2 with 1)
            # We look for i, j such that we can replace {i, j} with {k}
            # This reduces size by 1.
            # We iterate over pairs. To speed up, we can stop early if we find a good swap.
            # But we want to find *any* improvement.
            
            # Optimization: Only consider pairs where at least one is "critical" or overlapping?
            # Actually, if neither is critical, they would have been removed in Pass 1.
            # So at least one has a critical element.
            
            n_code = len(code)
            found_swap = False
            for i in range(n_code):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_code):
                    # Calculate G_ij: elements covered by i or j but not by others
                    # G_ij = (neigh[i] U neigh[j]) \ (Union others)
                    # Using cover_count:
                    # x is in G_ij if:
                    # 1. x in neigh[i] and x not in neigh[j] and cover_count[x] == 1
                    # 2. x in neigh[j] and x not in neigh[i] and cover_count[x] == 1
                    # 3. x in neigh[i] and x in neigh[j] and cover_count[x] == 2
                    
                    # Efficient way:
                    # Build mask of G_ij
                    # We can iterate through the union of neighbors?
                    # Union size <= 30.
                    # Let's create a set of candidates for G_ij
                    # Actually, we can just check the condition for each x in neigh[i] U neigh[j]
                    
                    # To avoid set creation overhead, we can use a temporary array or just check logic.
                    # Let's identify the "gap" elements.
                    gap_mask = 0
                    # Check elements in neigh[i]
                    for x in neigh_lists[code[i]]:
                        if cover_count[x] == 1:
                            # Must be covered by i (since count is 1)
                            # Is it covered by j?
                            # If not covered by j, it's in gap.
                            # If covered by j, count would be at least 2 (from i and j) UNLESS i==j (impossible)
                            # Wait, if x is in neigh[i] and neigh[j], and cover_count[x] == 1, that's a contradiction
                            # because i and j are distinct and both cover x, so count >= 2.
                            # So if cover_count[x] == 1, x is NOT in neigh[j].
                            gap_mask |= (1 << x)
                            
                    for x in neigh_lists[code[j]]:
                        if cover_count[x] == 1:
                            # Similarly, x is not in neigh[i]
                            gap_mask |= (1 << x)
                            
                    # Now check intersections
                    # We need to check x in neigh[i] AND neigh[j] with cover_count[x] == 2
                    # We can iterate neigh[i] and check if in neigh[j]
                    # To make "in neigh[j]" fast, we can use a set or bitmask.
                    # neigh[j] is small.
                    set_j = set(neigh_lists[code[j]])
                    for x in neigh_lists[code[i]]:
                        if x in set_j and cover_count[x] == 2:
                            gap_mask |= (1 << x)
                            
                    if gap_mask == 0:
                        # We can remove both i and j
                        # Remove j first to keep indices valid? Or just mark them.
                        # We will modify the list.
                        # Remove code[j] and code[i]
                        # Update counts
                        for x in neigh_lists[code[i]]:
                            cover_count[x] -= 1
                        for x in neigh_lists[code[j]]:
                            cover_count[x] -= 1
                        code.pop(j)
                        code.pop(i)
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Look for k such that gap_mask is subset of mask[k]
                        # Candidates for k must cover all bits in gap_mask
                        # Find intersection of coverers[x] for x in gap_mask
                        # gap_mask has at most 30 bits.
                        # Pick the first x in gap_mask to get initial candidate list
                        # Then intersect with coverers of other x in gap_mask
                        
                        # Extract bits from gap_mask
                        # We can just iterate through all k in 0..N-1?
                        # N=2187.
                        # For each k, check if (masks[k] & gap_mask) == gap_mask
                        # This is 2187 checks.
                        # 2187 * (n_code^2 / 2) operations.
                        # If n_code=200, 20k pairs. 20k * 2187 = 44M checks.
                        # Each check is a bitwise AND and comparison.
                        # In Python, this might be slow (10-20s?).
                        # We need to optimize the search for k.
                        
                        # Optimization: Use the intersection of coverers.
                        # Get list of x in gap_mask
                        gap_elements = []
                        gm = gap_mask
                        while gm:
                            lsb = gm & -gm
                            x = lsb.bit_length() - 1
                            gap_elements.append(x)
                            gm ^= lsb
                            
                        if not gap_elements:
                            continue
                            
                        # Start with coverers of the first element
                        candidates = set(coverers[gap_elements[0]])
                        for x in gap_elements[1:]:
                            candidates &= set(coverers[x])
                            if not candidates:
                                break
                                
                        # Filter out i and j
                        candidates.discard(code[i])
                        candidates.discard(code[j])
                        
                        if candidates:
                            # Pick a candidate (e.g., the one that covers most other uncovered? 
                            # Or just any. Let's pick the one with max total coverage?
                            # Or just the first one.
                            # To be safe, pick one that doesn't cause immediate issues?
                            # Any valid k works.
                            k = next(iter(candidates))
                            
                            # Perform swap
                            # Remove i and j
                            for x in neigh_lists[code[i]]:
                                cover_count[x] -= 1
                            for x in neigh_lists[code[j]]:
                                cover_count[x] -= 1
                            code.pop(j)
                            code.pop(i)
                            
                            # Add k
                            code.append(k)
                            for x in neigh_lists[k]:
                                cover_count[x] += 1
                                
                            improved = True
                            found_swap = True
                            break
                            
                if found_swap:
                    break
                    
        return code

    # Run Local Search
    # We might want to try multiple restarts or randomizations if time permits.
    # But Greedy+LS is deterministic.
    # We can randomize the tie-breaking in Greedy.
    
    best_code = local_search(best_code, deadline)
    best_size = len(best_code)
    
    # If time remains, try a few random restarts
    # Randomize by shuffling the order of candidates in Greedy when counts are equal?
    # Or start with a random valid cover?
    
    # Let's try 5 random restarts with randomized Greedy tie-breaking
    for restart in range(5):
        if time.time() >= deadline - 5.0:
            break
            
        # Randomized Greedy
        uncovered = (1 << N) - 1
        code = []
        while uncovered:
            best_indices = []
            best_count = -1
            for i in range(N):
                if not (masks[i] & uncovered):
                    continue
                count = bin(masks[i] & uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_indices = [i]
                elif count == best_count:
                    best_indices.append(i)
            
            if not best_indices:
                break
                
            # Pick randomly from best_indices
            idx = random.choice(best_indices)
            code.append(idx)
            uncovered &= ~masks[idx]
            
        code = local_search(code, deadline)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    # Convert indices back to words
    return [list(all_words[i]) for i in best_code]
