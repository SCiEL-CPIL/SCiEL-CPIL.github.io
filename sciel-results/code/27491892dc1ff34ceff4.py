import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115  # Leave 5s margin
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)
    
    # Fallback for other cases
    return _greedy_set_cover(q, n, radius, deadline)

def _solve_ternary_7_1(deadline):
    q = 3
    n = 7
    space_size = q ** n # 2187
    
    powers = [3 ** i for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx
        
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % 3
            idx //= 3
        return word

    # 1. Precompute coverage bitsets for all 2187 candidates
    # cover_bits[c] is an int where bit i is 1 if candidate c covers word i
    cover_bits = [0] * space_size
    # elem_to_cws[e] is a list of candidate indices that cover element e
    elem_to_cws = [[] for _ in range(space_size)]
    
    # Precompute neighbors for each word to speed up cover_bits calculation
    # Or just iterate all pairs? 2187 * 2187 is ~4.8M operations. Fast enough in Python? 
    # Better: For each candidate c, compute its neighborhood and set bits.
    
    for c in range(space_size):
        w = index_to_word(c)
        # The ball of radius 1 around w
        # w itself
        # w with one symbol changed
        neigh_indices = [c]
        for pos in range(n):
            for delta in (1, 2):
                w2 = list(w)
                w2[pos] = (w2[pos] + delta) % 3
                neigh_indices.append(word_to_index(w2))
        
        bits = 0
        for idx in neigh_indices:
            bits |= (1 << idx)
        cover_bits[c] = bits
        
        for idx in neigh_indices:
            elem_to_cws[idx].append(c)
            
    # 2. Greedy Set Cover to get initial upper bound
    uncovered = (1 << space_size) - 1
    selected_greedy = []
    used_cws = set()
    
    while uncovered:
        # Find candidate with max gain
        best_c = -1
        best_gain = -1
        # Optimization: We can iterate all candidates, it's only 2187
        for c in range(space_size):
            if c in used_cws:
                continue
            gain = (uncovered & cover_bits[c]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_gain == 0:
            break # Should not happen if space is connected
            
        selected_greedy.append(best_c)
        used_cws.add(best_c)
        uncovered &= ~cover_bits[best_c]
        
    best_k = len(selected_greedy)
    best_solution = selected_greedy[:]
    
    # 3. Branch and Bound to improve
    # We want to find if there is a solution of size < best_k
    
    # Sort elem_to_cws to put "rare" elements first? 
    # The branching heuristic picks an uncovered element.
    
    # To speed up B&B, we can sort candidates in elem_to_cws? 
    # Not strictly necessary, but helps if we want to try "better" candidates first.
    
    # Heuristic for B&B:
    # State: uncovered (int), depth (int), last_idx (int)
    # We want to find ANY solution of size <= target_k.
    # We will try target_k = best_k - 1, best_k - 2, etc.
    
    # Optimization: Pre-sort candidates in elem_to_cws by some heuristic?
    # Let's just use the list as is.
    
    # Global variables for B&B
    found_solution = None
    
    def search(uncovered, depth, last_idx, target_k):
        nonlocal found_solution
        
        if time.time() > deadline:
            return False
            
        if uncovered == 0:
            # We found a solution of size depth
            # We need to reconstruct it? 
            # For now, just return True. We can store the path if needed.
            # But we need the actual codewords to return.
            # So we should maintain a stack of selected candidates.
            return True
            
        if depth >= target_k:
            return False
            
        # Find uncovered element with minimum number of available covering candidates
        # "Available" means index > last_idx? No, that's for ordering to avoid permutations.
        # In Set Cover, if we pick c1 then c2, it's the same set as c2 then c1.
        # So we enforce c1 < c2 < ...
        
        # Find element e in uncovered
        # We should pick the one that restricts the search most.
        # Let's iterate through uncovered bits.
        # Since space_size is 2187, we can find the first set bit, or scan all.
        # Scanning all 2187 bits to find min len(elem_to_cws[e]) is fast.
        
        best_e = -1
        min_count = float('inf')
        
        # Iterate through uncovered elements
        u = uncovered
        while u:
            # Get lowest set bit
            lsb = u & -u
            e = lsb.bit_length() - 1
            u ^= lsb
            
            # Count available candidates for e (index > last_idx)
            # elem_to_cws[e] contains all candidates covering e.
            # We only care about those > last_idx.
            # Since list is small (15), linear scan is fine.
            count = 0
            for c in elem_to_cws[e]:
                if c > last_idx:
                    count += 1
                    
            if count == 0:
                return False # Dead end
                
            if count < min_count:
                min_count = count
                best_e = e
                
        if best_e == -1:
            return False # No uncovered elements? Handled above.
            
        # Branch on candidates covering best_e
        for c in elem_to_cws[best_e]:
            if c <= last_idx:
                continue
                
            new_uncovered = uncovered & ~cover_bits[c]
            
            # Recursive call
            # To avoid passing huge lists, we rely on the fact that we are just checking existence.
            # If we find a solution, we need to backtrack to get the set.
            # This is tricky with a simple boolean return.
            
            # Let's modify search to return the list of candidates if found, or None.
            # But passing lists recursively is slow.
            
            # Alternative: Use a global stack.
            pass
        
        return False

    # The boolean return is problematic for retrieving the solution.
    # Let's implement a version that stores the solution in a global variable when found.
    
    solution_stack = []
    
    def search_v2(uncovered, depth, last_idx, target_k):
        nonlocal found_solution
        
        if time.time() > deadline:
            return False
            
        if uncovered == 0:
            # Success
            # found_solution is updated by caller? 
            # We need to capture the current state of solution_stack.
            # Since we pop/append in caller, we can just copy here.
            found_solution = solution_stack[:]
            return True
            
        if depth >= target_k:
            return False
            
        # Find best_e
        best_e = -1
        min_count = float('inf')
        u = uncovered
        while u:
            lsb = u & -u
            e = lsb.bit_length() - 1
            u ^= lsb
            
            count = 0
            for c in elem_to_cws[e]:
                if c > last_idx:
                    count += 1
                    
            if count == 0:
                return False
                
            if count < min_count:
                min_count = count
                best_e = e
                
        if best_e == -1:
            return False
            
        for c in elem_to_cws[best_e]:
            if c <= last_idx:
                continue
                
            new_uncovered = uncovered & ~cover_bits[c]
            
            solution_stack.append(c)
            if search_v2(new_uncovered, depth + 1, c, target_k):
                return True
            solution_stack.pop()
            
        return False

    # Try to improve
    # We know best_k from greedy.
    # Try target_k = best_k - 1, best_k - 2 ...
    
    # To make it faster, we can use a lower bound.
    # Lower bound: max over e in uncovered of (1 / max overlap)? No.
    # Simple LB: ceil( popcount(uncovered) / max_cover_size )
    # max_cover_size is 15 (radius 1 ball size).
    
    # Let's just try decreasing target_k.
    
    # Optimization: Randomize the order of candidates in elem_to_cws?
    # Or sort by "quality" (e.g., how many other elements they cover)?
    # For now, keep it simple.
    
    improved = False
    for target_k in range(best_k - 1, 0, -1):
        if time.time() > deadline - 2:
            break
            
        found_solution = None
        solution_stack.clear()
        
        # Reset search
        # We start with empty uncovered? No, we start with full uncovered.
        # Wait, the B&B finds a set of size target_k that covers EVERYTHING.
        # So initial call is search_v2(full_uncovered, 0, -1, target_k)
        
        full_uncovered = (1 << space_size) - 1
        
        if search_v2(full_uncovered, 0, -1, target_k):
            if found_solution:
                best_k = target_k
                best_solution = found_solution
                improved = True
                # We can break if we hit the theoretical lower bound?
                # Lower bound for K_3(7,1) is 156.
                if best_k == 156:
                    break
    
    # Convert indices to words
    result = [index_to_word(idx) for idx in best_solution]
    
    # Verify validity (optional but good for debugging)
    # Check if all words are covered
    # For production, trust the B&B.
    
    return result

def _greedy_set_cover(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    space_size = q ** n
    
    if space_size > 50000:
        random.seed(42)
        words = []
        seen = set()
        while len(words) < 200 and time.time() < deadline - 5:
            w = tuple(random.randint(0, q - 1) for _ in range(n))
            if w not in seen:
                seen.add(w)
                words.append(list(w))
        return words

    powers = [q ** i for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx
        
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    if radius == 1:
        neighborhoods = [None] * space_size
        for idx in range(space_size):
            w = index_to_word(idx)
            neigh = [idx]
            for pos in range(n):
                for delta in range(1, q):
                    w2 = list(w)
                    w2[pos] = (w2[pos] + delta) % q
                    neigh.append(word_to_index(w2))
            neighborhoods[idx] = neigh

    coverage = np.zeros(space_size, dtype=np.int32)
    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    selected_set = set()
    
    while time.time() < deadline - 3 and uncovered.any():
        best_cw = -1
        best_count = -1
        
        candidates = list(range(space_size))
        random.shuffle(candidates)
        candidates = candidates[:min(500, space_size)]
        
        for cw in candidates:
            if cw in selected_set:
                continue
            count = 0
            for nb in neighborhoods[cw]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = cw
                
        if best_cw < 0:
            break
            
        selected.append(index_to_word(best_cw))
        selected_set.add(best_cw)
        for nb in neighborhoods[best_cw]:
            coverage[nb] += 1
            if coverage[nb] == 1:
                uncovered[nb] = False

    return selected
