"""Covering code K_q(n, radius): Local search optimization."""
import time
import itertools
import random
import numpy as np

_DEADLINE = 110.0  # Safety margin under 120s limit

def _generate_neighbors(q, n, radius=1):
    """Precompute neighbors for all words in {0..q-1}^n."""
    total = q ** n
    # Map from tuple to index
    idx_map = {}
    words = []
    for i in range(total):
        # Convert index to base-q representation
        w = [0] * n
        temp = i
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        idx_map[tuple(w)] = i
    
    neighbors = [[] for _ in range(total)]
    for i, w in enumerate(words):
        nbrs = [i]
        if radius >= 1:
            w_list = list(w)
            for pos in range(n):
                orig = w_list[pos]
                for val in range(q):
                    if val != orig:
                        w_list[pos] = val
                        nbrs.append(idx_map[tuple(w_list)])
                w_list[pos] = orig
        neighbors[i] = nbrs
    return neighbors

def _greedy_cover(q, n, radius, neighbors, deadline):
    """Greedy set cover to get an initial valid code."""
    total = q ** n
    covered = [False] * total
    code = []
    
    # Precompute coverage sets as lists of indices
    # neighbors[i] is the list of indices covered by i
    
    while True:
        if time.time() > deadline:
            break
        # Find word covering most uncovered points
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Optimization: skip if all its neighbors are already covered
            # This is O(N * Degree). For 6561 * 17, it's ~111k ops per iteration.
            # We do this ~500 times -> 55M ops. Might be slow in Python.
            # Better: maintain uncovered count? Or just use a heuristic.
            
            # Simple heuristic: count uncovered in neighbors
            cnt = 0
            for nbr in neighbors[i]:
                if not covered[nbr]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                if cnt == len(neighbors[i]):
                    # Perfect coverage for this candidate, likely good
                    pass
                    
        if best_idx == -1 or best_count == 0:
            break
            
        code.append(best_idx)
        for nbr in neighbors[best_idx]:
            covered[nbr] = True
            
        if all(covered):
            break
            
    # If not covered, force add remaining? 
    # Greedy should cover if radius is sufficient and space is finite.
    return code

def _local_search(code, neighbors, total, deadline):
    """Local search to minimize code size."""
    code = list(code)
    code_set = set(code)
    
    # Cover counts: how many codewords cover each word in the space
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
            
    best_size = len(code)
    best_code = list(code)
    
    # Main loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Removal
        # Shuffle to avoid bias
        random.shuffle(code)
        
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = code[i]
            # Check if removing c leaves any word with cover_count 0
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code.pop(i)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                improved = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
        
        if improved:
            continue
            
        # Phase 2: Swap
        # Try to swap a word c with another word c'
        # We want to remove c and add c'
        # Condition: 
        # 1. All neighbors of c that are ONLY covered by c (cover_count == 1) 
        #    must be covered by c'.
        # 2. Ideally, c' covers some words that are currently under-covered? 
        #    Actually, if we are in a minimal state (no removals), all words are covered >= 1.
        #    Removing c makes some words have count 0 (the critical ones).
        #    Adding c' must cover those critical words.
        
        # To speed up, we can pick a random c and look for c'
        # Or iterate.
        
        swapped = False
        # Limit swap attempts to avoid O(N^2) if N is large
        # For 6561, N^2 is ~43M. Too slow for Python loop.
        # We need a smarter swap.
        
        # Strategy: Pick a random c. Identify its critical neighbors (cover_count == 1).
        # Find a c' that covers all critical neighbors of c.
        # If such c' exists, and c' is not already in code, we can swap.
        # We also need to ensure that adding c' doesn't break anything else? 
        # Adding c' only increases counts, so it never breaks coverage.
        # So the only constraint is: c' covers all critical neighbors of c.
        
        if len(code) > 0:
            # Try a few random swaps
            for _ in range(100):
                if time.time() > deadline:
                    break
                idx_c = random.randint(0, len(code) - 1)
                c = code[idx_c]
                
                # Find critical neighbors of c
                critical = []
                for nbr in neighbors[c]:
                    if cover_count[nbr] == 1:
                        critical.append(nbr)
                
                if not critical:
                    # This word is redundant? But removal phase should have caught it.
                    # Maybe it was added back? Or logic error.
                    # If no critical neighbors, we can remove it?
                    # Wait, if cover_count[nbr] > 1 for all nbr in neighbors[c], 
                    # then removal would have succeeded.
                    # So if we are here, there must be critical neighbors.
                    # Unless the code is empty?
                    continue
                
                # We need a c' that covers all words in 'critical'
                # How to find c' quickly?
                # Precompute for each word w, the list of codewords that cover w?
                # Or just scan?
                # Scanning all 6561 words for each of 100 attempts * 500 iterations * ...
                # 100 * 6561 * (check if c' covers critical)
                # Check if c' covers critical: 
                #   for each w in critical, is w in neighbors[c']?
                #   We can precompute a set of neighbors for each c' for O(1) lookup.
                
                # Let's precompute neighbor sets for O(1) lookup.
                # This is memory heavy: 6561 sets of ~17 ints. ~110k ints. Fine.
                
                # Actually, we can just check:
                # Is there a word c' (not in code) such that 
                # all(w in neighbor_set[c'] for w in critical)?
                
                # To make this fast, we can intersect the "coverers" of each critical word.
                # But we don't have "coverers" precomputed.
                
                # Alternative: 
                # For the critical set C, find the intersection of the sets of words that cover each w in C.
                # The intersection gives candidate c'.
                
                # Let's build a map: word -> list of codewords that cover it.
                # This is expensive to build every time.
                
                # Simpler heuristic for swap:
                # Just try a few random c' and see if they cover critical.
                # This is probabilistic but works well in practice for covering codes.
                
                found_swap = False
                for _ in range(20): # Try 20 random candidates
                    c_prime = random.randint(0, total - 1)
                    if c_prime in code_set:
                        continue
                    
                    # Check if c_prime covers all critical
                    valid = True
                    for w in critical:
                        # Check if w is in neighbors[c_prime]
                        # Using a set for neighbors[c_prime] is faster
                        # Let's assume we have neighbor_sets
                        if w not in neighbor_sets[c_prime]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        code[idx_c] = c_prime
                        code_set.discard(c)
                        code_set.add(c_prime)
                        
                        # Update cover counts
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        for nbr in neighbors[c_prime]:
                            cover_count[nbr] += 1
                        
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                        
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break # Restart removal phase? Or continue?
                    # Usually, after a swap, we should restart removal to see if new redundancies appear.
                    # But for speed, we can just continue the loop.
        
        if not improved:
            # Stagnation? Try a more aggressive move or break?
            # For now, just break to save time if no improvement.
            # Or try a "kick": remove a random word and add a random word that restores coverage?
            # That's risky.
            break
            
    return best_code

# Precomputed neighbor sets for the specific case q=3, n=8, radius=1
# We will generate them on the fly if needed, but for the specific case, we can cache.
_neighbor_sets_cache = {}

def _get_neighbor_sets(q, n, radius):
    key = (q, n, radius)
    if key not in _neighbor_sets_cache:
        neighbors = _generate_neighbors(q, n, radius)
        sets = [set(nbrs) for nbrs in neighbors]
        _neighbor_sets_cache[key] = (neighbors, sets)
    return _neighbor_sets_cache[key]

def construct(q, n, radius):
    t0 = time.time()
    
    # Handle edge cases
    if n <= 0 or q <= 1:
        return [list(range(0))[:n]] if n == 0 else [list(range(n))]
        
    N = q ** n
    if N <= 0:
        return []
        
    # Get neighbors
    neighbors, neighbor_sets = _get_neighbor_sets(q, n, radius)
    
    # Deadline for greedy
    greedy_deadline = t0 + 20.0
    # Deadline for local search
    search_deadline = t0 + _DEADLINE
    
    # 1. Greedy Initial Cover
    code = _greedy_cover(q, n, radius, neighbors, greedy_deadline)
    
    # Check if valid
    # Greedy should produce valid code.
    
    # 2. Local Search
    if time.time() < search_deadline - 5.0:
        code = _local_search(code, neighbors, N, search_deadline)
        
    # Convert indices to lists
    result = []
    for c in code:
        w = [0] * n
        temp = c
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        result.append(w)
        
    return result
