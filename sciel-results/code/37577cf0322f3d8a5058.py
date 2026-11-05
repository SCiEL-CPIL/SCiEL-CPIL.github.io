"""
Optimized ternary covering code construction for K_3(6,1).
Combines direct-sum initialization with greedy and randomized pruning.
"""
import time
import random
import itertools
import sys

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # 5s margin before 120s limit
    
    # Handle the specific case q=3, n=6, radius=1 with specialized logic
    if q == 3 and n == 6 and radius == 1:
        return _construct_k3_6_1(deadline)
    
    # Fallback for other parameters: simple greedy set cover
    return _greedy_construct(q, n, radius, deadline - time.time())

def _greedy_construct(q, n, radius, time_limit):
    """Simple greedy fallback for non-standard parameters."""
    deadline = time.time() + time_limit
    total_words = q ** n
    if total_words > 100000:
        # Too large for simple greedy in time, return empty or small sample
        # But task implies small n usually. If huge, we can't do much.
        return []
        
    # Precompute neighbors
    # For radius 1, neighbors are self + q-1 changes per position
    neighbors = []
    for i in range(total_words):
        w = [(i // (q ** (n - 1 - pos))) % q for pos in range(n)]
        nbrs = [i]
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        nw = w[:]
                        nw[pos] = new_val
                        idx = 0
                        for j in range(n):
                            idx = idx * q + nw[j]
                        nbrs.append(idx)
        neighbors.append(nbrs)
        
    covered = [False] * total_words
    code = []
    
    while time.time() < deadline:
        # Find word with max uncovered neighbors
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            if covered[i]:
                continue
            gain = 0
            for nbr in neighbors[i]:
                if not covered[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if gain == 0:
                break # Optimization: if we hit 0, can't get better than 0? No, we want max.
                # Actually, if all remaining are covered, we break.
                # But we need to check all if best_gain is 0.
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code.append(best_idx)
        for nbr in neighbors[best_idx]:
            covered[nbr] = True
        
    result = []
    for idx in code:
        w = [(idx // (q ** (n - 1 - pos))) % q for pos in range(n)]
        result.append(w)
    return result

def _construct_k3_6_1(deadline):
    """
    Specialized constructor for K_3(6,1).
    1. Generate 81-word code via direct sum of tetracode and 2D space.
    2. Prune redundant codewords using coverage counts.
    3. Use randomized restarts to find minimal subset (target 73).
    """
    n = 6
    q = 3
    total_words = 729
    
    # 1. Generate initial code: Direct Sum of Tetracode (9 words len 4) and 3^2 (9 words len 2)
    # Tetracode: (a, b, a+b, a+2b) mod 3
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(tuple(h) + tail)
    
    # Map words to indices for fast lookup
    def word_to_index(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    # Precompute coverage sets for each codeword in initial_code
    # codeword_covers[i] = set of word indices covered by codeword i
    codeword_covers = []
    for c in initial_code:
        covers = set()
        c_idx = word_to_index(c)
        covers.add(c_idx)
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_w = list(c)
                    new_w[pos] = val
                    covers.add(word_to_index(tuple(new_w)))
        codeword_covers.append(covers)
    
    # Precompute for each word, which codewords cover it
    # covered_by[word_idx] = list of codeword indices
    covered_by = [[] for _ in range(total_words)]
    for ci, covers in enumerate(codeword_covers):
        for wi in covers:
            covered_by[wi].append(ci)
    
    def prune_code(active_indices, time_limit):
        """
        Given a list of active codeword indices from initial_code,
        greedily remove redundant ones.
        Returns the list of remaining indices.
        """
        active = set(active_indices)
        
        # Iterate until no more removals possible or time runs out
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Try to remove each active codeword
            # Order matters. We can randomize or use fixed order.
            # For deterministic pruning, fixed order. For random restart, shuffled order.
            items_to_check = list(active)
            for ci in items_to_check:
                if time.time() > time_limit:
                    break
                if ci not in active:
                    continue
                
                is_redundant = True
                for wi in codeword_covers[ci]:
                    # Check if wi is covered by another active codeword
                    covered_by_other = False
                    for cj in covered_by[wi]:
                        if cj != ci and cj in active:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    active.remove(ci)
                    changed = True
        
        return list(active)

    best_code = None
    best_size = len(initial_code)
    
    # Strategy: 
    # 1. Run one deterministic prune
    # 2. Run multiple randomized prunes to escape local minima
    
    # Deterministic pass
    current_time = time.time()
    if current_time < deadline - 5.0:
        pruned = prune_code(list(range(len(initial_code))), current_time + 5.0)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned
    
    # Randomized passes
    # We have ~100s left. Each prune is fast (81 items, 729 words).
    # Let's do as many as possible.
    
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        # Random shuffle of the initial code indices to break symmetry in pruning
        indices = list(range(len(initial_code)))
        random.shuffle(indices)
        
        # We can also try starting from a slightly perturbed set? 
        # No, pruning from full set is safest for validity.
        
        pruned = prune_code(indices, time.time() + 1.0) # Tight time limit per attempt
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned
            # If we hit the known optimal 73, we can stop early? 
            # Yes, 73 is the best known. 71 is lower bound. 
            # If we hit 73, we should try to see if 72 is possible? Unlikely.
            # But let's keep searching for a bit in case 72 exists or to be sure.
            if best_size == 73:
                # Do a few more to be safe, but break soon
                if attempts > 50:
                    break

    # If best_code is None (shouldn't happen), fallback to initial
    if best_code is None:
        best_code = list(range(len(initial_code)))
    
    # Convert indices back to words
    result = []
    for idx in best_code:
        result.append(list(initial_code[idx]))
    
    return result
