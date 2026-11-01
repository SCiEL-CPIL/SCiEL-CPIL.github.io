import time
import random
import itertools
import sys

def construct(q, n, radius):
    # Handle the specific known hard case efficiently
    if q == 3 and n == 6 and radius == 1:
        return solve_3_6_1()
    
    # Fallback for other parameters: Simple Greedy with bitset optimization
    # For general cases, we can use a similar ILS approach if N is small enough
    # but the task focuses on 3,6,1.
    # If N is large, return a trivial valid code or heuristic.
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    if N > 50000:
        return [list(w) for w in all_words]
        
    return solve_general(q, n, radius, all_words, N)

def solve_3_6_1():
    q = 3
    n = 6
    N = q ** n  # 729
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute sphere masks
    # Map word to index
    index_map = {w: i for i, w in enumerate(all_words)}
    
    sphere_masks = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = index_map[tuple(new_w)]
                mask |= (1 << j)
        sphere_masks[i] = mask
    
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_solution = None
    best_size = N
    
    # Initial Greedy Solution
    current_solution = greedy_initial(sphere_masks, N)
    current_size = len(current_solution)
    if current_size < best_size:
        best_size = current_size
        best_solution = current_solution[:]
    
    # Iterated Local Search
    # We maintain a list of codeword indices
    # We need to track which are covered? No, we can recompute uncovered mask from solution.
    # But for speed, we can maintain uncovered mask incrementally.
    # However, repair and reduction change the set significantly.
    # Let's use a function to get uncovered mask from a list of codewords.
    
    def get_uncovered_mask(codewords):
        covered = 0
        for c in codewords:
            covered |= sphere_masks[c]
        return ((1 << N) - 1) ^ covered
        
    # Optimize: Precompute full_mask
    full_mask = (1 << N) - 1
    
    def get_uncovered_mask_fast(codewords):
        covered = 0
        for c in codewords:
            covered |= sphere_masks[c]
        return full_mask ^ covered

    # Reduction Phase: Remove redundant codewords
    def reduce_solution(codewords):
        # codewords is a list of indices
        # We try to remove each one if possible
        # To speed up, we can shuffle
        c_list = codewords[:]
        c_list.sort(reverse=True) # Remove from end? Or random.
        random.shuffle(c_list)
        
        # We need to know what is covered by the rest.
        # Simple approach: for each candidate, check if it covers any point NOT covered by others.
        # This is O(K^2 * bitops). K is ~80. 80^2 = 6400 ops. Fast.
        
        new_list = []
        for c in c_list:
            # Check if c is redundant given new_list
            # Covered by new_list:
            covered_by_others = 0
            for o in new_list:
                covered_by_others |= sphere_masks[o]
            
            # Points covered by c but not by others:
            unique_to_c = sphere_masks[c] & (~covered_by_others)
            
            if unique_to_c == 0:
                # c is redundant, skip it
                continue
            else:
                new_list.append(c)
                
        return new_list

    # Repair Phase: Cover uncovered points greedily
    def repair_solution(codewords, uncovered_mask):
        # codewords: list of indices currently in solution
        # uncovered_mask: bits of points not covered
        # We add codewords to cover uncovered_mask
        # We prefer codewords that cover many uncovered points.
        # We should also consider codewords already in the solution?
        # If a codeword is already in the solution, adding it again is useless (set semantics).
        # So we only add NEW codewords.
        
        current_set = set(codewords)
        new_codewords = []
        
        while uncovered_mask != 0:
            best_c = -1
            best_gain = -1
            
            # Iterate through all possible codewords? Or just those not in current_set?
            # Checking all 729 is fast.
            for c in range(N):
                if c in current_set:
                    continue
                gain = (sphere_masks[c] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                break # Should not happen if valid problem
                
            new_codewords.append(best_c)
            current_set.add(best_c)
            uncovered_mask &= ~sphere_masks[best_c]
            
        return codewords + new_codewords

    # Main ILS Loop
    current = current_solution
    current_size = len(current)
    
    while time.time() < deadline:
        # Perturb: Remove some random codewords
        # Number to remove: proportional to size
        num_remove = max(1, int(len(current) * 0.1))
        
        # Randomly select indices to remove
        indices_to_remove = random.sample(range(len(current)), num_remove)
        
        # Create a new list without these
        # We need to be careful with indices
        new_current = current[:]
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            new_current.pop(idx)
            
        # Get uncovered mask
        uncovered = get_uncovered_mask_fast(new_current)
        
        # Repair
        repaired = repair_solution(new_current, uncovered)
        
        # Reduce
        reduced = reduce_solution(repaired)
        
        # Check size
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_solution = reduced[:]
            
        # Update current
        # Accept if better, or with some probability if equal/worse?
        # For finding minimum, strict improvement is good, but might get stuck.
        # Let's use a simple hill climbing: accept if <= current_size?
        # Or just always move to the repaired/reduced state to allow exploration?
        # Usually, ILS accepts if better. If stuck, restart?
        # Let's try: Accept if len(reduced) <= len(current).
        if len(reduced) <= len(current):
            current = reduced
            current_size = len(current)
        else:
            # Maybe keep current to avoid drifting to worse solutions?
            # But ILS relies on moving. Let's force move to allow escape?
            # If we don't move, we are stuck in local optimum of the current "basin".
            # Perturbation is the escape mechanism.
            # If repair+reduce makes it worse, it means the perturbation was bad or repair was bad.
            # Let's stick to current if it got worse, to preserve the good solution.
            pass

    # Final reduction on best_solution just in case
    if best_solution:
        best_solution = reduce_solution(best_solution)
        
    return [list(all_words[c]) for c in best_solution]

def solve_general(q, n, radius, all_words, N):
    # Similar logic, but radius might be > 1
    # Precompute spheres
    index_map = {w: i for i, w in enumerate(all_words)}
    
    sphere_masks = [0] * N
    # Generate all neighbors within radius
    # This can be expensive for large radius, but N is small here.
    for i, w in enumerate(all_words):
        mask = 0
        # Use product to generate neighbors? Or recursive.
        # For small N, just check all pairs? No, that's O(N^2 * n).
        # N=729, N^2 ~ 500k. 500k * 6 = 3M ops. Acceptable.
        for j, v in enumerate(all_words):
            dist = sum(1 for k in range(n) if w[k] != v[k])
            if dist <= radius:
                mask |= (1 << j)
        sphere_masks[i] = mask
        
    # Run ILS
    # (Reusing the logic from solve_3_6_1 would be better, but let's keep it simple for fallback)
    # Actually, let's just call a generic version of the ILS.
    
    start_time = time.time()
    deadline = start_time + 110
    
    def greedy_initial(sphere_masks, N):
        uncovered = (1 << N) - 1
        solution = []
        while uncovered != 0:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = (sphere_masks[c] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1: break
            solution.append(best_c)
            uncovered &= ~sphere_masks[best_c]
        return solution

    current = greedy_initial(sphere_masks, N)
    best_solution = current[:]
    best_size = len(current)
    
    full_mask = (1 << N) - 1
    def get_uncovered_mask_fast(codewords):
        covered = 0
        for c in codewords:
            covered |= sphere_masks[c]
        return full_mask ^ covered

    def reduce_solution(codewords):
        c_list = codewords[:]
        random.shuffle(c_list)
        new_list = []
        for c in c_list:
            covered_by_others = 0
            for o in new_list:
                covered_by_others |= sphere_masks[o]
            unique_to_c = sphere_masks[c] & (~covered_by_others)
            if unique_to_c == 0:
                continue
            else:
                new_list.append(c)
        return new_list

    def repair_solution(codewords, uncovered_mask):
        current_set = set(codewords)
        new_codewords = []
        while uncovered_mask != 0:
            best_c = -1
            best_gain = -1
            for c in range(N):
                if c in current_set:
                    continue
                gain = (sphere_masks[c] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1 or best_gain == 0:
                break
            new_codewords.append(best_c)
            current_set.add(best_c)
            uncovered_mask &= ~sphere_masks[best_c]
        return codewords + new_codewords

    while time.time() < deadline:
        num_remove = max(1, int(len(current) * 0.1))
        indices_to_remove = random.sample(range(len(current)), num_remove)
        new_current = current[:]
        for idx in sorted(indices_to_remove, reverse=True):
            new_current.pop(idx)
            
        uncovered = get_uncovered_mask_fast(new_current)
        repaired = repair_solution(new_current, uncovered)
        reduced = reduce_solution(repaired)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_solution = reduced[:]
            
        if len(reduced) <= len(current):
            current = reduced

    if best_solution:
        best_solution = reduce_solution(best_solution)
        
    return [list(all_words[c]) for c in best_solution]

def greedy_initial(sphere_masks, N):
    uncovered = (1 << N) - 1
    solution = []
    while uncovered != 0:
        best_c = -1
        best_gain = -1
        for c in range(N):
            gain = (sphere_masks[c] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1:
            break
        solution.append(best_c)
        uncovered &= ~sphere_masks[best_c]
    return solution
