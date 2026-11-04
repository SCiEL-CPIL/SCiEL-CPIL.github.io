import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    def greedy_construct_random(seed=None):
        """
        Greedy set cover with random tie-breaking among top candidates.
        """
        if seed is not None:
            random.seed(seed)
        
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        # First pick: 0 (symmetry)
        code.append(0)
        code_set.add(0)
        coverage[neighbors_list[0]] += 1
        uncovered_mask[neighbors_list[0]] = False
        
        while uncovered_mask.any():
            best_candidates = []
            max_gain = 0
            
            for c in range(N):
                if c in code_set:
                    continue
                cnt = np.sum(uncovered_mask[neighbors_list[c]])
                if cnt > max_gain:
                    max_gain = cnt
                    best_candidates = [c]
                elif cnt == max_gain and max_gain > 0:
                    best_candidates.append(c)
            
            if max_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            code_set.add(chosen)
            coverage[neighbors_list[chosen]] += 1
            uncovered_mask[neighbors_list[chosen]] = False
            
        return code, code_set, coverage, uncovered_mask

    def greedy_construct_weighted(seed=None):
        """
        Greedy with weighted random selection to break ties more diversely.
        """
        if seed is not None:
            random.seed(seed)
        
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        code.append(0)
        code_set.add(0)
        coverage[neighbors_list[0]] += 1
        uncovered_mask[neighbors_list[0]] = False
        
        while uncovered_mask.any():
            gains = np.zeros(N, dtype=np.int32)
            for c in range(N):
                if c not in code_set:
                    gains[c] = np.sum(uncovered_mask[neighbors_list[c]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Get all candidates with max gain
            candidates = np.where(gains == max_gain)[0].tolist()
            # Weighted random: prefer candidates that cover more unique uncovered words
            # For simplicity, just pick randomly from max gain candidates
            chosen = random.choice(candidates)
            code.append(chosen)
            code_set.add(chosen)
            coverage[neighbors_list[chosen]] += 1
            uncovered_mask[neighbors_list[chosen]] = False
            
        return code, code_set, coverage, uncovered_mask

    # Run multiple greedy constructions with different strategies
    best_initial_code = None
    best_initial_size = float('inf')
    best_initial_set = None
    best_initial_cov = None
    best_initial_um = None
    
    num_restarts = 8  # More restarts for better initial basin
    
    for i in range(num_restarts):
        if time.time() > deadline - 25:  # Keep time for local search
            break
        
        if i % 2 == 0:
            c, cs, cov, um = greedy_construct_random(seed=i * 7 + 1)
        else:
            c, cs, cov, um = greedy_construct_weighted(seed=i * 13 + 3)
        
        if len(c) < best_initial_size:
            best_initial_size = len(c)
            best_initial_code = c
            best_initial_set = cs
            best_initial_cov = cov
            best_initial_um = um
            
    if best_initial_code is None:
        c, cs, cov, um = greedy_construct_random(seed=0)
        best_initial_code = c
        best_initial_set = cs
        best_initial_cov = cov
        best_initial_um = um

    code = best_initial_code
    code_set = best_initial_set
    coverage = best_initial_cov
    uncovered_mask = best_initial_um

    # ---------- Local Search ----------
    
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])
            
        return added

    def apply_code_change(removed_list, added_list):
        nonlocal code, code_set, coverage
        for c in removed_list:
            if c in code_set:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        if len(code) < k:
            return False
            
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        # Filter out any added that are in removed (net zero)
        added = [a for a in added if a not in removed]
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            apply_code_change(removed, added)
            return True
        else:
            return False

    def try_swap():
        """
        Try to swap one codeword for another (keep size same but change structure).
        This can help escape local minima by changing the configuration.
        """
        if len(code) < 2:
            return False
        
        # Pick a random codeword to potentially replace
        c_remove = random.choice(code)
        
        # Find all codewords that are neighbors of c_remove's uncovered neighbors
        # Actually, let's just try: remove c_remove, see what's uncovered, 
        # then try to cover those with a DIFFERENT codeword (not c_remove)
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # c_remove is redundant, just remove it
            apply_code_change([c_remove], [])
            return True
        
        uncovered_set = set(uncovered.tolist())
        
        # Find candidates that cover at least one uncovered word
        best_c = -1
        best_count = 0
        
        for w in uncovered_set:
            for c in neighbors_set[w]:
                if c == c_remove or c in code_set:
                    continue
                cnt = len(neighbors_set[c] & uncovered_set)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
        
        if best_c == -1:
            return False
        
        # Check if swapping reduces total uncovered (should be 0 after swap)
        # Apply swap: remove c_remove, add best_c
        apply_code_change([c_remove], [best_c])
        return True

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 25
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks - try in random order for diversity
        kick_success = False
        
        # Randomly choose kick size based on desperation
        if no_improvement_count < 5:
            kick_sizes = [1, 2]
        elif no_improvement_count < 15:
            kick_sizes = [1, 2, 3]
        elif no_improvement_count < 30:
            kick_sizes = [2, 3, 4]
        else:
            kick_sizes = [3, 4, 5]
        
        random.shuffle(kick_sizes)
        
        for k in kick_sizes:
            if try_kick(k):
                kick_success = True
                break
        
        # 3. If still stuck, try swap moves
        if not kick_success:
            # Try a few swap moves
            for _ in range(3):
                if try_swap():
                    kick_success = True
                    break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb more aggressively: remove 2 random, repair
                if len(code) >= 2:
                    indices = random.sample(range(len(code)), 2)
                    removed = [code[i] for i in indices]
                    temp_coverage = coverage.copy()
                    for c in removed:
                        temp_coverage[neighbors_list[c]] -= 1
                    uncovered = np.where(temp_coverage == 0)[0]
                    if len(uncovered) > 0:
                        uncovered_set = set(uncovered.tolist())
                        added = greedy_repair(uncovered_set)
                        added = [a for a in added if a not in removed]
                        new_size = len(code) - len(removed) + len(added)
                        if new_size < len(code):
                            apply_code_change(removed, added)
                            no_improvement_count = 0
                no_improvement_count = 0  # Reset to try again

    result = [idx_to_tuple(c) for c in code]
    return result
