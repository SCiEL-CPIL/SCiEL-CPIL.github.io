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

    # ---------- Vectorized Greedy Set-Cover Construction ----------
    
    def greedy_construct(start_word=0, max_time=None):
        """
        Vectorized greedy set-cover construction.
        Returns (code_list, coverage_array, uncovered_mask).
        """
        if max_time is None:
            max_time = deadline
        
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        # Greedy initialization
        code.append(start_word)
        code_set.add(start_word)
        coverage[neighbors_list[start_word]] += 1
        uncovered_mask[neighbors_list[start_word]] = False
        
        # Precompute neighbor matrix for vectorized operations
        # neighbors_matrix[i] = list of neighbors of word i
        # We'll use a sparse representation via the neighbors_list
        
        while uncovered_mask.any() and time.time() < max_time:
            # Vectorized: compute coverage gain for all candidates at once
            # For each candidate c, gain = number of uncovered words in neighbors_list[c]
            # We can compute this by summing uncovered_mask over each row of the neighbor matrix
            
            # Build a boolean matrix where M[c, w] = True if w is a neighbor of c
            # But this is 2187x2187 which is manageable
            # Actually, let's just vectorize the inner loop using numpy
            
            # Create array of all candidate indices not in code
            candidates = [c for c in range(N) if c not in code_set]
            if not candidates:
                break
            
            # For each candidate, count uncovered neighbors
            # This is still O(N) but we can speed it up with numpy
            gains = np.array([np.sum(uncovered_mask[neighbors_list[c]]) for c in candidates], dtype=np.int32)
            
            best_idx = np.argmax(gains)
            best_count = gains[best_idx]
            best_c = candidates[best_idx]
            
            if best_count == 0:
                break
                
            code.append(best_c)
            code_set.add(best_c)
            coverage[neighbors_list[best_c]] += 1
            uncovered_mask[neighbors_list[best_c]] = False
        
        return code, coverage, uncovered_mask

    # ---------- Local Search ----------
    
    def is_redundant(c, cov):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(cov[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set, current_code_set, cov_base):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        temp_cov = cov_base.copy()
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in current_code_set or c in added:
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

    def local_search(code_list, coverage, code_set, time_limit):
        """
        Run local search on the given code.
        Returns improved (code_list, coverage, code_set).
        """
        nonlocal deadline
        
        def apply_change(removed_list, added_list):
            for c in removed_list:
                if c in code_set:
                    code_list.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
            for c in added_list:
                if c not in code_set:
                    code_list.append(c)
                    code_set.add(c)
                    coverage[neighbors_list[c]] += 1
        
        no_improvement_count = 0
        max_no_improvement = 30
        
        while time.time() < time_limit:
            # 1. Exhaustive Removal
            changed = True
            while changed:
                if time.time() > time_limit:
                    break
                changed = False
                for c in list(code_list):
                    if is_redundant(c, coverage):
                        apply_change([c], [])
                        changed = True
                        if time.time() > time_limit:
                            break
            
            # 2. Kicks
            kick_success = False
            
            # Try 1-removal
            if len(code_list) >= 1:
                idx = random.randint(0, len(code_list) - 1)
                removed = code_list[idx]
                temp_cov = coverage.copy()
                temp_cov[neighbors_list[removed]] -= 1
                uncovered = np.where(temp_cov == 0)[0]
                
                if len(uncovered) == 0:
                    apply_change([removed], [])
                    kick_success = True
                else:
                    uncovered_set = set(uncovered.tolist())
                    added = greedy_repair(uncovered_set, code_set, temp_cov)
                    if len(code_list) - 1 + len(added) < len(code_list):
                        apply_change([removed], added)
                        kick_success = True
            
            # Try 2-removal
            if not kick_success and len(code_list) >= 2:
                indices = random.sample(range(len(code_list)), 2)
                removed = [code_list[i] for i in indices]
                temp_cov = coverage.copy()
                for c in removed:
                    temp_cov[neighbors_list[c]] -= 1
                uncovered = np.where(temp_cov == 0)[0]
                
                if len(uncovered) == 0:
                    apply_change(removed, [])
                    kick_success = True
                else:
                    uncovered_set = set(uncovered.tolist())
                    added = greedy_repair(uncovered_set, code_set, temp_cov)
                    if len(code_list) - 2 + len(added) < len(code_list):
                        apply_change(removed, added)
                        kick_success = True
            
            # Try 3-removal
            if not kick_success and len(code_list) >= 3:
                indices = random.sample(range(len(code_list)), 3)
                removed = [code_list[i] for i in indices]
                temp_cov = coverage.copy()
                for c in removed:
                    temp_cov[neighbors_list[c]] -= 1
                uncovered = np.where(temp_cov == 0)[0]
                
                if len(uncovered) == 0:
                    apply_change(removed, [])
                    kick_success = True
                else:
                    uncovered_set = set(uncovered.tolist())
                    added = greedy_repair(uncovered_set, code_set, temp_cov)
                    if len(code_list) - 3 + len(added) < len(code_list):
                        apply_change(removed, added)
                        kick_success = True
            
            if kick_success:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    # Try larger kick
                    if len(code_list) > 5:
                        k = random.randint(4, min(8, len(code_list)))
                        indices = random.sample(range(len(code_list)), k)
                        removed = [code_list[i] for i in indices]
                        temp_cov = coverage.copy()
                        for c in removed:
                            temp_cov[neighbors_list[c]] -= 1
                        uncovered = np.where(temp_cov == 0)[0]
                        
                        if len(uncovered) == 0:
                            apply_change(removed, [])
                            no_improvement_count = 0
                        else:
                            uncovered_set = set(uncovered.tolist())
                            added = greedy_repair(uncovered_set, code_set, temp_cov)
                            if len(code_list) - k + len(added) < len(code_list):
                                apply_change(removed, added)
                                no_improvement_count = 0
        
        return code_list, coverage, code_set

    # ---------- Multi-restart Greedy + Local Search ----------
    random.seed(42)
    
    best_code = None
    best_coverage = None
    best_code_set = None
    best_size = float('inf')
    
    # Time budget allocation: 40% for greedy restarts, 60% for local search
    greedy_budget = (deadline - time.time()) * 0.4
    greedy_deadline = time.time() + greedy_budget
    
    # Multiple greedy restarts with different starting words
    num_restarts = 0
    while time.time() < greedy_deadline:
        start_word = random.randint(0, N - 1)
        code, coverage, uncovered_mask = greedy_construct(start_word, max_time=greedy_deadline)
        
        if len(uncovered_mask) > 0:
            # Greedy didn't complete, skip this restart
            continue
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            best_coverage = coverage.copy()
            best_code_set = set(code)
            
        num_restarts += 1
        if num_restarts >= 5:
            break
    
    # If no complete greedy was found, use the best partial one
    if best_code is None:
        # Fallback: use the first greedy result
        code, coverage, uncovered_mask = greedy_construct(0, max_time=deadline)
        best_code = code
        best_coverage = coverage
        best_code_set = set(code)
    
    # Local search on the best code
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        best_code, best_coverage, best_code_set = local_search(
            best_code, best_coverage, best_code_set, 
            deadline - 0.5  # Leave 0.5s margin
        )
    
    result = [idx_to_tuple(c) for c in best_code]
    return result
