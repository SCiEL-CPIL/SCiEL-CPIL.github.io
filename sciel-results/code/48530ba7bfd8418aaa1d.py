import time
import random
import itertools
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 120s limit

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

    # Precompute neighbor lists as both arrays and sets
    neighbors_list = []
    neighbors_set = []
    neighbors_bitset = []  # Python int bitsets for fast operations
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
        # Build bitset
        bs = 0
        for x in nb:
            bs |= (1 << x)
        neighbors_bitset.append(bs)

    # ---------- Greedy Set-Cover Construction with Multiple Restarts ----------
    
    def greedy_construct(seed_val):
        """Run greedy set-cover construction with heap-based lazy deletion."""
        rng = random.Random(seed_val)
        cov = np.zeros(N, dtype=np.int32)
        unc = np.ones(N, dtype=bool)
        cod = []
        cod_set = set()
        
        # Random start
        start = rng.randint(0, N - 1)
        cod.append(start)
        cod_set.add(start)
        cov[neighbors_list[start]] += 1
        unc[neighbors_list[start]] = False
        
        # Heap-based greedy: (negative_gain, random_tiebreak, candidate)
        heap = []
        in_heap = set()
        
        # Initialize heap with candidates adjacent to uncovered words
        uncovered_indices = np.where(unc)[0]
        candidate_pool = set()
        for w in uncovered_indices:
            candidate_pool.update(neighbors_set[w])
        
        for c in candidate_pool:
            if c not in cod_set:
                gain = int(np.sum(unc[neighbors_list[c]]))
                if gain > 0:
                    heapq.heappush(heap, (-gain, rng.random(), c))
                    in_heap.add(c)
        
        while unc.any() and time.time() < deadline:
            # Pop until we find a valid candidate with current gain
            best_c = -1
            best_count = 0
            
            while heap:
                neg_gain, tiebreak, c = heapq.heappop(heap)
                in_heap.discard(c)
                
                if c in cod_set:
                    continue
                
                # Recompute actual gain
                actual_gain = int(np.sum(unc[neighbors_list[c]]))
                
                if actual_gain <= 0:
                    continue
                
                # If gain changed, re-push with updated gain
                if -neg_gain != actual_gain:
                    heapq.heappush(heap, (-actual_gain, tiebreak, c))
                    in_heap.add(c)
                    continue
                
                best_c = c
                best_count = actual_gain
                break
            
            if best_c == -1:
                # Need to rebuild heap from scratch
                uncovered_indices = np.where(unc)[0]
                candidate_pool = set()
                for w in uncovered_indices:
                    candidate_pool.update(neighbors_set[w])
                
                for c in candidate_pool:
                    if c not in cod_set and c not in in_heap:
                        gain = int(np.sum(unc[neighbors_list[c]]))
                        if gain > 0:
                            heapq.heappush(heap, (-gain, rng.random(), c))
                            in_heap.add(c)
                
                # Try popping again
                while heap:
                    neg_gain, tiebreak, c = heapq.heappop(heap)
                    in_heap.discard(c)
                    
                    if c in cod_set:
                        continue
                    
                    actual_gain = int(np.sum(unc[neighbors_list[c]]))
                    
                    if actual_gain <= 0:
                        continue
                    
                    if -neg_gain != actual_gain:
                        heapq.heappush(heap, (-actual_gain, tiebreak, c))
                        in_heap.add(c)
                        continue
                    
                    best_c = c
                    best_count = actual_gain
                    break
            
            if best_c == -1:
                break
            
            cod.append(best_c)
            cod_set.add(best_c)
            covered = neighbors_list[best_c]
            cov[covered] += 1
            unc[covered] = False
            
        return cod, cod_set, cov

    # Try multiple greedy restarts with different seeds
    best_code = None
    best_code_set = None
    best_coverage = None
    best_size = float('inf')
    
    max_restarts = 12
    for restart in range(max_restarts):
        if time.time() > deadline - 50:  # Need more time for local search
            break
        cod, cod_set, cov = greedy_construct(42 + restart)
        if len(cod) < best_size:
            best_size = len(cod)
            best_code = cod
            best_code_set = cod_set
            best_coverage = cov
        # Early stop if we've found a very good solution
        if best_size <= 170:
            break
    
    code = best_code
    code_set = best_code_set
    coverage = best_coverage
    code_idx_map = {c: i for i, c in enumerate(code)}  # Fast index lookup

    # ---------- Local Search ----------
    
    def is_redundant(c):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        max_iters = len(temp_uncovered) + 5  # Safety bound
        
        for _ in range(max_iters):
            if not temp_uncovered:
                break
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
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
        """
        Apply the change to global state. Uses code_idx_map for O(1) removal.
        """
        nonlocal code, code_set, coverage, code_idx_map
        for c in removed_list:
            if c in code_set:
                idx = code_idx_map[c]
                # Swap with last element for O(1) removal
                last = code[-1]
                code[idx] = last
                code_idx_map[last] = idx
                code.pop()
                code_idx_map.pop(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code_idx_map[c] = len(code)
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False
            
        # Pick k distinct random indices
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            # Apply change
            apply_code_change(removed, added)
            return True
        else:
            # Try to find a better repair with more candidates
            if len(uncovered_set) <= 30:
                # Try to find codewords that cover multiple uncovered words
                candidate_pool = set()
                for w in uncovered_set:
                    candidate_pool.update(neighbors_set[w])
                
                # Sort candidates by coverage count
                scored = []
                for c in candidate_pool:
                    if c in code_set or c in removed:
                        continue
                    gain = len(neighbors_set[c] & uncovered_set)
                    if gain > 0:
                        scored.append((gain, c))
                scored.sort(reverse=True)
                
                # Try top few candidates
                for gain, c in scored[:10]:
                    if gain >= len(uncovered_set):
                        apply_code_change(removed, [c])
                        return True
            
            return False

    def try_swap():
        """
        Try to swap a random codeword with a different one.
        Returns True if size decreased (i.e., perfect removal found).
        """
        if len(code) < 2:
            return False
            
        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal, no need to add
            apply_code_change([removed], [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Only consider candidates adjacent to uncovered words
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])
        
        # Find a candidate to add
        best_c = -1
        best_gain = 0
        for c in candidate_pool:
            if c == removed or c in code_set:
                continue
            gain = len(neighbors_set[c] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if best_gain >= len(uncovered_set):
                    break
        
        if best_c == -1:
            return False
            
        # Apply the swap if it covers all uncovered
        if best_gain >= len(uncovered_set):
            apply_code_change([removed], [best_c])
            return True
        else:
            return False

    def try_2for1():
        """
        Try to replace 2 codewords with 1 that covers their union better.
        Returns True if size decreased.
        """
        if len(code) < 2:
            return False
            
        # Pick 2 random distinct codewords
        idx1 = random.randint(0, len(code) - 1)
        idx2 = random.randint(0, len(code) - 1)
        if idx1 == idx2:
            return False
        
        removed = [code[idx1], code[idx2]]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal of both
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Find a single codeword that covers all uncovered
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])
        
        for c in candidate_pool:
            if c in code_set or c in removed:
                continue
            if neighbors_set[c] >= uncovered_set:
                apply_code_change(removed, [c])
                return True
        
        return False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 30
    random.seed(42)
    
    while time.time() < deadline - 3:  # Safety margin
        # 1. Exhaustive Removal - repeated until no more redundant codewords
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
        
        # 2. Kicks - try multiple attempts per iteration for better mixing
        kick_success = False
        for _ in range(8):  # Try up to 8 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break
            
            # Try various kick strategies in random order
            strategy = random.random()
            if strategy < 0.25:
                # Try 1-removal
                if try_kick(1):
                    kick_success = True
                    break
            elif strategy < 0.45:
                # Try 2-removal
                if try_kick(2):
                    kick_success = True
                    break
            elif strategy < 0.60:
                # Try 3-removal
                if try_kick(3):
                    kick_success = True
                    break
            elif strategy < 0.75:
                # Try swap
                if try_swap():
                    kick_success = True
                    break
            else:
                # Try 2-for-1
                if try_2for1():
                    kick_success = True
                    break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum
                # Try a random kick with larger k
                if len(code) > 5:
                    k = random.randint(4, min(10, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                    elif try_2for1():
                        no_improvement_count = 0
                else:
                    # If we're stuck, try more aggressive perturbations
                    if len(code) > 10:
                        if try_kick(3):
                            no_improvement_count = 0
                        elif try_2for1():
                            no_improvement_count = 0

    result = [idx_to_tuple(c) for c in code]
    return result
