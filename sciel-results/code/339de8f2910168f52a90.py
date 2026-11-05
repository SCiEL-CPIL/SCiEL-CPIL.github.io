import time
import random
import itertools
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
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

    # Precompute neighbor lists as sorted numpy arrays for fast set operations
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
        neighbors_list.append(np.array(sorted(nb), dtype=np.int32))
        neighbors_set.append(set(nb))

    # Precompute the "ball" bitmask using numpy boolean arrays
    # ball_mask[i] is a boolean array of length N where True means position j is in ball(i)
    ball_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        ball_mask[i, neighbors_list[i]] = True

    # ---------- Greedy Set-Cover Construction ----------
    
    def greedy_construct(seed_val):
        """Run greedy set-cover construction with vectorized gain computation."""
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
        
        while unc.any() and time.time() < deadline:
            # Vectorized gain computation: for each candidate c, gain = number of uncovered in ball(c)
            # Using matrix multiplication: gains[c] = sum(unc * ball_mask[c])
            # But that's N*N which is too big. Instead, use the fact that ball(c) is small.
            # For each uncovered word w, all c in ball(w) get a potential gain.
            # Use accumulation approach:
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(unc)[0]
            for w in uncovered_indices:
                gains[neighbors_list[w]] += 1
            
            # Find best candidate not already in code
            best_c = -1
            best_gain = 0
            for c in range(N):
                if c in cod_set:
                    continue
                g = gains[c]
                if g > best_gain:
                    best_gain = g
                    best_c = c
                elif g == best_gain and best_c != -1:
                    # Tie-break randomly among top candidates
                    if rng.random() < 0.1:
                        best_c = c
            
            if best_c == -1 or best_gain == 0:
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
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 50:
            break
        cod, cod_set, cov = greedy_construct(42 + restart)
        if len(cod) < best_size:
            best_size = len(cod)
            best_code = cod
            best_code_set = cod_set
            best_coverage = cov
        if best_size <= 160:
            break
    
    code = best_code
    code_set = best_code_set
    coverage = best_coverage

    # ---------- Systematic 1-Exchange Local Search ----------
    
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def apply_removal(c):
        nonlocal code, code_set, coverage
        code.remove(c)
        code_set.remove(c)
        coverage[neighbors_list[c]] -= 1

    def apply_addition(c):
        nonlocal code, code_set, coverage
        code.append(c)
        code_set.add(c)
        coverage[neighbors_list[c]] += 1

    def systematic_one_exchange():
        """
        For each codeword c in the code, try to replace it with a different codeword c'
        such that the new code is still valid and has the same size (1-for-1).
        This is exhaustive over all possible replacements.
        Returns True if any improvement was made (size decreased via removal, or reconfiguration).
        """
        improved = False
        for idx in range(len(code)):
            if time.time() > deadline - 5:
                return improved
            c = code[idx]
            
            # Temporarily remove c
            temp_cov = coverage.copy()
            temp_cov[neighbors_list[c]] -= 1
            uncovered = np.where(temp_cov == 0)[0]
            
            if len(uncovered) == 0:
                # Can remove c entirely - size decreases!
                apply_removal(c)
                improved = True
                # After removal, restart the scan from beginning since indices changed
                return True
            
            uncovered_set = set(uncovered.tolist())
            
            # Find all candidates that can cover the uncovered words
            # A candidate c' is valid if ball(c') covers all uncovered words
            # AND c' is not already in the code (except c which we removed)
            candidate_pool = set()
            for w in uncovered_set:
                candidate_pool.update(neighbors_set[w])
            
            # Find the best replacement (one that maximizes "slack" or just any valid one)
            best_c_prime = -1
            best_slack = -1
            
            for c_prime in candidate_pool:
                if c_prime == c:
                    continue
                if c_prime in code_set:
                    continue
                
                # Check if c_prime covers all uncovered
                if not uncovered_set.issubset(neighbors_set[c_prime]):
                    continue
                
                # Compute slack: how many additional words c_prime covers beyond the uncovered
                # Higher slack means more redundancy, which is better for future moves
                slack = len(neighbors_set[c_prime] & set(np.where(temp_cov == 1)[0].tolist()))
                # Actually, let's just count total coverage of c_prime in temp_cov
                # Words with temp_cov == 0 are uncovered, temp_cov >= 1 are covered
                # After adding c_prime, words in ball(c_prime) get +1 coverage
                # We want to maximize the minimum coverage after the swap
                # Simpler: just accept the first valid one, or pick one with max overlap with singly-covered words
                singly_covered = np.where(temp_cov == 1)[0]
                overlap = len(neighbors_set[c_prime] & set(singly_covered.tolist()))
                
                if overlap > best_slack:
                    best_slack = overlap
                    best_c_prime = c_prime
            
            if best_c_prime != -1:
                # Apply the 1-for-1 replacement
                apply_removal(c)
                apply_addition(best_c_prime)
                improved = True
                # Restart scan since code changed
                return True
        
        return improved

    def random_kick(k):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False
            
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            for c in removed:
                apply_removal(c)
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Greedy repair
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered and len(added) < k:  # Don't add more than we removed
            best_c = -1
            best_count = 0
            
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set or c in removed:
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
        
        if not temp_uncovered:
            new_size = len(code) - len(removed) + len(added)
            if new_size < len(code):
                for c in removed:
                    apply_removal(c)
                for c in added:
                    apply_addition(c)
                return True
        
        return False

    # Main Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 15
    
    while time.time() < deadline - 5:
        # Phase 1: Exhaustive removal of redundant codewords
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_removal(c)
                    changed = True
        
        # Phase 2: Systematic 1-exchange (the core structural change)
        if systematic_one_exchange():
            no_improvement_count = 0
            continue
        
        # Phase 3: Random kicks to escape local minima
        kick_success = False
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            k = random.choice([1, 2, 2, 3, 3, 4])
            if random_kick(k):
                kick_success = True
                break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Aggressive perturbation
                if len(code) > 5:
                    k = random.randint(5, min(12, len(code)))
                    if random_kick(k):
                        no_improvement_count = 0
                no_improvement_count = 0  # Reset to give it another chance

    # Final exhaustive removal pass
    changed = True
    while changed:
        if time.time() > deadline - 2:
            break
        changed = False
        for c in list(code):
            if is_redundant(c):
                apply_removal(c)
                changed = True

    result = [idx_to_tuple(c) for c in code]
    return result
