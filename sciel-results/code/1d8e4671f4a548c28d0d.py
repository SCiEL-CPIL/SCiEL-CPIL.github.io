import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 128
    # A is a subset of [0, 128]
    # Target: every t in [128,256] must be expressible as a+b with a<=b, a,b in A
    # Minimize the largest representation multiplicity (unordered pairs, a<=b)
    
    # Precompute the target range
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    NUM_TARGETS = TARGET_END - TARGET_START + 1  # 129 targets
    
    # Precompute contribution vectors: for each x in [0,128], which targets does x contribute to
    # contrib[x] is a vector where contrib[x][t-128] = number of unordered pairs (a,b) with a<=b, a+b=t, and x is one of {a,b}
    # When we add x to A, the new pairs are (a,x) for all a in A with a<=x, plus (x,x) if x+x in range
    # But for incremental updates, we need: adding x adds pairs (a,x) for a in A (all a, since we count unordered)
    # Actually for unordered: pairs involving x are (min(a,x), max(a,x)) for each a in A, a != x, plus (x,x)
    # The contribution of x given current set A is: for each a in A, a != x: one pair at a+x
    # plus (x,x) at 2x
    # So the "delta" when adding x depends on current A. We can't fully precompute.
    # But we CAN precompute for each x, the target index for (a+x) for each possible a.
    
    # Better approach: precompute pair_index[a][b] = index in counts array for a+b, or -1 if out of range
    pair_index = np.full((129, 129), -1, dtype=np.int32)
    for a in range(129):
        for b in range(a, 129):
            s = a + b
            if TARGET_START <= s <= TARGET_END:
                pair_index[a][b] = s - TARGET_START
    
    def build_unordered_counts(A_set):
        """Build the unordered pair count for each target t in [128,256]."""
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:  # b >= a, unordered
                idx = pair_index[a][b]
                if idx >= 0:
                    counts[idx] += 1
        return counts
    
    def evaluate(A_set):
        """Return (max_multiplicity, covered) for the given set A."""
        counts = build_unordered_counts(A_set)
        max_mult = int(np.max(counts))
        covered = bool(np.all(counts > 0))
        return max_mult, covered
    
    def get_unordered_counts(A_set):
        """Return the unordered pair count array for targets [128,256]."""
        return build_unordered_counts(A_set)
    
    # Precompute for each x, the list of pair indices for pairs (a,x) where a in [0,128], a != x
    # And the self-pair index for (x,x)
    pair_indices_for_x = []
    self_pair_idx = []
    for x in range(129):
        indices = []
        for a in range(129):
            if a == x:
                continue
            idx = pair_index[min(a,x)][max(a,x)]
            if idx >= 0:
                indices.append(idx)
        pair_indices_for_x.append(np.array(indices, dtype=np.int32))
        sp = pair_index[x][x]
        self_pair_idx.append(sp)
    
    def update_counts_add(counts, A_set, x):
        """Update counts after adding x to A. Returns new counts (copy)."""
        new_counts = counts.copy()
        # Add pairs (a,x) for a in A_set, a != x
        for a in A_set:
            if a == x:
                continue
            idx = pair_index[min(a,x)][max(a,x)]
            if idx >= 0:
                new_counts[idx] += 1
        # Add (x,x)
        sp = self_pair_idx[x]
        if sp >= 0:
            new_counts[sp] += 1
        return new_counts
    
    def update_counts_remove(counts, A_set, x):
        """Update counts after removing x from A. Returns new counts (copy)."""
        new_counts = counts.copy()
        # Remove pairs (a,x) for a in A_set, a != x
        for a in A_set:
            if a == x:
                continue
            idx = pair_index[min(a,x)][max(a,x)]
            if idx >= 0:
                new_counts[idx] -= 1
        # Remove (x,x)
        sp = self_pair_idx[x]
        if sp >= 0:
            new_counts[sp] -= 1
        return new_counts

    def compute_cost(counts):
        """Compute a smooth cost function for the SA search.
        We want to minimize max multiplicity, but also flatten the distribution.
        Use soft-max via log-sum-exp for smoother gradient, plus coverage penalty.
        """
        max_mult = int(np.max(counts))
        uncovered = int(np.sum(counts == 0))
        if uncovered > 0:
            return max_mult + 100 * uncovered
        # Soft penalty for high multiplicities
        tau = 3.0
        soft_max_penalty = float(np.sum(np.exp(counts.astype(np.float64) / tau)) - NUM_TARGETS)
        return max_mult + 0.1 * soft_max_penalty

    def local_search(A_set, time_budget, seed=None):
        """Local search with incremental evaluation and swap moves. Returns (best_A, best_score)."""
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_set)
        current_counts = get_unordered_counts(current_A)
        
        # Best tracking
        best_A = set(current_A)
        best_counts = current_counts.copy()
        best_max = int(np.max(best_counts))
        best_covered = bool(np.all(best_counts > 0))
        best_score = best_max if best_covered else float('inf')
        
        T = 30.0
        T_min = 0.5
        cooling = 0.999
        
        iter_count = 0
        while time.time() < time_budget and T > T_min:
            iter_count += 1
            current_cost = compute_cost(current_counts)
            
            move_type = random.random()
            accepted = False
            new_A = None
            new_counts = None
            new_cost = None
            
            if move_type < 0.35:
                # Add move
                x = random.randint(0, 128)
                if x not in current_A:
                    new_A = current_A | {x}
                    new_counts = update_counts_add(current_counts, current_A, x)
                    new_cost = compute_cost(new_counts)
            elif move_type < 0.65:
                # Remove move
                if len(current_A) > 15:
                    x = random.choice(list(current_A))
                    new_A = current_A - {x}
                    new_counts = update_counts_remove(current_counts, current_A, x)
                    new_cost = compute_cost(new_counts)
            else:
                # Swap move
                if len(current_A) > 15:
                    x = random.choice(list(current_A))
                    y = random.randint(0, 128)
                    if y not in current_A:
                        new_A = (current_A - {x}) | {y}
                        # Efficient swap update:
                        # 1. Remove x
                        temp_counts = update_counts_remove(current_counts, current_A, x)
                        # 2. Add y to the set A \ {x}
                        new_counts = update_counts_add(temp_counts, current_A - {x}, y)
                        new_cost = compute_cost(new_counts)
            
            if new_A is not None:
                delta = new_cost - current_cost
                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                    current_A = new_A
                    current_counts = new_counts
                    accepted = True
                    
                    # Update best
                    new_max = int(np.max(new_counts))
                    new_covered = bool(np.all(new_counts > 0))
                    new_score = new_max if new_covered else float('inf')
                    if new_score < best_score:
                        best_score = new_score
                        best_A = set(new_A)
                        best_counts = new_counts.copy()
            
            T *= cooling
        
        # Final Polish: Try to reduce max multiplicity further using 1-out-1-in swaps
        # This is a steepest descent step on the max multiplicity
        if best_score < float('inf'):
            best_A_list = sorted(best_A)
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                best_max = int(np.max(best_counts))
                # Try removing each element
                for x in best_A_list[:]:
                    if len(best_A) <= 10:
                        break
                    # Try removing x
                    test_counts = update_counts_remove(best_counts, best_A, x)
                    test_max = int(np.max(test_counts))
                    test_covered = bool(np.all(test_counts > 0))
                    
                    if test_covered and test_max < best_max:
                        # Direct improvement
                        best_A = best_A - {x}
                        best_counts = test_counts
                        best_max = test_max
                        best_score = best_max
                        improved = True
                        best_A_list = sorted(best_A)
                        break
                    elif not test_covered:
                        # Coverage broken. Try to fix by adding one element y
                        # Find uncovered targets
                        uncovered = np.where(test_counts == 0)[0]
                        if len(uncovered) == 0: # Should not happen if not test_covered
                            continue
                        
                        # Try to add a y that covers the most uncovered targets
                        # and doesn't increase max too much
                        best_y = None
                        best_y_cost = float('inf')
                        # Sample some y's to keep it fast
                        for _ in range(5):
                            y = random.randint(0, 128)
                            if y in best_A:
                                continue
                            # Add y to test_counts
                            # We need to add y to the set (best_A - {x})
                            candidate_counts = update_counts_add(test_counts, best_A - {x}, y)
                            cand_max = int(np.max(candidate_counts))
                            cand_covered = bool(np.all(candidate_counts > 0))
                            if cand_covered:
                                # Cost: prefer lower max
                                cost = cand_max
                                if cost < best_y_cost:
                                    best_y_cost = cost
                                    best_y = y
                                    best_y_counts = candidate_counts
                        if best_y is not None:
                            # Perform swap
                            best_A = (best_A - {x}) | {best_y}
                            best_counts = best_y_counts
                            best_max = int(np.max(best_counts))
                            best_score = best_max
                            improved = True
                            best_A_list = sorted(best_A)
                            break
        
        return best_A, best_score

    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')
    
    def make_initials():
        initials = []
        # Full set
        initials.append(set(range(0, 129)))
        # Half: [64, 128]
        initials.append(set(range(64, 129)))
        # Every other in [0,128]: even numbers
        initials.append(set(range(0, 129, 2)))
        # Odd numbers
        initials.append(set(range(1, 129, 2)))
        # Random sets of various sizes
        random.seed(123)
        for size in [40, 50, 60, 70, 80, 90, 100]:
            s = set(random.sample(range(0, 129), size))
            initials.append(s)
        # Structured: multiples of k
        for k in [2, 3, 4]:
            s = set(range(0, 129, k))
            initials.append(s)
        # Complementary pairs
        s = set()
        for i in range(0, 129, 2):
            s.add(i)
            s.add(128 - i)
        initials.append(s)
        # New: balanced density patterns
        # Pattern: include x if (x mod period) < duty
        for period in [4, 5, 6, 7, 8]:
            for duty in range(1, period):
                s = set(x for x in range(129) if x % period < duty)
                if len(s) >= 20:
                    initials.append(s)
        # Pattern: include x if floor(x / block) % 2 == 0
        for block in [4, 5, 6, 8]:
            s = set(x for x in range(129) if (x // block) % 2 == 0)
            if len(s) >= 20:
                initials.append(s)
        # Pattern: symmetric around 64
        s = set(range(32, 97))  # [32, 96]
        initials.append(s)
        s = set(range(48, 81))  # [48, 80]
        initials.append(s)
        return initials

    def fix_coverage(A_set):
        """Greedily add elements to ensure all targets are covered."""
        A = set(A_set)
        for t in TARGETS:
            # Check if t is covered
            found = False
            for a in A:
                b = t - a
                if b >= a and b in A:
                    found = True
                    break
            if not found:
                mid = t // 2
                if 0 <= mid <= 128:
                    A.add(mid)
                if t % 2 == 1 and mid + 1 <= 128:
                    A.add(mid + 1)
        return A

    initials = make_initials()
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)
    
    # Phase 1: Quick evaluation of all initials
    scored_initials = []
    for init in initials:
        if time.time() > start_time + 10:
            break
        s, c = evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))
    
    # Sort by cost (best first)
    scored_initials.sort(key=lambda x: x[0])
    
    # Phase 2: Local search from best initials
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        time_alloc = max(2, min(remaining_time / remaining_starts, 25))
        
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    # Phase 3: Additional random restarts if time remains
    if time.time() < deadline - 10 and best_A is not None:
        num_extra = 5
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(35, 85)
            init = set(random.sample(range(0, 129), size))
            init = fix_coverage(init)
            
            time_alloc = max(2, (deadline - 5 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
    
    # Fallback: ensure we have a valid solution
    if best_A is None or best_score == float('inf'):
        best_A = set(range(0, 129))
        best_score, best_covered = evaluate(best_A)
        if not best_covered:
            best_A = set(range(64, 129))
    
    # Final verification
    final_score, final_covered = evaluate(best_A)
    if not final_covered:
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)
    
    # Final exhaustive polish: try every removal and swap to reduce peak multiplicity
    if final_covered and time.time() < deadline - 2:
        best_A_set = set(best_A)
        best_counts = get_unordered_counts(best_A_set)
        best_max = int(np.max(best_counts))
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try removing each element
            for x in sorted(best_A_set):
                if len(best_A_set) <= 5:
                    break
                test_counts = update_counts_remove(best_counts, best_A_set, x)
                if np.all(test_counts > 0):
                    test_max = int(np.max(test_counts))
                    if test_max < best_max:
                        best_A_set = best_A_set - {x}
                        best_counts = test_counts
                        best_max = test_max
                        improved = True
                        break
            if not improved:
                # Try swaps: remove x, add y
                for x in sorted(best_A_set):
                    if len(best_A_set) <= 5:
                        break
                    test_counts = update_counts_remove(best_counts, best_A_set, x)
                    # Try adding each y not in A\{x}
                    for y in range(129):
                        if y in best_A_set and y != x:
                            continue
                        cand_counts = update_counts_add(test_counts, best_A_set - {x}, y)
                        if np.all(cand_counts > 0):
                            cand_max = int(np.max(cand_counts))
                            if cand_max < best_max:
                                best_A_set = (best_A_set - {x}) | {y}
                                best_counts = cand_counts
                                best_max = cand_max
                                improved = True
                                break
                    if improved:
                        break
    
    final_A = sorted(best_A_set)
    return final_A
