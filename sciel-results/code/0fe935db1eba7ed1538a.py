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
    
    # Precompute for each target t and each element x in [0,128]:
    # the number of partners y in [0,128] such that x+y=t and y>=x (unordered)
    # Actually, let's use a different approach: maintain the unordered count array directly.
    
    def build_unordered_counts(A_set):
        """Build the unordered pair count for each target t in [128,256]."""
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:  # b >= a, unordered
                s = a + b
                if TARGET_START <= s <= TARGET_END:
                    counts[s - TARGET_START] += 1
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
    
    def update_counts_add(counts, A_set, x):
        """Update counts after adding x to A. Returns new counts (copy)."""
        new_counts = counts.copy()
        for a in A_set:
            s = a + x
            if TARGET_START <= s <= TARGET_END:
                new_counts[s - TARGET_START] += 1
        # Also (x, x) if x+x is in range
        s = 2 * x
        if TARGET_START <= s <= TARGET_END:
            new_counts[s - TARGET_START] += 1
        return new_counts
    
    def update_counts_remove(counts, A_set, x):
        """Update counts after removing x from A. Returns new counts (copy)."""
        new_counts = counts.copy()
        # Remove pairs (a, x) where a in A_set, a != x, and a < x (to avoid double count)
        # Actually, we need to remove all pairs involving x.
        # Pairs are (min(a,b), max(a,b)) with a<=b.
        # Pairs involving x: (a, x) where a < x, and (x, x).
        for a in A_set:
            if a == x:
                continue
            s = a + x
            if TARGET_START <= s <= TARGET_END:
                new_counts[s - TARGET_START] -= 1
        # Remove (x, x)
        s = 2 * x
        if TARGET_START <= s <= TARGET_END:
            new_counts[s - TARGET_START] -= 1
        return new_counts
    
    # Strategy: Multi-start local search with diverse initial solutions
    # 1. Try several structured initializations
    # 2. Local search with incremental evaluation and targeted moves
    # 3. Keep the best valid solution found
    
    # Candidate initial solutions
    def make_initials():
        initials = []
        # Full set
        initials.append(set(range(0, 129)))
        # Half: [64, 128]
        initials.append(set(range(64, 129)))
        # Every other in [0,128]: even numbers
        initials.append(set(range(0, 129, 2)))
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
    
    def local_search(A_set, time_budget, seed=None):
        """Local search with incremental evaluation. Returns (best_A, best_score)."""
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_set)
        current_counts = get_unordered_counts(current_A)
        current_max = int(np.max(current_counts))
        current_covered = bool(np.all(current_counts > 0))
        current_score = current_max if current_covered else float('inf')
        
        best_A = set(current_A)
        best_score = current_max if current_covered else float('inf')
        
        T = 20.0
        T_min = 0.3
        cooling = 0.9998
        
        iter_count = 0
        while time.time() < time_budget and T > T_min:
            iter_count += 1
            
            # Generate a move
            # Strategy: 70% targeted (remove from high-multiplicity targets or add to low-multiplicity), 30% random
            if random.random() < 0.7:
                # Targeted move
                if current_covered and len(current_A) > 10:
                    # Try to reduce the max multiplicity
                    # Find target(s) with max multiplicity
                    max_val = int(np.max(current_counts))
                    # Find elements that contribute to high-multiplicity targets
                    high_targets = np.where(current_counts == max_val)[0]
                    # Pick a random high target
                    ht = random.choice(high_targets)
                    t_val = TARGET_START + ht
                    # Find pairs that sum to t_val
                    candidates = []
                    for a in current_A:
                        b = t_val - a
                        if b >= a and b in current_A:
                            candidates.append((a, b))
                    if candidates:
                        # Remove one of the elements from a random pair
                        a, b = random.choice(candidates)
                        # Prefer removing the one that appears in fewer high-multiplicity targets
                        # Simple: just remove a random one
                        if random.random() < 0.5:
                            x = a
                        else:
                            x = b
                        if x in current_A:
                            new_A = current_A - {x}
                            new_counts = update_counts_remove(current_counts, current_A, x)
                            new_max = int(np.max(new_counts))
                            new_covered = bool(np.all(new_counts > 0))
                            new_score = new_max if new_covered else float('inf')
                            
                            if new_score < best_score:
                                best_A = set(new_A)
                                best_score = new_score
                            
                            # Acceptance criterion
                            if new_covered:
                                new_cost = new_max
                            else:
                                new_cost = new_max + 1000
                            
                            if current_covered:
                                current_cost = current_max
                            else:
                                current_cost = current_max + 1000
                            
                            delta = new_cost - current_cost
                            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                                current_A = new_A
                                current_counts = new_counts
                                current_max = new_max
                                current_covered = new_covered
                                current_score = new_score
                        continue
                elif not current_covered:
                    # Find uncovered targets and try to cover them
                    uncovered = np.where(current_counts == 0)[0]
                    if len(uncovered) > 0:
                        ut = random.choice(uncovered)
                        t_val = TARGET_START + ut
                        # Try adding an element that helps cover this target
                        # Find x such that there exists y in A with x+y=t_val
                        # Or add x such that 2*x=t_val
                        mid = t_val // 2
                        candidates = []
                        if 0 <= mid <= 128:
                            candidates.append(mid)
                        if t_val % 2 == 1 and mid + 1 <= 128:
                            candidates.append(mid + 1)
                        # Also try random elements
                        for _ in range(3):
                            x = random.randint(0, 128)
                            if x not in current_A:
                                candidates.append(x)
                        if candidates:
                            x = random.choice(candidates)
                            if x not in current_A:
                                new_A = current_A | {x}
                                new_counts = update_counts_add(current_counts, current_A, x)
                                new_max = int(np.max(new_counts))
                                new_covered = bool(np.all(new_counts > 0))
                                new_score = new_max if new_covered else float('inf')
                                
                                if new_score < best_score:
                                    best_A = set(new_A)
                                    best_score = new_score
                                
                                new_cost = new_max if new_covered else new_max + 1000
                                current_cost = current_max if current_covered else current_max + 1000
                                delta = new_cost - current_cost
                                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                                    current_A = new_A
                                    current_counts = new_counts
                                    current_max = new_max
                                    current_covered = new_covered
                                    current_score = new_score
                        continue
            
            # Random move
            if len(current_A) > 5 and random.random() < 0.5:
                # Remove
                x = random.choice(list(current_A))
                new_A = current_A - {x}
                new_counts = update_counts_remove(current_counts, current_A, x)
            else:
                # Add
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A = current_A | {x}
                new_counts = update_counts_add(current_counts, current_A, x)
            
            new_max = int(np.max(new_counts))
            new_covered = bool(np.all(new_counts > 0))
            new_score = new_max if new_covered else float('inf')
            
            if new_score < best_score:
                best_A = set(new_A)
                best_score = new_score
            
            new_cost = new_max if new_covered else new_max + 1000
            current_cost = current_max if current_covered else current_max + 1000
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                current_A = new_A
                current_counts = new_counts
                current_max = new_max
                current_covered = new_covered
                current_score = new_score
            
            T *= cooling
        
        # Final polish: try to remove elements that don't affect coverage
        if best_covered if 'best_covered' in dir() else True:
            pass
        # Try greedy removal on best solution
        if best_score < float('inf'):
            best_A_list = sorted(best_A)
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                for x in best_A_list[:]:
                    if len(best_A) <= 5:
                        break
                    test_A = best_A - {x}
                    test_counts = get_unordered_counts(test_A)
                    test_max = int(np.max(test_counts))
                    test_covered = bool(np.all(test_counts > 0))
                    if test_covered and test_max < best_score:
                        best_A = test_A
                        best_score = test_max
                        improved = True
                        best_A_list = sorted(best_A)
                        break
        
        return best_A, best_score
    
    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')
    
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
        num_extra = 3
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
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
    
    return sorted(best_A)
