"""Linear triple systems on 16 points with small independence number."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    N = 16
    FULL_MASK = (1 << N) - 1
    ALL_TRIPLES = list(itertools.combinations(range(N), 3))
    num_triples = len(ALL_TRIPLES)
    
    # Precompute masks
    triple_masks = [0] * num_triples
    for i, t in enumerate(ALL_TRIPLES):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])

    # Precompute conflicts
    # conflict[i] is a bitmask of triples j that conflict with i
    # Using bitmasks for fast intersection checks if needed, 
    # but for small sets, list of lists is fine.
    # Let's use a set of pairs for quick lookup.
    pair_to_triples = {}
    for i, t in enumerate(ALL_TRIPLES):
        for k in range(3):
            for l in range(k+1, 3):
                p = (min(t[k], t[l]), max(t[k], t[l]))
                if p not in pair_to_triples:
                    pair_to_triples[p] = []
                pair_to_triples[p].append(i)
                
    # For B&B, we need to check compatibility quickly.
    # We will maintain a set of used pairs.
    
    def independence_number(triples):
        """Exact independence number calculation."""
        if not triples:
            return N
        
        tmasks = []
        for t in triples:
            tmasks.append((1 << t[0]) | (1 << t[1]) | (1 << t[2]))
        
        bad = bytearray(1 << N)
        for m in tmasks:
            free = FULL_MASK ^ m
            sub = free
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & free
                
        best = 0
        for S in range(1 << N):
            if not bad[S]:
                pc = S.bit_count()
                if pc > best:
                    best = pc
        return best

    def is_linear(triples):
        for i in range(len(triples)):
            ti = set(triples[i])
            for j in range(i + 1, len(triples)):
                tj = set(triples[j])
                if len(ti & tj) > 1:
                    return False
        return True

    def b_and_b_search(K, max_time=30.0):
        """
        Try to find a linear triple system with independence number <= K.
        K is the target alpha. We need to cover all subsets of size K+1.
        """
        target_size = K + 1
        # Generate all subsets of size target_size
        all_subsets = list(itertools.combinations(range(N), target_size))
        num_subsets = len(all_subsets)
        
        # Map subset to index
        subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
        
        # For each triple, precompute which subsets it covers
        triple_covers = [[] for _ in range(num_triples)]
        for i, t in enumerate(ALL_TRIPLES):
            t_set = set(t)
            for s in all_subsets:
                if t_set.issubset(s):
                    triple_covers[i].append(subset_to_idx[s])
                    
        # State:
        # covered: list of bools, length num_subsets
        # used_pairs: set of pairs
        # current_triples: list of triple indices
        
        # We use a recursive search with backtracking.
        # To avoid deep recursion issues, we use an explicit stack or limit depth.
        # But Python recursion limit is usually 1000, which is fine.
        
        best_solution = None
        
        def search(covered_count, covered_arr, used_pairs, current_triples, depth):
            nonlocal best_solution
            if best_solution is not None:
                return True # Found solution
            
            if time.time() > deadline - 5.0:
                return False # Timeout
            
            if covered_count == num_subsets:
                best_solution = current_triples[:]
                return True
            
            # Find an uncovered subset
            # Heuristic: pick the one with fewest available covering triples
            # For simplicity, pick the first uncovered one.
            # Optimization: pick the one that is hardest to cover.
            # Let's just pick the first one for now.
            uncovered_idx = -1
            for i in range(num_subsets):
                if not covered_arr[i]:
                    uncovered_idx = i
                    break
            
            if uncovered_idx == -1:
                return False # Should not happen if covered_count < num_subsets
            
            s = all_subsets[uncovered_idx]
            s_set = set(s)
            
            # Find all triples that are subsets of s
            candidates = []
            for i, t in enumerate(ALL_TRIPLES):
                if set(t).issubset(s_set):
                    # Check if t is compatible with used_pairs
                    compatible = True
                    for k in range(3):
                        for l in range(k+1, 3):
                            p = (min(t[k], t[l]), max(t[k], t[l]))
                            if p in used_pairs:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if compatible:
                        candidates.append(i)
            
            if not candidates:
                return False # Dead end
            
            # Order candidates: those that cover more uncovered subsets first
            # Pre-computed covers list helps here
            candidates.sort(key=lambda i: -len([c for c in triple_covers[i] if not covered_arr[c]]))
            
            for i in candidates:
                t = ALL_TRIPLES[i]
                new_used_pairs = used_pairs.copy()
                for k in range(3):
                    for l in range(k+1, 3):
                        new_used_pairs.add((min(t[k], t[l]), max(t[k], t[l])))
                        
                new_covered_arr = covered_arr.copy()
                new_covered_count = covered_count
                for c_idx in triple_covers[i]:
                    if not new_covered_arr[c_idx]:
                        new_covered_arr[c_idx] = True
                        new_covered_count += 1
                        
                new_triples = current_triples + [i]
                
                if search(new_covered_count, new_covered_arr, new_used_pairs, new_triples, depth + 1):
                    return True
            
            return False

        # Initialize
        initial_covered = [False] * num_subsets
        initial_pairs = set()
        initial_triples = []
        
        # Try to find a solution
        # We might need to try different starting points or randomization?
        # The B&B is deterministic based on the order.
        # If it fails, it means no solution exists with this specific branching strategy?
        # No, B&B explores all possibilities. If it returns False, no solution exists 
        # within the time limit or the search space is exhausted (unlikely for small K).
        # But for K=4, it might be very hard.
        
        if search(0, initial_covered, initial_pairs, initial_triples, 0):
            if best_solution:
                return [ALL_TRIPLES[i] for i in best_solution]
                
        return None

    # Try B&B for K=5, then K=6
    # If B&B finds a solution with alpha <= 5, it's a huge improvement.
    # If not, we fall back to local search.
    
    best_triples = None
    best_score = N + 1
    
    # Attempt B&B
    if time.time() < deadline - 20:
        res = b_and_b_search(5, max_time=20.0)
        if res:
            best_triples = res
            best_score = independence_number(res)
    
    if best_triples is None:
        # Fallback to previous logic
        # ... (copy previous local search code) ...
        
        def random_greedy_packing(seed_offset=0):
            rng = random.Random(42 + seed_offset)
            used_pairs = set()
            triples = []
            candidates = list(range(num_triples))
            rng.shuffle(candidates)
            for idx in candidates:
                t = ALL_TRIPLES[idx]
                pairs = [
                    frozenset((t[0], t[1])),
                    frozenset((t[0], t[2])),
                    frozenset((t[1], t[2]))
                ]
                can_add = True
                for p in pairs:
                    if p in used_pairs:
                        can_add = False
                        break
                if can_add:
                    for p in pairs:
                        used_pairs.add(p)
                    triples.append(t)
            return triples

        def local_search_sa(triples, max_time=60):
            rng = random.Random(1000)
            current_triples = triples[:]
            current_score = independence_number(current_triples)
            best_triples = current_triples[:]
            best_score = current_score
            timeout = time.time() + max_time
            T0 = 5.0
            Tend = 0.01
            steps = 0
            while time.time() < timeout:
                steps += 1
                T = max(Tend, T0 * (0.999 ** steps))
                move_type = rng.random()
                new_triples = current_triples[:]
                valid_move = False

                if move_type < 0.4:
                    idx = rng.randint(0, num_triples - 1)
                    t = ALL_TRIPLES[idx]
                    can_add = True
                    t_set = set(t)
                    for ct in current_triples:
                        if len(t_set & set(ct)) > 1:
                            can_add = False
                            break
                    if can_add:
                        new_triples.append(t)
                        valid_move = True
                    else:
                        for _ in range(10):
                            idx = rng.randint(0, num_triples - 1)
                            t = ALL_TRIPLES[idx]
                            t_set = set(t)
                            can_add = True
                            for ct in current_triples:
                                if len(t_set & set(ct)) > 1:
                                    can_add = False
                                    break
                            if can_add:
                                new_triples.append(t)
                                valid_move = True
                                break

                elif move_type < 0.7:
                    if len(current_triples) > 0:
                        idx = rng.randint(0, len(current_triples) - 1)
                        new_triples.pop(idx)
                        valid_move = True
                else:
                    if len(current_triples) > 0:
                        idx = rng.randint(0, len(current_triples) - 1)
                        removed = new_triples.pop(idx)
                        added = False
                        for _ in range(20):
                            nidx = rng.randint(0, num_triples - 1)
                            t = ALL_TRIPLES[nidx]
                            t_set = set(t)
                            can_add = True
                            for ct in new_triples:
                                if len(t_set & set(ct)) > 1:
                                    can_add = False
                                    break
                            if can_add:
                                new_triples.append(t)
                                added = True
                                valid_move = True
                                break
                        if not added:
                            new_triples.append(removed)
                            valid_move = False

                if not valid_move:
                    continue

                new_score = independence_number(new_triples)
                delta = new_score - current_score

                if delta < 0 or (T > 0 and rng.random() < 2.71828 ** (-delta / T)):
                    current_triples = new_triples
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = current_triples[:]

            return best_triples, best_score

        for seed_offset in range(20):
            if time.time() > deadline - 30:
                break
            initial = random_greedy_packing(seed_offset)
            if not initial:
                continue
            score = independence_number(initial)
            if score < best_score:
                best_score = score
                best_triples = initial[:]

        if best_triples is None:
            best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
            best_score = independence_number(best_triples)

        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            best_triples, best_score = local_search_sa(best_triples, max_time=min(remaining_time, 60))

    if not is_linear(best_triples):
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    return [list(t) for t in best_triples]
