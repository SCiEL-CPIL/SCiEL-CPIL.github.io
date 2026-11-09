import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1

    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])

    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def check_sidon_incremental(S, x, used_diffs):
        """Check if adding x to S (sorted) maintains Sidon property."""
        for s in S:
            d = abs(x - s)
            if d in used_diffs:
                return False
        return True

    def compute_used_diffs(S):
        return get_diffs(S)

    def covered_mask_from_diffs(diffs):
        mask = 0
        for d in diffs:
            if 1 <= d <= 30:
                mask |= (1 << (d - 1))
        return mask

    # Greedy construction to get a valid starting solution
    def greedy_construct():
        a = [0]
        used_diffs = set()
        covered_mask = 0
        
        for _ in range(31):
            if time.time() - start_time > deadline - 10:
                break
                
            missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
            if not missing:
                break
            
            best_x = None
            best_score = -1
            
            # Generate candidates from existing elements + missing diffs
            candidates = set()
            for d in missing:
                for s in a:
                    x1 = s + d
                    x2 = s - d
                    if x1 > 0:
                        candidates.add(x1)
                    if x2 > 0:
                        candidates.add(x2)
            
            # Also add some small values
            for i in range(1, 50):
                candidates.add(i)
            
            for x in candidates:
                if x in a or x <= 0:
                    continue
                if not check_sidon_incremental(a, x, used_diffs):
                    continue
                
                # Calculate score: how many new missing diffs covered, then smaller x
                new_covered = 0
                for s in a:
                    d = abs(x - s)
                    if 1 <= d <= 30 and not (covered_mask & (1 << (d - 1))):
                        new_covered += 1
                
                score = (new_covered, -x)
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is not None:
                a.append(best_x)
                a.sort()
                for s in a[:-1]:
                    d = abs(best_x - s)
                    used_diffs.add(d)
                    if 1 <= d <= 30:
                        covered_mask |= (1 << (d - 1))
            else:
                # Fallback: add a large number
                x = 2 * max(a) + 3
                while True:
                    if check_sidon_incremental(a, x, used_diffs):
                        break
                    x += 1
                a.append(x)
                a.sort()
                for s in a[:-1]:
                    d = abs(x - s)
                    used_diffs.add(d)
                    if 1 <= d <= 30:
                        covered_mask |= (1 << (d - 1))
        
        return a, covered_mask, used_diffs

    # Backtracking with pruning to minimize max element
    def backtrack(S, used_diffs, covered_mask, depth, max_depth, best_max):
        if time.time() - start_time > deadline:
            return None, best_max
            
        if covered_mask == full_mask:
            current_max = max(S)
            if current_max < best_max:
                best_max = current_max
            return list(S), best_max
            
        if depth >= max_depth:
            return None, best_max
            
        if max(S) >= best_max:
            return None, best_max
            
        missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # Generate candidates
        candidates = set()
        for d in missing:
            for s in S:
                x1 = s + d
                x2 = s - d
                if x1 > 0 and x1 < best_max:
                    candidates.add(x1)
                if x2 > 0 and x2 < best_max:
                    candidates.add(x2)
        
        # Score candidates
        def candidate_score(x):
            if x in S or x <= 0 or x >= best_max:
                return -1
            if not check_sidon_incremental(S, x, used_diffs):
                return -1
            new_covered = 0
            for s in S:
                d = abs(x - s)
                if 1 <= d <= 30 and not (covered_mask & (1 << (d - 1))):
                    new_covered += 1
            return (-new_covered, x)
        
        valid_candidates = []
        for x in candidates:
            if x <= 0 or x in S or x >= best_max:
                continue
            if not check_sidon_incremental(S, x, used_diffs):
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            return None, best_max
        
        valid_candidates.sort(key=candidate_score)
        
        # Limit candidates to avoid explosion
        for x in valid_candidates[:15]:
            new_S = S + [x]
            new_S.sort()
            new_used_diffs = set(used_diffs)
            new_mask = covered_mask
            for s in S:
                d = abs(x - s)
                new_used_diffs.add(d)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            res, best_max = backtrack(new_S, new_used_diffs, new_mask, depth + 1, max_depth, best_max)
            if res is not None:
                return res, best_max
        
        return None, best_max

    # Local search / hill climbing to improve solution
    def local_search(S, max_iters=5000):
        S = list(S)
        used_diffs = compute_used_diffs(S)
        covered_mask = covered_mask_from_diffs(used_diffs)
        current_max = max(S)
        
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            if covered_mask == full_mask and current_max < 100:
                break
                
            improved = False
            # Try removing an element and replacing with a smaller one
            for i in range(len(S)):
                if time.time() - start_time > deadline - 5:
                    break
                    
                old_val = S[i]
                S.remove(old_val)
                temp_used = compute_used_diffs(S)
                temp_mask = covered_mask_from_diffs(temp_used)
                
                # Try to add a smaller value
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    if not check_sidon_incremental(S, new_val, temp_used):
                        continue
                    
                    new_S = S + [new_val]
                    new_S.sort()
                    new_used = compute_used_diffs(new_S)
                    new_mask = covered_mask_from_diffs(new_used)
                    
                    if new_mask == full_mask and max(new_S) < current_max:
                        S = new_S
                        used_diffs = new_used
                        covered_mask = new_mask
                        current_max = max(S)
                        improved = True
                        break
                
                if not improved:
                    S.append(old_val)
                    S.sort()
                    used_diffs = compute_used_diffs(S)
                    covered_mask = covered_mask_from_diffs(used_diffs)
                else:
                    break
            
            if not improved:
                break
        
        return S, covered_mask

    # Main construction
    best_solution = None
    best_max = float('inf')

    # 1. Greedy initialization
    greedy_sol, greedy_mask, greedy_diffs = greedy_construct()
    if greedy_mask == full_mask:
        best_solution = greedy_sol
        best_max = max(greedy_sol)
    
    # 2. Backtracking to improve
    if best_solution is not None:
        res, best_max = backtrack(best_solution, greedy_diffs, greedy_mask, len(best_solution), 32, best_max)
        if res is not None:
            best_solution = res
    
    # 3. Local search refinement
    if best_solution is not None:
        refined_sol, refined_mask = local_search(best_solution)
        if refined_mask == full_mask:
            refined_max = max(refined_sol)
            if refined_max < best_max:
                best_solution = refined_sol
                best_max = refined_max
    
    # 4. If still no valid solution, do a more aggressive greedy
    if best_solution is None:
        for attempt in range(10):
            if time.time() - start_time > deadline - 5:
                break
            random.seed(attempt)
            sol, mask, diffs = greedy_construct()
            if mask == full_mask:
                if max(sol) < best_max:
                    best_solution = sol
                    best_max = max(sol)
                break
    
    # Final validation
    if best_solution is None:
        # Absolute fallback
        best_solution = [0]
        used = set()
        for d in range(1, 31):
            x = max(best_solution) + d + 1
            while not check_sidon_incremental(best_solution, x, used):
                x += 1
            best_solution.append(x)
            best_solution.sort()
            for s in best_solution[:-1]:
                used.add(abs(x - s))
    
    # Ensure sorted and valid
    best_solution = sorted(set(best_solution))
    
    return best_solution
