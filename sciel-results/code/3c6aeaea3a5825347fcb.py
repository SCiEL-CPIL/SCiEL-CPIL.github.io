import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    all_elements = list(range(1, 25))
    
    def find_violation(S):
        """
        Returns (x, subset) if 1/x == sum(1/y for y in subset) for some x in S and subset of S\{x}.
        Returns None if valid.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return None
            
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # MITM
            left_sums = [Fraction(0)]
            for y in left:
                ry = recip[y]
                new_sums = [s + ry for s in left_sums]
                left_sums.extend(new_sums)
                
            right_sums = [Fraction(0)]
            for y in right:
                ry = recip[y]
                new_sums = [s + ry for s in right_sums]
                right_sums.extend(new_sums)
                
            right_set = set(right_sums)
            
            # Check if target is in left_sums + right_set
            found = False
            for l in left_sums:
                if (target - l) in right_set:
                    found = True
                    break
                    
            if found:
                return (x, others)
                
        return None

    def is_valid(S):
        return find_violation(S) is None

    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Precompute conflict information: for each x, which subsets of other elements sum to 1/x
    # This helps guide removal decisions
    def compute_conflicts(S):
        """For each x in S, find all minimal subsets T of S\{x} such that 1/x = sum(1/t for t in T)."""
        S_list = list(S)
        conflicts = {x: [] for x in S}
        
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            m = len(others)
            if m <= 10:
                # Brute force all subsets
                from itertools import combinations
                for r in range(1, m+1):
                    for combo in combinations(others, r):
                        s = sum(recip[y] for y in combo)
                        if s == target:
                            conflicts[x].append(list(combo))
            else:
                # Use MITM to find at least one violation
                half = m // 2
                left = others[:half]
                right = others[half:]
                
                left_sums = [Fraction(0)]
                left_masks = [0]
                for idx, y in enumerate(left):
                    ry = recip[y]
                    new_sums = [s + ry for s in left_sums]
                    left_sums.extend(new_sums)
                    left_masks.extend([mask | (1 << idx) for mask in left_masks])
                    
                right_sums = [Fraction(0)]
                right_masks = [0]
                for idx, y in enumerate(right):
                    ry = recip[y]
                    new_sums = [s + ry for s in right_sums]
                    right_sums.extend(new_sums)
                    right_masks.extend([mask | (1 << idx) for mask in right_masks])
                    
                right_map = {}
                for s, mask in zip(right_sums, right_masks):
                    if s not in right_map or mask < right_map[s]:
                        right_map[s] = mask
                        
                for ls, lm in zip(left_sums, left_masks):
                    rem = target - ls
                    if rem in right_map:
                        # Reconstruct subset
                        subset = []
                        for idx in range(len(left)):
                            if lm & (1 << idx):
                                subset.append(left[idx])
                        for idx in range(len(right)):
                            if right_map[rem] & (1 << idx):
                                subset.append(right[idx])
                        conflicts[x].append(subset)
                        break  # One is enough for guidance
        
        return conflicts

    # Strategy 1: Branch and bound on conflict graph
    def branch_and_bound():
        """Try to find a large valid set using branch and bound."""
        best = set()
        
        def dfs(current, remaining, depth):
            nonlocal best
            if time.time() > deadline:
                return
            if len(current) > len(best):
                best = set(current)
            
            if not remaining:
                return
            
            # Upper bound: current size + remaining size
            if len(current) + len(remaining) <= len(best):
                return
            
            # Try adding the first element
            x = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include x
            new_current = current | {x}
            if is_valid(new_current):
                # Filter remaining to only elements compatible with new_current
                compatible = [y for y in rest if is_valid(new_current | {y})]
                dfs(new_current, compatible, depth + 1)
            
            # Option 2: Exclude x
            dfs(current, rest, depth + 1)
        
        # Start with elements in some order
        # Try different orderings
        for order in [all_elements, sorted(all_elements, reverse=True), 
                     random.sample(all_elements, 24)]:
            if time.time() > deadline:
                break
            dfs(set(), list(order), 0)
            
        return best

    # Run branch and bound first
    bb_result = branch_and_bound()
    if bb_result:
        update_best(bb_result)
    
    # Strategy 2: Iterative Repair and Grow (original approach, enhanced)
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Initialize with full set or random large set
        if attempt == 0:
            S = set(all_elements)
        else:
            size = random.randint(15, 20)
            S = set(random.sample(all_elements, size))
            
        # Phase 1: Repair with conflict-aware removal
        max_repair_steps = 50
        for _ in range(max_repair_steps):
            if time.time() > deadline:
                break
            viol = find_violation(S)
            if viol is None:
                break
            
            x, others = viol
            # Compute conflicts to make a smarter removal choice
            conflicts = compute_conflicts(S)
            
            # Score each candidate for removal: prefer removing elements that appear in many conflicts
            candidate_scores = {}
            for elem in [x] + list(others):
                count = 0
                for other_x, subsets in conflicts.items():
                    for subset in subsets:
                        if elem in subset:
                            count += 1
                # Also count if elem is a target with conflicts
                if elem in conflicts and conflicts[elem]:
                    count += len(conflicts[elem])
                candidate_scores[elem] = count
            
            # Remove the element with highest score (most "conflicted")
            if candidate_scores:
                max_score = max(candidate_scores.values())
                top_candidates = [k for k, v in candidate_scores.items() if v == max_score]
                rem = random.choice(top_candidates)
            else:
                rem = random.choice([x] + list(others))
            S.remove(rem)
            
        if not S:
            continue
            
        update_best(S)
        
        # Phase 2: Grow
        complement = list(set(all_elements) - S)
        random.shuffle(complement)
        
        for c in complement:
            if time.time() > deadline:
                break
            new_S = S | {c}
            if is_valid(new_S):
                S = new_S
                update_best(S)
                
        # Phase 3: Perturb
        if len(S) > 5:
            kick_size = random.randint(1, 2)
            to_remove = random.sample(list(S), kick_size)
            for r in to_remove:
                S.remove(r)
                
            complement = list(set(all_elements) - S)
            random.shuffle(complement)
            for c in complement:
                if time.time() > deadline:
                    break
                new_S = S | {c}
                if is_valid(new_S):
                    S = new_S
                    update_best(S)

    # Fallback
    if not best_set:
        best_set = {24}
        
    return list(best_set)
