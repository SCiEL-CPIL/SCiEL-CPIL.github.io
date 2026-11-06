"""Improved program: greedy + local search for max 3-APs without 4-APs in [1,256]."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d)
    # For each element x, store which 4-APs it participates in
    four_ap_conflicts = [set() for _ in range(N + 1)]
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            ap = (a, b, c, e)
            all_4aps.append(ap)
            for x in ap:
                four_ap_conflicts[x].add(ap)
    
    # Precompute: for each element x, which other elements would complete a 4-AP with x
    # Actually, let's precompute for each pair (x, y) whether adding y to a set containing x
    # could create a 4-AP. Better: for each x, precompute the set of "forbidden" elements
    # given current set. We'll do this incrementally.
    
    # Precompute 3-APs: (a, a+d, a+2d)
    # For counting 3-APs in a set, we can iterate over pairs
    # Precompute for each x, the list of (y, z) such that x,y,z form a 3-AP (x<y<z, y-x=z-y)
    # i.e., y = x + d, z = x + 2d for various d
    # Or: for each middle element m, the pairs (m-d, m+d)
    
    # For fast 3-AP counting when adding element x:
    # A new 3-AP involving x can be:
    # - x is the first: x, x+d, x+2d (need x+d, x+2d in set)
    # - x is the middle: x-d, x, x+d (need x-d, x+d in set)
    # - x is the last: x-2d, x-d, x (need x-2d, x-d in set)
    
    def count_3aps_involving(x, selected):
        """Count 3-APs that include x and at least one other element in selected."""
        count = 0
        # x is first term: x, x+d, x+2d
        for d in range(1, N - x + 1):
            if (x + d) in selected and (x + 2 * d) in selected:
                count += 1
        # x is middle term: x-d, x, x+d
        for d in range(1, min(x, N - x) + 1):
            if (x - d) in selected and (x + d) in selected:
                count += 1
        # x is last term: x-2d, x-d, x
        for d in range(1, (x + 1) // 2):
            if (x - 2 * d) in selected and (x - d) in selected:
                count += 1
        return count
    
    def would_create_4ap(x, selected):
        """Check if adding x to selected would create a 4-AP."""
        for ap in four_ap_conflicts[x]:
            # Check if the other 3 elements of this 4-AP are all in selected
            others = [y for y in ap if y != x]
            if all(y in selected for y in others):
                return True
        return False
    
    def total_3aps(selected):
        """Count all distinct 3-APs in selected."""
        count = 0
        sel = sorted(selected)
        sel_set = set(sel)
        for i, a in enumerate(sel):
            for j in range(i + 1, len(sel)):
                b = sel[j]
                d = b - a
                c = b + d
                if c in sel_set:
                    count += 1
        return count
    
    # Strategy 1: Greedy in natural order
    def greedy_natural():
        selected = set()
        for x in range(1, N + 1):
            if time.time() - start_time > DEADLINE:
                break
            if not would_create_4ap(x, selected):
                selected.add(x)
        return selected
    
    # Strategy 2: Greedy with random order
    def greedy_random(seed):
        rng = random.Random(seed)
        selected = set()
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() - start_time > DEADLINE:
                break
            if not would_create_4ap(x, selected):
                selected.add(x)
        return selected
    
    # Strategy 3: Greedy prioritizing elements that create more 3-APs
    def greedy_by_gain():
        selected = set()
        remaining = set(range(1, N + 1))
        while remaining:
            if time.time() - start_time > DEADLINE:
                break
            best_x = None
            best_gain = -1
            for x in remaining:
                if would_create_4ap(x, selected):
                    continue
                gain = count_3aps_involving(x, selected)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            selected.add(best_x)
            remaining.discard(best_x)
        return selected
    
    # Local search
    def local_search(selected, max_iters=5000):
        selected = set(selected)
        best_score = total_3aps(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            iters += 1
            
            # Try adding elements
            for x in range(1, N + 1):
                if x in selected:
                    continue
                if would_create_4ap(x, selected):
                    continue
                new_selected = selected | {x}
                score = total_3aps(new_selected)
                if score > best_score:
                    selected = new_selected
                    best_score = score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing elements
            for x in list(selected):
                new_selected = selected - {x}
                score = total_3aps(new_selected)
                # We want to remove if it allows adding more later, but for now just check
                # Actually, removing reduces 3-APs, so we only remove if it enables a better add
                pass  # Skip pure removals for now
            
            # Try swaps: remove x, add y
            for x in list(selected):
                for y in range(1, N + 1):
                    if y in selected:
                        continue
                    new_selected = (selected - {x}) | {y}
                    if would_create_4ap(y, selected - {x}):
                        continue
                    score = total_3aps(new_selected)
                    if score > best_score:
                        selected = new_selected
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            
            # Try remove + add (net zero or positive change)
            if not improved:
                for x in list(selected):
                    for y in range(1, N + 1):
                        if y in selected:
                            continue
                        temp = selected - {x}
                        if would_create_4ap(y, temp):
                            continue
                        new_selected = temp | {y}
                        score = total_3aps(new_selected)
                        if score > best_score:
                            selected = new_selected
                            best_score = score
                            improved = True
                            break
                    if improved:
                        break
        
        return selected, best_score
    
    # Run strategies and keep best
    best_set = set()
    best_score = -1
    
    # Strategy 1: Natural greedy
    s1 = greedy_natural()
    score1 = total_3aps(s1)
    if score1 > best_score:
        best_score = score1
        best_set = s1
    
    # Strategy 2: Gain-based greedy
    if time.time() - start_time < DEADLINE:
        s2 = greedy_by_gain()
        score2 = total_3aps(s2)
        if score2 > best_score:
            best_score = score2
            best_set = s2
    
    # Strategy 3: Random greedy (a few seeds)
    if time.time() - start_time < DEADLINE:
        for seed in range(5):
            if time.time() - start_time > DEADLINE:
                break
            s3 = greedy_random(seed)
            score3 = total_3aps(s3)
            if score3 > best_score:
                best_score = score3
                best_set = s3
    
    # Local search on the best
    if time.time() - start_time < DEADLINE:
        best_set, best_score = local_search(best_set)
    
    # Try local search on other good solutions too
    if time.time() - start_time < DEADLINE:
        s1 = greedy_natural()
        s1, sc1 = local_search(s1, max_iters=2000)
        if sc1 > best_score:
            best_score = sc1
            best_set = s1
    
    if time.time() - start_time < DEADLINE:
        s2 = greedy_by_gain()
        s2, sc2 = local_search(s2, max_iters=2000)
        if sc2 > best_score:
            best_score = sc2
            best_set = s2
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
