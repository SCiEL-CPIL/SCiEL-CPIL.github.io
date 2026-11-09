import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = 180
    
    # Precompute all forbidden triples (a, b, c) with distinct elements in [1, n]
    # such that 1/a = 1/b + 1/c. This is equivalent to c = ab/(a+b) being an integer.
    # We generate all unique sets {a, b, c}.
    forbidden = []
    unique_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if c != a and c != b:
                    # Sort to create a canonical representation for uniqueness
                    t = tuple(sorted((a, b, c)))
                    if t not in unique_triples:
                        unique_triples.add(t)
                        forbidden.append(t)
    
    # Build incidence lists: element -> list of triple indices containing it
    elem_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(forbidden):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)

    def count_conflicts(in_set, x):
        """Count how many forbidden triples are fully active if x were in the set."""
        count = 0
        for idx in elem_to_triples[x]:
            a, b, c = forbidden[idx]
            others = [y for y in (a, b, c) if y != x]
            if in_set[others[0]] and in_set[others[1]]:
                count += 1
        return count

    def polish(in_set, removed):
        """
        Greedily try to add back removed elements.
        Priority: elements with fewer conflicts (0 is best).
        Repeat until no more can be added.
        """
        changed = True
        while changed:
            changed = False
            # Sort removed elements by current conflict count (ascending)
            # We only consider elements not currently in the set
            candidates = [x for x in removed if not in_set[x]]
            if not candidates:
                break
            
            # Compute conflicts for each candidate
            scored = []
            for x in candidates:
                conf = count_conflicts(in_set, x)
                scored.append((conf, x))
            
            # Sort by conflicts, then by value (to break ties deterministically/randomly)
            scored.sort(key=lambda t: (t[0], t[1]))
            
            for conf, x in scored:
                if conf == 0:
                    in_set[x] = True
                    removed.discard(x)
                    changed = True
                    # After adding one, conflict counts for others might change,
                    # so we break and restart the loop to re-evaluate
                    break
        return in_set, removed

    def solve_greedy_removal(seed):
        """
        Phase 1: Start with all elements.
        Iteratively remove the element involved in the most active triples.
        Tie-breaking: random.
        Phase 2: Polish to add back elements.
        """
        random.seed(seed)
        
        in_set = [True] * (n + 1)
        in_set[0] = False
        removed = set()
        
        # Track active status of triples
        # A triple is active if all 3 elements are in in_set
        # We can maintain a count of "violations" per element
        # Or simpler: just scan. But N=180, T is small (~few hundred?).
        # Let's count triples per element initially.
        
        # For efficiency, let's maintain active_triple_count for each element
        # active_count[x] = number of triples containing x where all 3 are in in_set
        
        active_count = [0] * (n + 1)
        for idx, (a, b, c) in enumerate(forbidden):
            # Initially all are in set
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
        
        while True:
            if time.time() > deadline - 5:
                break
            
            # Find max active count among elements in set
            max_count = 0
            candidates = []
            for x in range(1, n + 1):
                if in_set[x]:
                    if active_count[x] > max_count:
                        max_count = active_count[x]
                        candidates = [x]
                    elif active_count[x] == max_count:
                        candidates.append(x)
            
            if max_count == 0:
                break
            
            # Pick random candidate
            x = random.choice(candidates)
            
            # Remove x
            in_set[x] = False
            removed.add(x)
            
            # Update active counts
            # For each triple containing x, it becomes inactive.
            # Decrement active_count for the other two elements in that triple.
            for idx in elem_to_triples[x]:
                a, b, c = forbidden[idx]
                # The triple was active because x was in set and others were (implicitly, 
                # because active_count[x] counted it). 
                # Now x is out, so this triple is no longer active for the other two.
                # We need to decrement active_count for the other two.
                # Note: We don't care about active_count[x] anymore since x is removed.
                
                # Identify the other two
                others = [y for y in (a, b, c) if y != x]
                y1, y2 = others[0], others[1]
                
                # Decrement if they are still in set (they are, because we only remove one at a time 
                # and we are processing the consequence of removing x)
                # Actually, we must ensure we don't double decrement if the triple was already inactive?
                # No, active_count[x] included this triple, which means at the time of counting, 
                # the other two were in the set. Since we haven't removed them yet, they are still in.
                # So the triple was active for y1 and y2 as well.
                # So we decrement y1 and y2.
                
                if in_set[y1]:
                    active_count[y1] -= 1
                if in_set[y2]:
                    active_count[y2] -= 1
        
        # Phase 2: Polish
        in_set, removed = polish(in_set, removed)
        
        return [x for x in range(1, n + 1) if in_set[x]]

    def solve_greedy_addition(seed):
        """
        Alternative: Start with empty set.
        Add elements in order of least conflicts (0 first).
        """
        random.seed(seed)
        in_set = [False] * (n + 1)
        in_set[0] = False
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # We will repeatedly pick the candidate with minimum conflicts
        # If multiple have 0 conflicts, pick one randomly.
        
        while candidates:
            if time.time() > deadline - 5:
                break
            
            # Calculate conflicts for all candidates
            min_conf = float('inf')
            best_candidates = []
            
            for x in candidates:
                conf = count_conflicts(in_set, x)
                if conf < min_conf:
                    min_conf = conf
                    best_candidates = [x]
                elif conf == min_conf:
                    best_candidates.append(x)
            
            # If min_conf > 0, adding any will create a conflict?
            # No, count_conflicts returns the number of NEW conflicts created.
            # If min_conf > 0, it means we can't add any without creating a conflict.
            # But we can still add them if we are willing to violate? No, we must maintain validity.
            # So if min_conf > 0, we stop?
            # Wait, if min_conf > 0, it means for every x, there is at least one triple 
            # where the other two are already in the set.
            # So we cannot add x without creating a forbidden triple.
            # Thus, we are stuck?
            # Actually, we might be able to add x if we remove one of the others?
            # But this is a pure addition heuristic.
            # So if min_conf > 0, we break.
            
            if min_conf > 0:
                break
            
            # Pick one from best_candidates (all have 0 conflicts)
            x = random.choice(best_candidates)
            in_set[x] = True
            candidates.remove(x)
            
        # Polish is less necessary here as we only added valid ones, 
        # but maybe we skipped some that could fit if we rearranged?
        # Polish tries to add back "removed" ones, but here nothing was removed.
        # So just return.
        return [x for x in range(1, n + 1) if in_set[x]]

    best_result = None
    best_size = 0

    # Run multiple trials of the removal heuristic
    trials = 0
    while time.time() < deadline - 5:
        # Alternate between removal and addition heuristics?
        # Removal seems more robust for maximizing size from full set.
        # Addition might get stuck in local minima early.
        
        result = solve_greedy_removal(trials)
        
        # Validate result (safety check)
        valid = True
        for a, b, c in forbidden:
            if a in result and b in result and c in result:
                valid = False
                break
        
        if valid and len(result) > best_size:
            best_size = len(result)
            best_result = result
            
        trials += 1
        
        # Occasionally try addition heuristic
        if trials % 10 == 0:
            result_add = solve_greedy_addition(trials)
            valid_add = True
            for a, b, c in forbidden:
                if a in result_add and b in result_add and c in result_add:
                    valid_add = False
                    break
            if valid_add and len(result_add) > best_size:
                best_size = len(result_add)
                best_result = result_add

    if not best_result:
        # Fallback: a large valid set.
        # Numbers > 90 are safe? 
        # If a, b, c > 90, then 1/a, 1/b, 1/c < 1/90.
        # 1/a = 1/b + 1/c < 2/90 = 1/45.
        # So a > 45.
        # This doesn't guarantee no triples.
        # However, the set [91, ..., 180] has size 90.
        # Let's just return the best found or a simple valid set.
        # [180] is valid.
        # Better fallback: all even numbers? No.
        # Let's use the result of the first trial if it failed validation?
        # No, just return [180] to be safe if something went wrong.
        best_result = [180]

    return best_result
