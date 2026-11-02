import time
import random
import sys

sys.setrecursionlimit(10000)


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all differences between two entries of the same row are distinct over all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 115  # Safety margin for 120s limit

    # Phase 1: Multi-start greedy initialization
    best_solution = None
    best_score = float('inf')

    # Try several greedy starts with different randomizations
    n_starts = 10
    for s in range(n_starts):
        if time.time() > deadline - 60:
            break
        sol = _greedy_construct(n_rows, k, random_seed=s * 7919 + 13)
        if sol:
            score = max(max(r) for r in sol)
            if score < best_score:
                best_score = score
                best_solution = [r[:] for r in sol]

    if best_solution is None:
        best_solution = _greedy_construct(n_rows, k, random_seed=0)
        if not best_solution:
            return []
        best_score = max(max(r) for r in best_solution)

    # Phase 2: Simulated Annealing with Row Replacement
    # We use SA to escape local minima by allowing temporary increases in scope
    # or accepting neutral moves that change the structure.
    current_solution = [r[:] for r in best_solution]
    current_score = best_score
    
    # SA Parameters
    # Temperature should start high enough to accept worse solutions
    # The "cost" is the difference in max value.
    T = 5.0 
    T_min = 0.1
    cooling_rate = 0.995
    
    # Pre-compute diffs for current solution to speed up validity checks
    # Actually, for row replacement, we need diffs of OTHER rows.
    
    steps = 0
    while time.time() < deadline - 10 and T > T_min:
        steps += 1
        # Pick a random row to modify
        r_idx = random.randint(0, n_rows - 1)
        
        # Calculate diffs of all OTHER rows
        other_diffs = set()
        for i in range(n_rows):
            if i == r_idx:
                continue
            r = current_solution[i]
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    other_diffs.add(r[b] - r[a])
        
        # Try to generate a new row that is valid against other_diffs
        # We want to minimize the max value of this row, ideally below current max
        # But SA allows us to accept rows that might be slightly worse if they
        # lead to a better global state later.
        
        # Strategy: Try to find a row with max <= current_score (neutral) or < current_score (improvement)
        # If we find one, we accept it.
        # If we don't find one quickly, we might try to find one with max <= current_score + T
        # to allow exploration.
        
        max_allowed = current_score
        # Occasionally, allow exploring slightly larger scopes to escape
        if random.random() < 0.1 and T > 1.0:
            max_allowed = current_score + int(T)
            
        new_row = _generate_row(k, other_diffs, max_allowed, random_seed=random.randint(0, 10000))
        
        if new_row is not None:
            new_max = max(new_row)
            new_solution = [r[:] for r in current_solution]
            new_solution[r_idx] = new_row
            new_score = max(max(r) for r in new_solution)
            
            delta = new_score - current_score
            
            # Metropolis criterion
            if delta < 0:
                # Improvement, always accept
                current_solution = new_solution
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_solution = [r[:] for r in current_solution]
            else:
                # Worse or equal, accept with probability exp(-delta / T)
                # If delta is 0, we accept to reshuffle
                if delta == 0:
                    accept_prob = 0.5 # Half the time accept neutral moves to explore
                else:
                    accept_prob = pow(2.718281828, -delta / T)
                
                if random.random() < accept_prob:
                    current_solution = new_solution
                    current_score = new_score
            
        T *= cooling_rate

    # Phase 3: Final Local Search Polish
    # Use the best solution found so far
    if time.time() < deadline - 2:
        final = _local_search_minimize([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score

    return best_solution


def _generate_row(k, forbidden_diffs, max_val, random_seed=0):
    """
    Generate a single valid row of length k+1 starting at 0.
    The row must not use any difference in forbidden_diffs.
    The maximum value in the row must be <= max_val.
    Returns the row as a list, or None if not found.
    """
    rng = random.Random(random_seed)
    n = k + 1
    row = [0]
    
    # We need to fill positions 1..k
    # We use a backtracking/greedy hybrid approach
    # To make it faster, we try to keep values small.
    
    # Recursive helper
    def backtrack(pos, current_row):
        if pos == n:
            return current_row
        
        last_val = current_row[-1]
        # Try values from last_val + 1 up to max_val
        # To improve diversity, we can randomize the order slightly or start from a random point
        # But greedy (smallest first) is good for minimizing max.
        
        # Optimization: If we are at the last position, we just need to check validity.
        # We can also prune: if max_val - last_val < (n - pos), it's impossible? No, differences don't have to be 1.
        
        # To speed up, we can limit the search range.
        # If we pick a value too high, we might exceed max_val later.
        # Simple heuristic: iterate from last_val + 1 to max_val
        
        # Randomization for diversity in SA context?
        # If we always pick the smallest valid, we get the same row every time for a given forbidden set.
        # We need diverse rows for SA to work.
        # So, we should sample from valid candidates.
        
        candidates = []
        # Limit candidates to avoid O(max_val^k) explosion in backtracking
        # We only need a few good candidates
        max_candidates = 50
        
        for val in range(last_val + 1, max_val + 1):
            # Check validity against current_row and forbidden_diffs
            valid = True
            new_diffs = set()
            for prev in current_row:
                d = val - prev
                if d in forbidden_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                candidates.append(val)
                if len(candidates) >= max_candidates:
                    break
        
        if not candidates:
            return None
        
        # Pick a random candidate to allow different structures
        # But bias towards smaller values to keep max low
        # Weighted choice: smaller values have higher weight
        weights = [1.0 / (c + 1) for c in candidates]
        # Normalize not strictly necessary for random.choices
        chosen_val = rng.choices(candidates, weights=weights, k=1)[0]
        
        res = backtrack(pos + 1, current_row + [chosen_val])
        if res:
            return res
        
        # If backtracking fails, we could try other candidates, but that's expensive.
        # For simplicity in this phase, we just return None if the first random pick fails deep down.
        # Actually, let's do a limited retry: if the chosen path fails, try the next smallest?
        # Given the time constraints, a simple random pick is often sufficient for SA to explore.
        return None

    return backtrack(1, row)


def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Local search that tries to reduce the maximum value by moving elements to smaller positions.
    Uses incremental diff tracking for speed.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1

    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.add(r[j] - r[i])

    no_progress_rounds = 0

    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k + 1)]
        random.shuffle(indices)

        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break

            row = best_rows[r_idx]
            old_val = row[pos]

            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)

            forbidden_diffs = all_diffs - diffs_involving

            # Try to find smaller valid value
            lower_bound = row[pos - 1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos + 1] - 1)

            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break

                if found:
                    continue

        if not progress:
            no_progress_rounds += 1
            if no_progress_rounds >= 3:
                break
        else:
            no_progress_rounds = 0

    return best_rows


def _greedy_construct(n_rows, k, random_seed=0):
    """
    Greedy construction: build rows one by one, choosing smallest possible values.
    Uses randomization to explore different starting points.
    """
    rng = random.Random(random_seed)
    used_diffs = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            base = current_max + 1
            search_range = min(200, 300 - current_max)

            # First try sequential
            for x in range(base, base + search_range):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break

            if not found:
                # Try randomized search
                for _retry in range(200):
                    x = base + rng.randint(0, 150)
                    if x > 250:
                        continue
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    new_diffs.append(d)

                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                if not found:
                    return None
        rows.append(row)

    return rows
