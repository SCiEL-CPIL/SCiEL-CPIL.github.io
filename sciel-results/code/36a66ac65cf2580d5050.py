"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

MAX_SUM = 2000  # max possible sum of two elements from 1..1000

def _count_sums_array(chosen):
    """Count all pairwise sums a+b where a<=b from chosen. Returns list of length MAX_SUM+1."""
    counts = [0] * (MAX_SUM + 1)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            s = chosen[i] + chosen[j]
            counts[s] += 1
    return counts

def _can_add_array(chosen, counts, x):
    """Check if x can be added to chosen without violating B2[2] using array counts."""
    for y in chosen:
        s = x + y
        if counts[s] >= 2:
            return False
    if counts[2 * x] >= 2:
        return False
    return True

def _add_to_counts(counts, chosen, x):
    """Update counts in-place after adding x to chosen."""
    for y in chosen:
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1

def _remove_from_counts(counts, chosen, x):
    """Update counts in-place after removing x from chosen."""
    for y in chosen:
        if y != x:
            s = x + y
            counts[s] -= 1
    counts[2 * x] -= 1

def _greedy_construct(order, deadline=None):
    """Greedy construction following a specific order of candidates."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if _can_add_array(chosen, counts, x):
            chosen.append(x)
            _add_to_counts(counts, chosen, x)
    return chosen

def _greedy_density_aware(deadline=None):
    """Greedy that at each step picks the candidate whose addition creates the fewest new sum collisions.
    Prefers candidates with zero hard violations and minimal soft collisions."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    remaining = list(range(1, 1001))
    
    while remaining:
        if deadline is not None and time.time() > deadline:
            break
        
        # Find all valid candidates (no hard violations)
        valid_candidates = []
        for x in remaining:
            hard_violation = False
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    hard_violation = True
                    break
            if not hard_violation and counts[2 * x] >= 2:
                hard_violation = True
            if not hard_violation:
                # Count soft collisions (sums that would go from 1 to 2)
                soft = 0
                for y in chosen:
                    s = x + y
                    if counts[s] == 1:
                        soft += 1
                if counts[2 * x] == 1:
                    soft += 1
                valid_candidates.append((soft, x))
        
        if not valid_candidates:
            break
        
        # Sort by soft collisions, then by value (prefer smaller to leave room)
        valid_candidates.sort(key=lambda t: (t[0], t[1]))
        best_x = valid_candidates[0][1]
        
        remaining.remove(best_x)
        chosen.append(best_x)
        _add_to_counts(counts, chosen, best_x)
    
    return sorted(chosen)

def _greedy_weighted(order, deadline=None):
    """Greedy with a scoring function that balances density and spread."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if _can_add_array(chosen, counts, x):
            # Score: prefer elements that minimize future constraint tightness
            chosen.append(x)
            _add_to_counts(counts, chosen, x)
    return chosen

def _local_improve(chosen, deadline):
    """Local improvement: try removing each element and finding a better replacement.
    Focus on elements that cause the most 'tight' sums (count==2)."""
    chosen = sorted(chosen)
    counts = _count_sums_array(chosen)
    chosen_set = set(chosen)
    
    # Count how many sums are at capacity (==2) for each element
    def _tightness(chosen, counts):
        """Count sums at capacity (==2) that involve each element."""
        tightness = {x: 0 for x in chosen}
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                if counts[s] == 2:
                    tightness[chosen[i]] += 1
                    if j > i:
                        tightness[chosen[j]] += 1
        return tightness
    
    improved = True
    max_passes = 50
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        
        # Find the element with highest tightness
        tightness = _tightness(chosen, counts)
        sorted_by_tightness = sorted(chosen, key=lambda x: -tightness.get(x, 0))
        
        for x_remove in sorted_by_tightness:
            if time.time() > deadline:
                break
            
            # Remove x_remove
            idx = chosen.index(x_remove)
            _remove_from_counts(counts, chosen, x_remove)
            chosen_set.discard(x_remove)
            temp_chosen = chosen[:idx] + chosen[idx+1:]
            
            # Find best replacement: prefer elements that minimize new tightness
            best_add = None
            best_score = float('inf')
            
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                if not _can_add_array(temp_chosen, counts, x):
                    continue
                # Score: count how many sums would be at capacity after adding x
                new_tight = 0
                for y in temp_chosen:
                    s = x + y
                    if counts[s] == 1:
                        new_tight += 1
                if counts[2 * x] == 1:
                    new_tight += 1
                # Also prefer elements that are "central" to reduce future constraints
                if new_tight < best_score:
                    best_score = new_tight
                    best_add = x
                elif new_tight == best_score and best_add is not None:
                    # Tie-break: prefer smaller x
                    if x < best_add:
                        best_add = x
            
            if best_add is not None:
                # Accept the swap
                temp_chosen.append(best_add)
                _add_to_counts(counts, temp_chosen, best_add)
                chosen_set.add(best_add)
                chosen = sorted(temp_chosen)
                improved = True
            else:
                # Put x_remove back
                chosen_set.add(x_remove)
                chosen = sorted(temp_chosen + [x_remove])
                _add_to_counts(counts, chosen, x_remove)
        
        # After one full pass, if we improved, do another pass
        if not improved:
            break
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (5s buffer before 150s kill)
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Density-aware greedy (most promising)
    set1 = _greedy_density_aware(deadline=deadline - 100)
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
    
    if time.time() > deadline - 80:
        return best_set
    
    # Strategy 2: Ascending greedy
    order2 = list(range(1, 1001))
    set2 = _greedy_construct(order2, deadline=deadline - 70)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2
    
    if time.time() > deadline - 60:
        return best_set
    
    # Strategy 3: Descending greedy
    order3 = list(range(1000, 0, -1))
    set3 = _greedy_construct(order3, deadline=deadline - 50)
    if len(set3) > best_size:
        best_size = len(set3)
        best_set = set3
    
    if time.time() > deadline - 40:
        return best_set
    
    # Strategy 4: Multiple random shuffles
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        order = list(range(1, 1001))
        random.shuffle(order)
        s = _greedy_construct(order, deadline=deadline - 20)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local improvement on the best set found
    if time.time() < deadline - 5:
        improved = _local_improve(best_set, deadline=deadline - 2)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set
# EVOLVE-BLOCK-END
