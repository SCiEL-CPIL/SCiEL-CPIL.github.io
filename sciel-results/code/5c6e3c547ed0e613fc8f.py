"""Combined program: greedy + simulated annealing for B2[2] sets in 1..1000."""
import time
import random
import math
import sys

def _count_sums(chosen):
    """Return dict: sum -> count of representations a+b with a<=b from chosen."""
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            s = a + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    return counts

def _is_valid(chosen):
    """Check if chosen is a valid B2[2] set (every sum has at most 2 reps)."""
    counts = _count_sums(chosen)
    return all(c <= 2 for c in counts.values())

def _greedy_build():
    """Greedy: add x if it keeps the B2[2] constraint."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = []
        for y in chosen:
            new_sums.append(x + y)
        new_sums.append(2 * x)
        trial = dict(counts)
        ok = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            counts = trial
    return chosen

def _try_add(chosen, counts, x):
    """Try to add x to chosen; return (success, new_counts)."""
    new_sums = []
    for y in chosen:
        new_sums.append(x + y)
    new_sums.append(2 * x)
    trial = dict(counts)
    for s in new_sums:
        trial[s] = trial.get(s, 0) + 1
        if trial[s] > 2:
            return False, None
    return True, trial

def _local_search(chosen, counts, deadline, seed=42):
    """Simulated annealing with add/swap/remove moves to improve the set size."""
    rng = random.Random(seed)
    chosen_set = set(chosen)
    best_size = len(chosen)
    best_chosen = list(chosen)
    best_counts = dict(counts)
    
    # Track conflicts: number of sums with count > 2
    def get_conflicts(cnt):
        c = 0
        for s, val in cnt.items():
            if val > 2:
                c += val - 2
        return c
    
    current_conflicts = get_conflicts(counts)
    T = 10.0
    cooling = 0.9995
    
    while time.time() < deadline and T > 0.01:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.3 and len(chosen) < 1000:
            # Try ADD
            candidates = [x for x in range(1, 1001) if x not in chosen_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            
            # Calculate delta conflicts
            delta_conf = 0
            new_counts = dict(counts)
            for y in chosen:
                s = x + y
                old = new_counts.get(s, 0)
                new_counts[s] = old + 1
                if old == 2:
                    delta_conf += 1
                elif old > 2:
                    delta_conf += 1  # still violating
            s2 = 2 * x
            old = new_counts.get(s2, 0)
            new_counts[s2] = old + 1
            if old == 2:
                delta_conf += 1
            elif old > 2:
                delta_conf += 1
            
            # Acceptance
            if delta_conf <= 0 or (delta_conf > 0 and rng.random() < math.exp(-delta_conf / T)):
                chosen.append(x)
                chosen_set.add(x)
                counts = new_counts
                current_conflicts += delta_conf
                if current_conflicts == 0 and len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = list(chosen)
                    best_counts = dict(counts)
                    
        elif move_type < 0.7 and len(chosen) > 0:
            # Try SWAP
            idx = rng.randrange(len(chosen))
            removed = chosen[idx]
            candidates = [x for x in range(1, 1001) if x not in chosen_set and x != removed]
            if not candidates:
                continue
            add_candidate = rng.choice(candidates)
            
            # Remove 'removed'
            new_chosen = [x for i, x in enumerate(chosen) if i != idx]
            new_set = set(new_chosen)
            
            # Recompute counts without 'removed'
            new_counts = {}
            for i, a in enumerate(new_chosen):
                for j in range(i, len(new_chosen)):
                    s = a + new_chosen[j]
                    new_counts[s] = new_counts.get(s, 0) + 1
            
            # Try to add add_candidate
            ok = True
            delta_conf = 0
            for y in new_chosen:
                s = add_candidate + y
                old = new_counts.get(s, 0)
                if old >= 2:
                    delta_conf += 1
            s2 = 2 * add_candidate
            old = new_counts.get(s2, 0)
            if old >= 2:
                delta_conf += 1
            
            # Calculate conflicts in new_counts
            new_conflicts = 0
            for s, val in new_counts.items():
                if val > 2:
                    new_conflicts += val - 2
            
            # Acceptance
            if delta_conf <= 0 or (delta_conf > 0 and rng.random() < math.exp(-delta_conf / T)):
                new_chosen.append(add_candidate)
                new_counts[2 * add_candidate] = new_counts.get(2 * add_candidate, 0) + 1
                for y in new_chosen:
                    if y != add_candidate:
                        s = add_candidate + y
                        new_counts[s] = new_counts.get(s, 0) + 1
                
                # Check validity
                if all(c <= 2 for c in new_counts.values()):
                    if len(new_chosen) > best_size:
                        best_size = len(new_chosen)
                        best_chosen = list(new_chosen)
                        best_counts = dict(new_counts)
                    # Accept the swap
                    chosen = new_chosen
                    chosen_set = set(chosen)
                    counts = new_counts
                    current_conflicts = 0
                    break
        else:
            # Try REMOVE (if we have conflicts)
            if len(chosen) > 0 and current_conflicts > 0:
                idx = rng.randrange(len(chosen))
                removed = chosen[idx]
                new_chosen = [x for i, x in enumerate(chosen) if i != idx]
                new_set = set(new_chosen)
                
                # Recompute counts
                new_counts = {}
                for i, a in enumerate(new_chosen):
                    for j in range(i, len(new_chosen)):
                        s = a + new_chosen[j]
                        new_counts[s] = new_counts.get(s, 0) + 1
                
                new_conflicts = get_conflicts(new_counts)
                delta_conf = new_conflicts - current_conflicts
                
                if delta_conf <= 0 or (delta_conf > 0 and rng.random() < math.exp(-delta_conf / T)):
                    chosen = new_chosen
                    chosen_set = set(chosen)
                    counts = new_counts
                    current_conflicts = new_conflicts
        
        T *= cooling
    
    # Final polish: try to add remaining elements greedily
    if time.time() < deadline:
        chosen_set = set(best_chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                ok, new_counts = _try_add(best_chosen, best_counts, x)
                if ok:
                    best_chosen.append(x)
                    chosen_set.add(x)
                    best_counts = new_counts
                    best_size += 1
                    improved = True
                    break
    
    return best_chosen, best_counts, best_size

def construct():
    deadline = time.time() + 105  # 15s safety margin
    
    # Phase 1: Greedy construction
    chosen = _greedy_build()
    counts = _count_sums(chosen)
    best_size = len(chosen)
    best_chosen = list(chosen)
    best_counts = dict(counts)
    
    print(f"Greedy size: {best_size}", file=sys.stderr)
    
    # Phase 2: Simulated annealing with multiple restarts
    if time.time() < deadline:
        for seed in range(15):
            if time.time() >= deadline:
                break
            c, cnt, sz = _local_search(list(best_chosen), dict(best_counts), deadline, seed=seed)
            if sz > best_size:
                best_size = sz
                best_chosen = c
                best_counts = cnt
                print(f"Improved to {best_size} at seed {seed}", file=sys.stderr)
    
    # Phase 3: Try to add remaining elements greedily from the best found
    if time.time() < deadline:
        chosen_set = set(best_chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                ok, new_counts = _try_add(best_chosen, best_counts, x)
                if ok:
                    best_chosen.append(x)
                    chosen_set.add(x)
                    best_counts = new_counts
                    best_size += 1
                    improved = True
                    break
    
    print(f"Final size: {best_size}", file=sys.stderr)
    return best_chosen
