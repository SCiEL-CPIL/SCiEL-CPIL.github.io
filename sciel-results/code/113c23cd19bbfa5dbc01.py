import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 135.0  # 15s safety margin for 150s limit
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Get initial solution via greedy
    best_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(best_rows)
    
    # Lower bound for M
    M_lower = max(total_diffs_needed, k * (k + 1) // 2 + k, 20)
    
    # Try to find a solution with smaller M using randomized search
    for M in range(M_lower, best_M):
        if time.time() > deadline - 10:
            break
        
        # Try to find a packing with max entry <= M
        rows = _solve_with_sa(n_rows, k, M, deadline, best_rows, best_M)
        
        if rows is not None and _is_valid(rows, k):
            # Found a valid solution
            new_M = _max_entry(rows)
            if new_M < best_M:
                best_rows = rows
                best_M = new_M
                # Since we're iterating M upward, if we found a valid solution,
                # it might not have max entry exactly M, but <= M
                # Continue trying smaller M? No, we're going up.
                # Actually, we should break since we found the minimal M that works?
                # Not necessarily - the search might not be exhaustive.
                # But for now, let's continue to see if we can do better with more search time
                # Actually, let's stop since we found a solution for this M
                break
    
    return best_rows

def _solve_with_sa(n_rows, k, M, deadline, initial_rows, initial_M):
    """
    Use simulated annealing to find a valid packing with max entry <= M.
    """
    # Precompute all valid single rows (Sidon-like sets) up to M
    # A row is valid if all pairwise differences are distinct
    valid_rows = _generate_valid_rows(k, M)
    
    if len(valid_rows) < n_rows:
        return None
    
    # Simulated annealing:
    # State: selection of n_rows rows (indices into valid_rows)
    # Energy: number of conflicts + penalty for max entry > M
    
    random.seed(int(time.time() * 1000) % (2**31))
    
    # Start with a random valid selection or greedy
    current_selection = _greedy_select(valid_rows, n_rows, k, M)
    if current_selection is None:
        # Try random initialization
        current_selection = random.sample(range(len(valid_rows)), n_rows)
    
    # Compute initial energy
    current_energy = _compute_energy(current_selection, valid_rows, k, M)
    
    best_selection = current_selection[:]
    best_energy = current_energy
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.999
    
    steps = 0
    while time.time() < deadline - 15 and T > 0.1:
        # Generate neighbor: change one row in the selection
        i = random.randint(0, n_rows - 1)
        old_row_idx = current_selection[i]
        
        # Try a random new row
        new_row_idx = random.randint(0, len(valid_rows) - 1)
        if new_row_idx == old_row_idx:
            continue
        # Make sure it's not already in the selection
        if new_row_idx in current_selection:
            continue
        
        current_selection[i] = new_row_idx
        new_energy = _compute_energy(current_selection, valid_rows, k, M)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_selection = current_selection[:]
        else:
            current_selection[i] = old_row_idx
        
        T *= cooling_rate
        steps += 1
        
        if steps % 10000 == 0 and best_energy == 0:
            break
    
    if best_energy == 0:
        # Found a conflict-free selection
        rows = [list(valid_rows[i]) for i in best_selection]
        # Verify max entry <= M
        if max(max(row) for row in rows) <= M:
            return rows
    
    return None

def _generate_valid_rows(k, M):
    """
    Generate all valid rows (k+1 integers from {0,...,M} starting at 0)
    where all pairwise differences are distinct.
    """
    valid_rows = []
    
    # We need to choose k numbers from 1 to M such that
    # all pairwise differences among {0, a1, a2, ..., ak} are distinct
    # This is equivalent to: the set {0, a1, ..., ak} is a Sidon set (B2 set)
    
    # For efficiency, use recursive generation with pruning
    def gen(start, chosen, current_diffs):
        if len(chosen) == k:
            valid_rows.append([0] + chosen[:])
            return
        
        # Pruning: check if current partial set already has conflicts
        # (already ensured by construction)
        
        for v in range(start, M + 1):
            # Check if v creates any new difference conflict
            ok = True
            new_diffs = set()
            for a in chosen:
                d = v - a
                if d in current_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            
            if not ok:
                continue
            
            # Also check differences among new_diffs themselves? No, we only need
            # that all pairwise differences are distinct globally in the final set.
            # But we're building incrementally, so we need to ensure that the new
            # differences don't conflict with each other or with existing ones.
            
            # Check that new_diffs are all distinct from each other
            if len(new_diffs) != len(chosen):
                continue
            
            chosen.append(v)
            current_diffs |= new_diffs
            gen(v + 1, chosen, current_diffs)
            chosen.pop()
    
    gen(1, [], set())
    
    # Remove duplicates (there shouldn't be any since we generate in sorted order)
    seen = set()
    unique_rows = []
    for row in valid_rows:
        t = tuple(row)
        if t not in seen:
            seen.add(t)
            unique_rows.append(row)
    
    return unique_rows

def _compute_energy(selection, valid_rows, k, M):
    """
    Compute energy: number of conflicting difference pairs + penalty for max > M.
    """
    used_diffs = set()
    conflicts = 0
    max_entry = 0
    
    for row_idx in selection:
        row = valid_rows[row_idx]
        max_entry = max(max_entry, row[-1])
        
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs:
                    conflicts += 1
                if d in row_diffs:
                    conflicts += 1
                row_diffs.add(d)
        
        used_diffs |= row_diffs
    
    # Penalty for exceeding M
    penalty = 0
    if max_entry > M:
        penalty = (max_entry - M) * 100
    
    return conflicts + penalty

def _greedy_select(valid_rows, n_rows, k, M):
    """
    Greedily select n_rows rows from valid_rows to minimize conflicts.
    """
    if len(valid_rows) < n_rows:
        return None
    
    selection = []
    used_diffs = set()
    
    # Sort valid_rows by max entry (prefer compact rows)
    indexed_rows = [(row[-1], i, row) for i, row in enumerate(valid_rows)]
    indexed_rows.sort()
    
    for _, idx, row in indexed_rows:
        if len(selection) >= n_rows:
            break
        
        row_diffs = set()
        ok = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs or d in row_diffs:
                    ok = False
                    break
                row_diffs.add(d)
            if not ok:
                break
        
        if ok:
            selection.append(idx)
            used_diffs |= row_diffs
    
    if len(selection) == n_rows:
        return selection
    
    return None

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
