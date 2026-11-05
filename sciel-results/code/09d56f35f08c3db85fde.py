"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses Variable Neighborhood Search (VNS) on a difference-permutation matrix
    representation, with CSP fallback for tight bounds.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Helper: reconstruct row from difference set
    # Given a set of differences that form a valid difference set for k+1 points,
    # reconstruct the row
    def diff_set_to_row(diff_set, k):
        """
        Given a set of positive differences that should be all pairwise differences
        of k+1 points starting at 0, reconstruct the row.
        Returns None if the diff_set doesn't correspond to a valid row.
        """
        if len(diff_set) != k * (k + 1) // 2:
            return None
        
        # The largest difference must be the last element
        max_diff = max(diff_set)
        
        # Try to reconstruct: the row elements must be such that all pairwise diffs are in diff_set
        # Use a greedy approach: 0 is first, then find next element
        row = [0]
        remaining_diffs = set(diff_set)
        
        for _ in range(k):
            # Find the smallest positive difference not yet "explained"
            # that can be the next element
            found_next = False
            for candidate in sorted(remaining_diffs):
                # Check if candidate can be the next element
                # All differences from candidate to existing row elements must be in diff_set
                valid = True
                new_diffs = set()
                for a in row:
                    d = candidate - a
                    if d not in remaining_diffs and d not in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Verify that all new diffs are consistent
                    row.append(candidate)
                    remaining_diffs -= new_diffs
                    found_next = True
                    break
            
            if not found_next:
                return None
        
        # Verify all differences are accounted for
        if remaining_diffs:
            return None
        
        # Final verification
        actual_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                actual_diffs.add(row[j] - row[i])
        
        if actual_diffs != diff_set:
            return None
        
        return row
    
    # VNS solver on difference-permutation matrix
    def solve_vns(n_rows, k, max_entry, timeout, seed=0):
        """
        Variable Neighborhood Search on difference assignments.
        Each row is defined by its set of k*(k+1)/2 differences.
        The global assignment is a partition of {1..max_entry} into n_rows subsets,
        each of size k*(k+1)/2, such that each subset corresponds to a valid row.
        
        Neighborhood moves:
        1. Transposition: swap a difference between two rows
        2. Switch: move a difference from one row to another and vice versa
        3. Perturbation: random reassignment of several differences
        """
        import random as rnd
        rng = rnd.Random(seed)
        deadline = time.time() + timeout
        
        diffs_per_row = k * (k + 1) // 2
        total_slots = n_rows * diffs_per_row
        
        if max_entry < total_slots:
            return None
        
        # Initialize: try to build a valid assignment
        # Start with greedy-like initialization
        def initialize():
            """Create initial difference assignment"""
            used = set()
            row_diffs = []
            
            for _ in range(n_rows):
                row_diff = set()
                # Greedily pick differences
                for d in range(1, max_entry + 1):
                    if d in used or d in row_diff:
                        continue
                    if len(row_diff) == diffs_per_row:
                        break
                    row_diff.add(d)
                
                if len(row_diff) < diffs_per_row:
                    return None
                
                # Verify this diff set corresponds to a valid row
                row = diff_set_to_row(row_diff, k)
                if row is None:
                    return None
                
                used |= row_diff
                row_diffs.append(row_diff)
            
            return row_diffs
        
        # Try to initialize
        row_diffs = initialize()
        if row_diffs is None:
            # Try random initialization
            for attempt in range(50):
                if time.time() > deadline:
                    return None
                all_diffs = list(range(1, max_entry + 1))
                rng.shuffle(all_diffs)
                
                row_diffs = []
                valid = True
                for i in range(n_rows):
                    row_diff = set(all_diffs[i * diffs_per_row : (i + 1) * diffs_per_row])
                    row = diff_set_to_row(row_diff, k)
                    if row is None:
                        valid = False
                        break
                    row_diffs.append(row_diff)
                
                if valid:
                    break
            else:
                return None
        
        # Track best solution
        best_rows = None
        best_max = None
        
        def get_rows_from_diffs(row_diffs):
            """Convert difference sets to rows"""
            rows = []
            for rd in row_diffs:
                row = diff_set_to_row(rd, k)
                if row is None:
                    return None
                rows.append(row)
            return rows
        
        # Check if current solution is valid and get its max
        def evaluate(row_diffs):
            rows = get_rows_from_diffs(row_diffs)
            if rows is None:
                return None, None
            mx = max(max(r) for r in rows)
            return rows, mx
        
        # Local search: try to improve by swapping differences between rows
        def local_search(row_diffs, deadline):
            """
            Perform local search by trying all possible swaps between rows.
            Returns improved row_diffs or None.
            """
            improved = True
            iterations = 0
            max_iterations = 10000
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                if time.time() > deadline:
                    break
                
                # Try all pairs of rows and all possible swaps
                for i in range(n_rows):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n_rows):
                        if time.time() > deadline:
                            break
                        
                        # Try swapping one diff from row i with one from row j
                        for d1 in list(row_diffs[i]):
                            for d2 in list(row_diffs[j]):
                                if d1 == d2:
                                    continue
                                
                                # Remove d1 from i, d2 from j, add d2 to i, d1 to j
                                row_diffs[i].discard(d1)
                                row_diffs[j].discard(d2)
                                row_diffs[i].add(d2)
                                row_diffs[j].add(d1)
                                
                                # Check if both rows are still valid
                                row_i = diff_set_to_row(row_diffs[i], k)
                                row_j = diff_set_to_row(row_diffs[j], k)
                                
                                if row_i is not None and row_j is not None:
                                    # Check global validity (no duplicate diffs)
                                    all_diffs = set()
                                    valid = True
                                    for rd in row_diffs:
                                        if all_diffs & rd:
                                            valid = False
                                            break
                                        all_diffs |= rd
                                    
                                    if valid:
                                        # Check if this improves the max
                                        rows = get_rows_from_diffs(row_diffs)
                                        if rows is not None:
                                            mx = max(max(r) for r in rows)
                                            if best_rows is None or mx < best_max:
                                                best_rows = [list(r) for r in rows]
                                                best_max = mx
                                            improved = True
                                else:
                                    # Revert
                                    row_diffs[i].discard(d2)
                                    row_diffs[j].discard(d1)
                                    row_diffs[i].add(d1)
                                    row_diffs[j].add(d2)
                                    # Actually we already swapped, so revert properly
                                    row_diffs[i].discard(d2)
                                    row_diffs[j].discard(d1)
                                    row_diffs[i].add(d1)
                                    row_diffs[j].add(d2)
                        
                        if improved:
                            break
                    if improved:
                        break
            
            return row_diffs
        
        # VNS: alternate between local search and perturbation
        current_row_diffs = [set(rd) for rd in row_diffs]
        
        # Initial evaluation
        rows, mx = evaluate(current_row_diffs)
        if rows is not None:
            best_rows = [list(r) for r in rows]
            best_max = mx
        
        # Main VNS loop
        while time.time() < deadline:
            # Local search
            current_row_diffs = local_search(current_row_diffs, deadline)
            
            if time.time() > deadline:
                break
            
            # Perturbation: randomly swap several differences
            n_perturb = rng.randint(2, min(10, n_rows * 2))
            for _ in range(n_perturb):
                if time.time() > deadline:
                    break
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(0, n_rows - 1)
                if i == j:
                    continue
                d1 = rng.choice(list(current_row_diffs[i]))
                d2 = rng.choice(list(current_row_diffs[j]))
                if d1 == d2:
                    continue
                current_row_diffs[i].discard(d1)
                current_row_diffs[j].discard(d2)
                current_row_diffs[i].add(d2)
                current_row_diffs[j].add(d1)
            
            # Check if perturbed solution is valid
            rows, mx = evaluate(current_row_diffs)
            if rows is not None:
                if best_rows is None or mx < best_max:
                    best_rows = [list(r) for r in rows]
                    best_max = mx
            
            # If perturbation broke validity, try to repair
            if rows is None:
                # Try to fix by reverting last perturbation
                # For simplicity, just restart local search from best
                current_row_diffs = [set(r) for r in best_rows] if best_rows else None
                if current_row_diffs is None:
                    break
                # Reconstruct diff sets from rows
                new_diffs = []
                for row in current_row_diffs:
                    rd = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            rd.add(row[j] - row[i])
                    new_diffs.append(rd)
                current_row_diffs = new_diffs
        
        if best_rows is None:
            return None
        
        return best_rows
    
    # CSP solver (kept as fallback for tight bounds)
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        import random as rnd
        rng = rnd.Random(seed)
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        def available_diffs(used_bits):
            return max_entry - used_bits.bit_count()

        def build_row(current_row, used_bits, deadline, rng):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Forward Checking: ensure enough differences are available
            needed_diffs = remaining * len(current_row) + (remaining * (remaining - 1)) // 2
            if available_diffs(used_bits) < needed_diffs:
                return False

            candidates = []
            for x in range(last + 1, max_entry + 1):
                if max_entry - x < remaining - 1:
                    break
                
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Heuristic: Prefer values that use fewer small differences
                small_mask = new_diff_bits & 0x7FFF # 15 bits
                score = small_mask.bit_count()
                
                # Add noise to randomize order
                noise = rng.random() * 0.5
                candidates.append((score + noise, x, new_diff_bits))
            
            if not candidates:
                return False
            
            candidates.sort()
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline, rng):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline, rng):
                rows.append(current_row)
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows
    
    # Try to find better solution than greedy using both VNS and CSP
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: Try VNS first for larger ranges, CSP for tight bounds
    max_search_range = min(greedy_max, lb + 50)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        
        if remaining_time < 1.0:
            break
        
        distance_from_lb = max_entry - lb
        
        # Allocate budget based on distance from lower bound
        if distance_from_lb <= 3:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 10:
            budget = min(8.0, remaining_time)
        else:
            budget = min(4.0, remaining_time)
        
        if budget < 0.5:
            break
        
        found = False
        
        # Try VNS first (better for larger search spaces)
        if budget > 2.0:
            vns_budget = budget * 0.6
            num_vns_seeds = max(1, int(vns_budget / 3.0))
            for seed in range(num_vns_seeds):
                if time.time() > start_time + time_limit - 5:
                    break
                sub_budget = vns_budget / (num_vns_seeds - seed)
                if sub_budget < 0.5:
                    break
                
                result = solve_vns(n_rows, k, max_entry, sub_budget, seed=seed)
                if result is not None:
                    best_rows = result
                    best_max = max_entry
                    found = True
                    break
        
        # Try CSP if VNS didn't find a solution or for very tight bounds
        if not found and distance_from_lb <= 5:
            csp_budget = budget * 0.4
            num_csp_seeds = max(1, int(csp_budget / 2.0))
            for seed in range(num_csp_seeds):
                if time.time() > start_time + time_limit - 5:
                    break
                sub_budget = csp_budget / (num_csp_seeds - seed)
                if sub_budget < 0.5:
                    break
                
                result = solve_csp(n_rows, k, max_entry, sub_budget, seed=seed)
                if result is not None:
                    best_rows = result
                    best_max = max_entry
                    found = True
                    break
        
        if found:
            break
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                budget = min(remaining_time, 15.0)
                # Try VNS with more seeds
                num_seeds = max(2, int(budget / 3.0))
                for seed in range(num_seeds):
                    if time.time() > start_time + time_limit - 5:
                        break
                    sub_budget = budget / (num_seeds - seed)
                    if sub_budget < 0.5:
                        break
                    result = solve_vns(n_rows, k, mid, sub_budget, seed=seed)
                    if result is not None:
                        best_rows = result
                        best_max = mid
                        break
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
