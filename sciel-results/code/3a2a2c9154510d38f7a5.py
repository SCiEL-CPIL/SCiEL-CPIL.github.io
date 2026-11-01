"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of length k+1,
    starting at 0, strictly increasing, such that all positive differences
    within each row are distinct across all rows.
    
    Strategy: Penalty-based Simulated Annealing with incremental difference tracking.
    """
    
    # Precompute indices for difference calculation
    # For a row of length k+1, differences are between i and j where i < j
    diff_indices = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]
    num_diffs_per_row = len(diff_indices)
    total_diffs = n_rows * num_diffs_per_row
    
    def greedy_construct(seed=0):
        """Build rows greedily, one by one, using smallest available marks."""
        random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def get_scope(rows):
        """Return the maximum entry across all rows."""
        return max(max(row) for row in rows)

    def calculate_initial_diffs(rows):
        """
        Initialize the difference count array.
        Returns a dictionary mapping diff_value -> count.
        """
        diff_counts = {}
        for row in rows:
            for i, j in diff_indices:
                d = row[j] - row[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
        return diff_counts

    def get_collisions(diff_counts):
        """Return the number of differences that appear more than once."""
        collisions = 0
        for count in diff_counts.values():
            if count > 1:
                collisions += count - 1
        return collisions

    def simulated_annealing(initial_rows, time_limit=110):
        """
        Perform Simulated Annealing to minimize scope while ensuring validity.
        Uses a penalty for collisions to allow temporary invalid states.
        """
        import math
        
        # Deep copy
        current_rows = [r[:] for r in initial_rows]
        diff_counts = calculate_initial_diffs(current_rows)
        current_collisions = get_collisions(diff_counts)
        current_scope = get_scope(current_rows)
        
        # Cost function: Scope is primary, collisions are penalized heavily
        # We want to reach 0 collisions.
        # Cost = Scope + Penalty * Collisions
        # Since Scope is ~100-200 and Collisions can be large, we need a good penalty.
        # If we allow collisions, we can explore.
        
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_collisions = current_collisions
        
        # SA Parameters
        T = 50.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < time_limit:
            # Pick a random row and position
            row_idx = random.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            pos = random.randint(1, k) # 1 to k (index 0 is always 0)
            
            old_val = row[pos]
            
            # Determine new value
            # We want to explore the space.
            # Bias towards reducing the max element (scope).
            # If this element is the max, try smaller.
            # If not, try nearby.
            
            # Get bounds
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos + 1 <= k else current_scope + 50
            
            if max_val < min_val:
                continue
                
            # Random move
            # 80% chance to stay in a local window, 20% chance to jump
            if random.random() < 0.8:
                window = max(5, current_scope // 10)
                new_val = random.randint(max(min_val, old_val - window), min(max_val, old_val + window))
            else:
                new_val = random.randint(min_val, max_val)
                
            if new_val == old_val:
                continue
                
            # Calculate delta in cost
            # 1. Update diff counts for old_val
            old_diffs = []
            for i, j in diff_indices:
                if i == pos:
                    d = row[j] - old_val
                    old_diffs.append(d)
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                elif j == pos:
                    d = old_val - row[i]
                    old_diffs.append(d)
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
            
            # 2. Update diff counts for new_val
            new_diffs = []
            for i, j in diff_indices:
                if i == pos:
                    d = row[j] - new_val
                    new_diffs.append(d)
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                elif j == pos:
                    d = new_val - row[i]
                    new_diffs.append(d)
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            
            # Check for internal collisions in the new row?
            # The diff_counts tracks global collisions.
            # We need to ensure the row itself is strictly increasing.
            # new_val > row[pos-1] and new_val < row[pos+1] (if exists) is handled by bounds.
            
            # Calculate new collisions
            # We can track collisions incrementally, but recalculating from dict is O(num_unique_diffs)
            # num_unique_diffs is small (< 200).
            new_collisions = 0
            for count in diff_counts.values():
                if count > 1:
                    new_collisions += count - 1
            
            delta_collisions = new_collisions - current_collisions
            
            # Calculate new scope
            new_scope = current_scope
            if new_val > current_scope:
                new_scope = new_val
            elif old_val == current_scope:
                # If we removed the max, we need to find the new max
                # This is O(n_rows * k). Since k is small, it's fine.
                # Optimization: only recalc if old_val was the unique max
                is_unique_max = True
                for r in current_rows:
                    if r[-1] > old_val: # Actually max can be anywhere? No, rows are increasing, so max is last element.
                        is_unique_max = False
                        break
                    if max(r) > old_val:
                        is_unique_max = False
                        break
                if is_unique_max:
                    new_scope = get_scope(current_rows) # Temporary, we'll revert if rejected, but need accurate delta
                else:
                    new_scope = current_scope
            
            # Cost Delta
            # We prioritize validity.
            # If collisions increase, it's bad.
            # If scope increases, it's bad.
            
            # Simple cost: Scope * 10 + Collisions * 100
            # This makes collisions very expensive, but allows some if scope drops significantly.
            old_cost = current_scope * 10 + current_collisions * 100
            new_cost = new_scope * 10 + new_collisions * 100
            delta_cost = new_cost - old_cost
            
            # Acceptance
            accepted = False
            if delta_cost < 0:
                accepted = True
            else:
                if T > 0:
                    prob = math.exp(-delta_cost / T)
                    if random.random() < prob:
                        accepted = True
            
            if accepted:
                current_rows[row_idx][pos] = new_val
                current_collisions = new_collisions
                current_scope = new_scope
                
                # Update best if valid
                if current_collisions == 0 and current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                
            else:
                # Revert diff counts
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
            
            iteration += 1
            if iteration % 10000 == 0:
                # Optional: restart or cool faster if stuck?
                pass

        return best_rows, best_scope

    # Run multiple SA instances from different greedy starts
    best_overall_rows = None
    best_overall_scope = float('inf')
    
    # Time budget for each SA run
    num_restarts = 5
    time_per_run = 20 # seconds
    
    for i in range(num_restarts):
        rows = greedy_construct(seed=i)
        sa_rows, sa_scope = simulated_annealing(rows, time_limit=time_per_run)
        
        if sa_scope < best_overall_scope:
            best_overall_scope = sa_scope
            best_overall_rows = sa_rows
            
        # Check time
        if time.time() - (time.time() - 110) > 100: # Rough check
             break

    if best_overall_rows is None:
        # Fallback
        best_overall_rows = greedy_construct(0)
        
    return best_overall_rows
# EVOLVE-BLOCK-END
