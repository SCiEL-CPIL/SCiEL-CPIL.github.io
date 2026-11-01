import time
import heapq
from collections import defaultdict

def construct(n_rows, k):
    deadline = time.time() + 130  # safety margin before 150s kill
    total_marks = k + 1  # each row needs k+1 marks (including 0)
    
    # Greedy fallback (same as seed) to always have a valid answer
    def greedy_fallback():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < total_marks:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Get a baseline solution first
    baseline = greedy_fallback()
    best_max = max(max(r) for r in baseline)
    
    # Beam search
    # State: (rows, used_diffs) where rows is list of partial rows
    # We track for each state: the list of rows, the set of used differences,
    # and a heuristic score for pruning
    
    def get_valid_next(row, used):
        """Get sorted list of valid next values for a row."""
        # The next value must be > row[-1]
        start = row[-1] + 1
        # Upper bound: we don't want to go too far
        # A reasonable upper bound: current best_max + some slack
        # But we want to find solutions with max < best_max
        upper = best_max + 10
        valid = []
        x = start
        while x <= upper:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                valid.append(x)
            x += 1
        return valid
    
    def state_score(rows, used):
        """Heuristic: prefer states with smaller max entry and fewer 'wasted' small differences."""
        # Score = max entry so far + penalty for having used many small differences
        # Lower is better
        max_entry = max(r[-1] for r in rows)
        # Count how many differences are used (more used = less room)
        # But we want to penalize using small differences early
        small_diffs = sum(1 for d in used if d <= 20)
        return max_entry * 10 + small_diffs
    
    # Initialize beam: one state with all rows having just [0]
    # Actually, we need to build all rows simultaneously
    # State: list of rows (each a list), used set of differences
    
    # Start: each row has [0]
    init_rows = [[0] for _ in range(n_rows)]
    init_used = set()
    
    # Beam: list of (score, rows, used, step_count)
    # We'll use a list and sort by score each iteration
    
    beam = [(0, init_rows, init_used, 0)]
    
    # We need to track which row to extend next for each state
    # The "most constrained" row is the one with fewest valid next values
    
    iteration = 0
    max_beam_width = 500
    
    while time.time() < deadline:
        iteration += 1
        new_beam = []
        
        for score, rows, used, step_count in beam:
            if time.time() > deadline - 10:
                break
            
            # Check if all rows are complete
            if all(len(r) == total_marks for r in rows):
                max_entry = max(max(r) for r in rows)
                if max_entry < best_max:
                    best_max = max_entry
                    baseline = [list(r) for r in rows]
                continue
            
            # Find the most constrained incomplete row
            # A row is incomplete if len < total_marks
            incomplete_rows = [i for i, r in enumerate(rows) if len(r) < total_marks]
            if not incomplete_rows:
                continue
            
            # For each incomplete row, count valid next values
            row_info = []
            for i in incomplete_rows:
                valid = get_valid_next(rows[i], used)
                row_info.append((len(valid), i, valid))
            
            # Sort by number of valid next values (most constrained first)
            row_info.sort(key=lambda x: x[0])
            
            # If any row has no valid next values, this state is dead
            if row_info[0][0] == 0:
                continue
            
            # Branch on the most constrained row
            # But to keep beam manageable, we branch on a limited number of choices
            # for the most constrained row
            _, target_idx, valid_choices = row_info[0]
            
            # Limit branching: take top few choices (smallest values first)
            # But also consider: we want to explore, so take a few
            branch_limit = min(len(valid_choices), 5)
            
            for x in valid_choices[:branch_limit]:
                new_rows = [list(r) for r in rows]
                new_rows[target_idx].append(x)
                new_diffs = {x - a for a in new_rows[target_idx]}
                new_used = used | new_diffs
                new_score = state_score(new_rows, new_used)
                new_beam.append((new_score, new_rows, new_used, step_count + 1))
        
        if not new_beam:
            break
        
        # Keep top beam_width states by score
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:max_beam_width]
        
        # If beam is empty or we've been iterating too long without progress, break
        if len(beam) < 10:
            # Beam is shrinking, might be stuck
            if iteration > 100:
                break
    
    return baseline

# EVOLVE-BLOCK-END
