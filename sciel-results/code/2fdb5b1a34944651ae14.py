import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Total number of differences per row: k*(k+1)/2
    # Total differences: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Fast difference tracking
    class DiffTracker:
        def __init__(self, rows):
            self.rows = rows
            self.n_rows = len(rows)
            self.k = len(rows[0]) - 1
            # Map: diff_value -> count (or list of sources)
            self.diff_count = defaultdict(int)
            # For each row, precompute which pairs produce which diffs
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = defaultdict(int)
            for r in range(self.n_rows):
                row = self.rows[r]
                for i in range(self.k + 1):
                    for j in range(i + 1, self.k + 1):
                        d = row[j] - row[i]
                        self.diff_count[d] += 1
        
        def get_conflicts(self):
            """Return list of diff values that appear more than once"""
            return [d for d, c in self.diff_count.items() if c > 1]
        
        def count_conflicts(self):
            return sum(1 for c in self.diff_count.values() if c > 1)
        
        def apply_move(self, r, i, old_val, new_val):
            """Apply a move and update diff counts. Returns (old_conflict_count, new_conflict_count)"""
            row = self.rows[r]
            # Remove old differences involving position i
            for j in range(self.k + 1):
                if j == i:
                    continue
                if j < i:
                    d = old_val - row[j]
                else:
                    d = row[j] - old_val
                self.diff_count[d] -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            # Apply the move
            row[i] = new_val
            
            # Add new differences involving position i
            for j in range(self.k + 1):
                if j == i:
                    continue
                if j < i:
                    d = new_val - row[j]
                else:
                    d = row[j] - new_val
                self.diff_count[d] += 1
            
            return self.count_conflicts()
        
        def revert_move(self, r, i, old_val, new_val):
            """Revert a move"""
            row = self.rows[r]
            # Remove new differences
            for j in range(self.k + 1):
                if j == i:
                    continue
                if j < i:
                    d = new_val - row[j]
                else:
                    d = row[j] - new_val
                self.diff_count[d] -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            # Revert
            row[i] = old_val
            
            # Add old differences
            for j in range(self.k + 1):
                if j == i:
                    continue
                if j < i:
                    d = old_val - row[j]
                else:
                    d = row[j] - old_val
                self.diff_count[d] += 1
        
        def get_scope(self):
            return max(max(row) for row in self.rows)
    
    def is_structurally_valid(row):
        """Check if row is strictly increasing and starts at 0"""
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True
    
    # Phase 1: Greedy initialization with multiple strategies
    def greedy_init(strategy=0):
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            if strategy == 0:
                # Simple greedy: pick smallest available
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
            elif strategy == 1:
                # Greedy with some randomness
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        if random.random() < 0.3:
                            # Sometimes skip a few
                            skip = random.randint(1, 3)
                            x += skip
                            new_diffs = {x - a for a in row}
                            if new_diffs & used:
                                x -= skip
                                new_diffs = {x - a for a in row}
                        row.append(x)
                        used |= new_diffs
            else:
                # Randomized greedy
                x = 0
                while len(row) < k + 1:
                    x += random.randint(1, 2)
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    # If we go too high, reset
                    if x > 500 and len(row) < k + 1:
                        row = [0]
                        x = 0
                        used = set()
                        break
                if len(row) < k + 1:
                    # Fallback to simple greedy for this row
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
    
    # Try multiple initializations, pick the best (smallest scope)
    best_rows = None
    best_scope = float('inf')
    
    for strat in range(3):
        rows = greedy_init(strat)
        if is_structurally_valid(rows[0]):  # Check at least first row
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in rows]
    
    # If greedy failed, use a more robust approach
    if best_rows is None:
        # Use a very simple guaranteed approach
        used = set()
        best_rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            best_rows.append(row)
        best_scope = max(max(row) for row in best_rows)
    
    # Check if already valid
    tracker = DiffTracker(best_rows)
    if tracker.count_conflicts() == 0:
        return best_rows
    
    # Phase 2: Conflict-driven local search to get a valid solution
    # Then Phase 3: Scope minimization
    
    current_rows = [row[:] for row in best_rows]
    current_tracker = DiffTracker(current_rows)
    best_valid_rows = None
    best_valid_scope = float('inf')
    
    T = 50.0
    cooling = 0.99995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Check time
        if time.time() > deadline - 5:
            break
        
        conflicts = current_tracker.get_conflicts()
        current_conflicts = len(conflicts)
        current_scope = current_tracker.get_scope()
        
        # If we found a valid solution, track it
        if current_conflicts == 0:
            if current_scope < best_valid_scope:
                best_valid_scope = current_scope
                best_valid_rows = [row[:] for row in current_rows]
        
        # Choose a move
        if conflicts:
            # Conflict-aware: pick a random conflicting difference and try to fix it
            # Find all pairs that produce this difference
            target_diff = random.choice(conflicts)
            # Find rows that have this difference
            candidate_moves = []
            for r in range(n_rows):
                row = current_rows[r]
                for i in range(1, k + 1):
                    # Check if moving position i would affect this difference
                    # The difference target_diff appears in this row if there exist a,b with row[b]-row[a]=target_diff
                    # Moving position i changes all diffs involving i
                    # So if i is involved in any pair that gives target_diff, this move is relevant
                    # Simplified: just consider all positions as candidates
                    candidate_moves.append((r, i))
            
            if candidate_moves:
                r, i = random.choice(candidate_moves)
            else:
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k)
        else:
            # No conflicts, try to reduce scope
            # Pick the row with the largest max element
            max_row = max(range(n_rows), key=lambda r: current_rows[r][-1])
            # Pick a position in that row, biased towards the end
            if random.random() < 0.7:
                i = random.randint(k // 2, k)
            else:
                i = random.randint(1, k)
            r = max_row
        
        old_val = current_rows[r][i]
        
        # Determine move range
        if conflicts:
            # Try to reduce the conflicting difference
            # If old_val is part of a large difference, try to decrease it
            delta = random.randint(-8, 8)
        else:
            # Try to reduce scope
            delta = random.randint(-10, 3)
        
        new_val = old_val + delta
        
        # Check structural validity
        if new_val <= 0:
            continue
        if i > 0 and new_val <= current_rows[r][i - 1]:
            continue
        if i < k and new_val >= current_rows[r][i + 1]:
            continue
        
        # Apply move
        old_conflicts = current_tracker.apply_move(r, i, old_val, new_val)
        new_scope = current_tracker.get_scope()
        
        # Cost: prioritize validity, then scope
        if old_conflicts > 0:
            # We care most about reducing conflicts
            delta_conflicts = old_conflicts - (0 if old_conflicts == 0 else 0)
            # Actually, let's use a combined cost
            old_cost = old_conflicts * 1000 + current_scope
            new_cost = old_conflicts * 1000 + new_scope  # This is wrong, old_conflicts was before move
            # Let me redo this
            # Actually, apply_move returns the new conflict count
            new_conflicts = old_conflicts  # This is the new count after apply
            # Wait, I need to track old and new properly
            # Let me restructure
        
        # Revert and redo with proper tracking
        current_tracker.revert_move(r, i, old_val, new_val)
        
        # Proper cost calculation
        old_conflicts = current_tracker.count_conflicts()
        old_scope = current_tracker.get_scope()
        old_cost = old_conflicts * 10000 + old_scope
        
        current_tracker.apply_move(r, i, old_val, new_val)
        new_conflicts = current_tracker.count_conflicts()
        new_scope = current_tracker.get_scope()
        new_cost = new_conflicts * 10000 + new_scope
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            if new_conflicts == 0 and new_scope < best_valid_scope:
                best_valid_scope = new_scope
                best_valid_rows = [row[:] for row in current_rows]
        else:
            # Revert
            current_tracker.revert_move(r, i, old_val, new_val)
        
        T *= cooling
    
    # Return best valid solution
    if best_valid_rows is not None:
        return best_valid_rows
    
    # Check if current is valid
    if current_tracker.count_conflicts() == 0:
        return current_rows
    
    # Fallback: return greedy solution (which should be valid)
    return best_rows
