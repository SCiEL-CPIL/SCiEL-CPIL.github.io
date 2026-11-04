"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within each row are distinct across all rows.
    
    Uses a randomized greedy approach with backtracking-like restarts
    and multiple strategies to find good solutions within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # For the specific case (4, 7): k=7 means each row has 8 elements
    # Number of differences per row = k*(k+1)/2 = 7*8/2 = 28
    # Total differences = 4*28 = 112
    
    def generate_greedy_row(used_diffs, k, rng, start_val=0):
        """Generate a single row using greedy approach."""
        row = [0]
        current = 0
        for _ in range(k):
            # Try values starting from current+1
            found = False
            for x in range(current + 1, 300):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    current = x
                    found = True
                    break
            if not found:
                return None
        return row
    
    def generate_row_with_priority(used_diffs, k, rng, strategy="greedy"):
        """
        Generate a row with different strategies for choosing next value.
        """
        row = [0]
        current = 0
        
        for _ in range(k):
            found = False
            
            if strategy == "greedy":
                # Simple greedy: try smallest available
                for x in range(current + 1, 300):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        current = x
                        found = True
                        break
                        
            elif strategy == "randomized":
                # Randomized: try a random sample of candidates
                candidates = list(range(current + 1, min(current + 50, 300)))
                rng.shuffle(candidates)
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        current = x
                        found = True
                        break
                        
            elif strategy == "spread":
                # Try to spread out the values more
                for x in range(current + 1, 300):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        # Prefer larger gaps sometimes
                        if rng.random() < 0.3 and x - current > 2:
                            row.append(x)
                            current = x
                            found = True
                            break
                        elif x - current == 1:
                            row.append(x)
                            current = x
                            found = True
                            break
                            
            if not found:
                return None
        return row
    
    def score_solution(rows):
        """Return the maximum entry in the solution."""
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def validate_solution(rows, n_rows, k):
        """Validate that all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    best_solution = None
    best_score = float('inf')
    
    # Strategy 1: Pure greedy (deterministic)
    def try_greedy():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = generate_greedy_row(used_diffs, k, random.Random(42))
            if row is None:
                return None
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        return rows
    
    # Try initial greedy
    sol = try_greedy()
    if sol and validate_solution(sol, n_rows, k):
        best_solution = sol
        best_score = score_solution(sol)
    
    # Multiple randomized restarts with different strategies
    strategies = ["greedy", "randomized", "spread"]
    
    while time.time() < deadline:
        rng = random.Random()
        strategy = rng.choice(strategies)
        
        used_diffs = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = generate_row_with_priority(used_diffs, k, rng, strategy)
            if row is None:
                success = False
                break
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        if success and validate_solution(rows, n_rows, k):
            sc = score_solution(rows)
            if sc < best_score:
                best_score = sc
                best_solution = rows
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    if best_solution is None:
        # Fallback: return a simple valid solution
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_diffs):
                    row.append(x)
                    used_diffs |= new
            rows.append(row)
        return rows
    
    return best_solution
# EVOLVE-BLOCK-END
