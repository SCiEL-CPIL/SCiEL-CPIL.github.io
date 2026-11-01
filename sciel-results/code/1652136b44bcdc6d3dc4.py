"""Solver using local search on greedy solution."""
import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 140.0
    num_elements = k + 1
    num_diffs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lb = total_diffs
    
    # Get greedy baseline
    best_solution = _greedy(n_rows, k)
    best_max = max(max(row) for row in best_solution)
    
    # Local search to improve
    current = [list(row) for row in best_solution]
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(trial * 104729 + 1)
        
        # Restart from best or perturbed
        if trial == 0:
            current = [list(row) for row in best_solution]
        else:
            current = _perturb(best_solution, rng)
        
        # Local search
        for step in range(10000):
            if time.time() > deadline:
                break
            
            # Try to improve
            improved = False
            for _ in range(10):  # Try multiple random moves per step
                move = _random_move(current, rng)
                if move is None:
                    continue
                
                new_solution = move[0]
                new_max = max(max(row) for row in new_solution)
                
                # Check validity
                if not _is_valid(new_solution):
                    continue
                
                if new_max < best_max:
                    best_max = new_max
                    best_solution = [list(row) for row in new_solution]
                    current = new_solution
                    improved = True
                    break
                
                # Accept if same max but different structure (for diversity)
                if new_max == best_max and rng.random() < 0.1:
                    current = new_solution
                    improved = True
                    break
            
            if not improved:
                break
        
        if best_max == lb:
            break
    
    return best_solution


def _greedy(n_rows, k):
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


def _is_valid(solution):
    """Check if solution is valid (all differences distinct)."""
    seen = set()
    for row in solution:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _perturb(solution, rng):
    """Create a perturbed version."""
    num_rows = len(solution)
    num_elements = len(solution[0])
    
    result = [list(row) for row in solution]
    
    # Swap some rows
    if rng.random() < 0.5:
        i, j = rng.sample(range(num_rows), 2)
        result[i], result[j] = result[j], result[i]
    
    # Slightly adjust some values
    for _ in range(rng.randint(1, 3)):
        row_idx = rng.randint(0, num_rows - 1)
        pos = rng.randint(1, num_elements - 1)
        delta = rng.randint(-5, 5)
        new_val = result[row_idx][pos] + delta
        
        # Check bounds
        if pos > 0 and new_val <= result[row_idx][pos - 1]:
            continue
        if pos < num_elements - 1 and new_val >= result[row_idx][pos + 1]:
            continue
        if new_val < 1:
            continue
        
        result[row_idx][pos] = new_val
    
    return result


def _random_move(solution, rng):
    """Make a random move."""
    num_rows = len(solution)
    num_elements = len(solution[0])
    
    # Type 1: Adjust one element
    if rng.random() < 0.7:
        row_idx = rng.randint(0, num_rows - 1)
        pos = rng.randint(1, num_elements - 1)
        delta = rng.randint(-10, 10)
        new_val = solution[row_idx][pos] + delta
        
        if pos > 0 and new_val <= solution[row_idx][pos - 1]:
            return None
        if pos < num_elements - 1 and new_val >= solution[row_idx][pos + 1]:
            return None
        if new_val < 1:
            return None
        
        new_solution = [list(row) for row in solution]
        new_solution[row_idx][pos] = new_val
        return (new_solution, None)
    
    # Type 2: Swap two elements in different rows
    else:
        row1, row2 = rng.sample(range(num_rows), 2)
        pos1 = rng.randint(1, num_elements - 1)
        pos2 = rng.randint(1, num_elements - 1)
        
        val1 = solution[row1][pos1]
        val2 = solution[row2][pos2]
        
        # Check if swap maintains ordering
        new_row1 = solution[row1][:]
        new_row2 = solution[row2][:]
        new_row1[pos1] = val2
        new_row2[pos2] = val1
        
        if not all(new_row1[i] < new_row1[i+1] for i in range(num_elements - 1)):
            return None
        if not all(new_row2[i] < new_row2[i+1] for i in range(num_elements - 1)):
            return None
        
        new_solution = [list(row) for row in solution]
        new_solution[row1] = new_row1
        new_solution[row2] = new_row2
        return (new_solution, None)
