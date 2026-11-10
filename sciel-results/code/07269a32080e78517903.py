import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using greedy initialization
    and simulated annealing to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return deadline - time.time()
    
    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry
    
    def greedy_initial(n_rows, k):
        """Simple greedy initialization."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                if time.time() > deadline - 15:
                    return None
            rows.append(row)
        return rows
    
    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]
        
        # Collect old differences involving position ci
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])
        
        # Remove old differences
        temp_used = used_diffs - old_diffs
        
        # Check if new_val is valid
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None
        
        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
        
        if not valid:
            return False, None
        
        # Success: return the new used set
        new_used = temp_used | new_diffs
        return True, new_used
    
    def simulated_annealing(rows, n_rows, k, max_iters=5000, initial_temp=100.0):
        """
        Simulated annealing to minimize the maximum entry.
        Uses multiple move types: decrease, increase, swap.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        current_score = max(row[-1] for row in rows)
        best_score = current_score
        best_rows = [row[:] for row in rows]
        
        rng = random.Random(int(time.time() * 1000) % 1000000)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            temp = initial_temp * (0.9995 ** iteration)
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.6:
                # Decrease an element
                ri = rng.randint(0, n_rows - 1)
                row = rows[ri]
                ci = rng.randint(1, len(row) - 1)
                current = row[ci]
                min_val = row[ci - 1] + 1
                max_val = (row[ci + 1] - 1) if ci < len(row) - 1 else current
                if min_val < max_val:
                    new_val = rng.randint(min_val, max_val)
                    if new_val < current:
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            new_score = max(r[-1] for r in rows)
                            delta = new_score - current_score
                            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_rows = [r[:] for r in rows]
                
            elif move_type < 0.85:
                # Increase an element (to escape local minima)
                ri = rng.randint(0, n_rows - 1)
                row = rows[ri]
                ci = rng.randint(1, len(row) - 1)
                current = row[ci]
                min_val = row[ci - 1] + 1
                max_val = (row[ci + 1] - 1) if ci < len(row) - 1 else current + 10
                if min_val < max_val:
                    new_val = rng.randint(min_val, max_val)
                    if new_val > current:
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            new_score = max(r[-1] for r in rows)
                            delta = new_score - current_score
                            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_rows = [r[:] for r in rows]
            
            else:
                # Swap two elements in different rows
                ri1 = rng.randint(0, n_rows - 1)
                ri2 = rng.randint(0, n_rows - 1)
                if ri1 != ri2:
                    row1 = rows[ri1]
                    row2 = rows[ri2]
                    ci1 = rng.randint(1, len(row1) - 1)
                    ci2 = rng.randint(1, len(row2) - 1)
                    
                    # Try to swap values
                    val1 = row1[ci1]
                    val2 = row2[ci2]
                    
                    # Check if swap is valid
                    # First remove both elements' contributions
                    old_diffs1 = set()
                    for i in range(ci1):
                        old_diffs1.add(row1[ci1] - row1[i])
                    for j in range(ci1 + 1, len(row1)):
                        old_diffs1.add(row1[j] - row1[ci1])
                    
                    old_diffs2 = set()
                    for i in range(ci2):
                        old_diffs2.add(row2[ci2] - row2[i])
                    for j in range(ci2 + 1, len(row2)):
                        old_diffs2.add(row2[j] - row2[ci2])
                    
                    temp_used = used_diffs - old_diffs1 - old_diffs2
                    
                    # Check if swapped values are valid
                    valid = True
                    new_diffs = set()
                    
                    # Check val2 in row1 at ci1
                    if ci1 > 0 and val2 <= row1[ci1 - 1]:
                        valid = False
                    elif ci1 < len(row1) - 1 and val2 >= row1[ci1 + 1]:
                        valid = False
                    else:
                        for i in range(ci1):
                            d = val2 - row1[i]
                            if d <= 0 or d in temp_used:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            for j in range(ci1 + 1, len(row1)):
                                d = row1[j] - val2
                                if d <= 0 or d in temp_used:
                                    valid = False
                                    break
                                new_diffs.add(d)
                    
                    if valid:
                        # Check val1 in row2 at ci2
                        if ci2 > 0 and val1 <= row2[ci2 - 1]:
                            valid = False
                        elif ci2 < len(row2) - 1 and val1 >= row2[ci2 + 1]:
                            valid = False
                        else:
                            for i in range(ci2):
                                d = val1 - row2[i]
                                if d <= 0 or d in (temp_used | new_diffs):
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                for j in range(ci2 + 1, len(row2)):
                                    d = row2[j] - val1
                                    if d <= 0 or d in (temp_used | new_diffs):
                                        valid = False
                                        break
                                    new_diffs.add(d)
                    
                    if valid:
                        new_score = max(r[-1] for r in rows)
                        delta = new_score - current_score
                        if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                            rows[ri1][ci1] = val2
                            rows[ri2][ci2] = val1
                            used_diffs = temp_used | new_diffs
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_rows = [r[:] for r in rows]
        
        return best_rows, best_score
    
    def greedy_reduce_max(rows, n_rows, k, max_iters=3000):
        """
        Greedy local search to reduce the maximum entry.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            max_val = rows[max_row_idx][-1]
            
            # Try to reduce the last element of the max row
            row = rows[max_row_idx]
            ci = len(row) - 1  # Last element
            
            # Try smaller values for the last element
            found = False
            for new_val in range(max_val - 1, row[ci - 1], -1):
                success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                if success:
                    rows[max_row_idx][ci] = new_val
                    used_diffs = new_used
                    new_score = max(r[-1] for r in rows)
                    if new_score < best_score:
                        best_score = new_score
                    found = True
                    break
            
            # If we couldn't reduce the last element, try other positions in that row
            if not found:
                for ci in range(len(row) - 2, 0, -1):
                    current = row[ci]
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                        if success:
                            rows[max_row_idx][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            found = True
                            break
                    if found:
                        break
            
            # Also try to reduce elements in other rows to free up differences
            if not found:
                for ri in range(n_rows):
                    if ri == max_row_idx:
                        continue
                    row = rows[ri]
                    for ci in range(len(row) - 1, 0, -1):
                        current = row[ci]
                        for new_val in range(current - 1, row[ci - 1], -1):
                            success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                            if success:
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not found:
                break
        
        return rows, best_score
    
    # Initialize with greedy
    best_rows = greedy_initial(n_rows, k)
    if best_rows is None:
        # Fallback: simple construction
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows
    
    best_score = validate(best_rows)[1]
    
    # First pass: greedy local search to reduce max
    improved_rows, improved_score = greedy_reduce_max(best_rows, n_rows, k, max_iters=2000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Second pass: simulated annealing
    improved_rows, improved_score = simulated_annealing(best_rows, n_rows, k, max_iters=8000, initial_temp=50.0)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Random restarts with perturbation
    rng = random.Random(42)
    
    for restart in range(15):
        if time_remaining() < 30:
            break
        
        # Perturb the solution
        perturbed = [row[:] for row in best_rows]
        for ri in range(n_rows):
            for ci in range(1, len(perturbed[ri])):
                if rng.random() < 0.15:
                    perturbed[ri][ci] += rng.randint(1, 10)
        
        # Validate perturbed
        valid, score = validate(perturbed)
        if not valid:
            continue
        
        # Simulated annealing from perturbed
        improved_rows, improved_score = simulated_annealing(perturbed, n_rows, k, max_iters=4000, initial_temp=30.0)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
    
    return best_rows
# EVOLVE-BLOCK-END
