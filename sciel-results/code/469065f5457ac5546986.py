"""Difference triangle set construction with deadline-aware search."""
import time
import random
import numpy as np

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return deadline - time.time()
    
    def is_valid(rows):
        """Check if all positive differences are distinct."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
    
    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(row) for row in rows)
    
    def greedy_construct(n_rows, k, randomize=False):
        """Greedy construction with optional randomization."""
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                if randomize and random.random() < 0.1:
                    # Sometimes skip a valid position to explore alternatives
                    if random.random() < 0.3:
                        continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows
    
    def local_search(rows, max_iterations=10000):
        """
        Local search to reduce the scope.
        Try to reduce individual entries while maintaining validity.
        """
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        
        for iteration in range(max_iterations):
            if time_remaining() < 5:
                break
            
            improved = False
            # Try to reduce each entry in each row
            for r in range(len(rows)):
                for pos in range(1, len(rows[r])):  # Skip position 0 (always 0)
                    if time_remaining() < 5:
                        break
                    
                    current_val = rows[r][pos]
                    # Try reducing the value
                    for new_val in range(current_val - 1, rows[r][pos - 1], -1):
                        if new_val <= rows[r][pos - 1]:
                            break
                        
                        # Check if reducing this value maintains validity
                        old_row = rows[r]
                        rows[r][pos] = new_val
                        
                        # Check increasing order
                        if rows[r][pos] <= rows[r][pos - 1]:
                            rows[r][pos] = current_val
                            continue
                        
                        # Check all differences are still distinct
                        valid = True
                        used = set()
                        for row in rows:
                            for i in range(len(row)):
                                for j in range(i + 1, len(row)):
                                    diff = row[j] - row[i]
                                    if diff in used:
                                        valid = False
                                        break
                                    used.add(diff)
                            if not valid:
                                break
                        
                        if valid:
                            scope = min(scope, max(max(row) for row in rows))
                            improved = True
                            break
                        else:
                            rows[r][pos] = current_val
                            break
            
            if not improved:
                break
        
        return rows
    
    def hill_climbing_with_restarts(rows, n_restarts=50):
        """
        Hill climbing with random restarts to escape local optima.
        """
        best = [list(r) for r in rows]
        best_scope = get_scope(best)
        
        for restart in range(n_restarts):
            if time_remaining() < 10:
                break
            
            # Create a perturbed version
            current = [list(r) for r in rows]
            scope = get_scope(current)
            
            # Random perturbation: randomly increase some values
            for _ in range(random.randint(1, 3)):
                r = random.randint(0, len(current) - 1)
                pos = random.randint(1, len(current[r]) - 1)
                current[r][pos] += random.randint(1, 5)
            
            # Local search from perturbed state
            current = local_search(current, max_iterations=2000)
            scope = get_scope(current)
            
            if scope < best_scope:
                best = [list(r) for r in current]
                best_scope = scope
        
        return best
    
    def simulated_annealing(rows, max_iterations=50000):
        """
        Simulated annealing to find better solutions.
        """
        current = [list(r) for r in rows]
        current_scope = get_scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope
        
        temperature = 10.0
        cooling_rate = 0.9995
        
        for iteration in range(max_iterations):
            if time_remaining() < 10:
                break
            
            # Generate a neighbor
            neighbor = [list(r) for r in current]
            r = random.randint(0, len(neighbor) - 1)
            pos = random.randint(1, len(neighbor[r]) - 1)
            
            # Try to decrease or slightly increase
            if random.random() < 0.7:
                delta = -random.randint(1, 3)
            else:
                delta = random.randint(1, 3)
            
            new_val = neighbor[r][pos] + delta
            
            # Check bounds
            if new_val <= neighbor[r][pos - 1] or (pos == len(neighbor[r]) - 1 and new_val < 0):
                continue
            
            neighbor[r][pos] = new_val
            
            # Check validity
            if not is_valid(neighbor):
                continue
            
            neighbor_scope = get_scope(neighbor)
            delta_scope = neighbor_scope - current_scope
            
            if delta_scope < 0 or random.random() < np.exp(-delta_scope / max(temperature, 0.01)):
                current = neighbor
                current_scope = neighbor_scope
                
                if current_scope < best_scope:
                    best = [list(r) for r in current]
                    best_scope = current_scope
            
            temperature *= cooling_rate
        
        return best
    
    # Initial construction
    initial_rows = greedy_construct(n_rows, k)
    best_rows = [list(r) for r in initial_rows]
    best_scope = get_scope(best_rows)
    
    # Apply local search
    polished = local_search(best_rows, max_iterations=5000)
    if get_scope(polished) < best_scope:
        best_rows = polished
        best_scope = get_scope(best_rows)
    
    # Try simulated annealing
    if time_remaining() > 20:
        sa_result = simulated_annealing(best_rows, max_iterations=30000)
        if get_scope(sa_result) < best_scope:
            best_rows = sa_result
            best_scope = get_scope(best_rows)
    
    # Try hill climbing with restarts
    if time_remaining() > 15:
        hc_result = hill_climbing_with_restarts(best_rows, n_restarts=20)
        if get_scope(hc_result) < best_scope:
            best_rows = hc_result
            best_scope = get_scope(best_rows)
    
    # Final local search polish
    final = local_search(best_rows, max_iterations=10000)
    if get_scope(final) < best_scope:
        best_rows = final
    
    # Ensure validity
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows
