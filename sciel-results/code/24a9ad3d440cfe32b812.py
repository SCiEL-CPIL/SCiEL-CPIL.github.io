import time
import random
import numpy as np

def _count_representations(A, targets):
    """Count representations for each target in targets using unordered pairs from A."""
    A_set = set(A)
    counts = {t: 0 for t in targets}
    A_list = sorted(A)
    n = len(A_list)
    for i in range(n):
        for j in range(i, n):
            s = A_list[i] + A_list[j]
            if s in counts:
                counts[s] += 1
    return counts

def _peak_mult(A):
    """Return the peak multiplicity for targets [128, 256]."""
    targets = set(range(128, 257))
    A_set = set(A)
    counts = {t: 0 for t in targets}
    A_list = sorted(A)
    n = len(A_list)
    for i in range(n):
        for j in range(i, n):
            s = A_list[i] + A_list[j]
            if s in counts:
                counts[s] += 1
    # Check coverage
    for t in targets:
        if counts[t] == 0:
            return float('inf')
    return max(counts[t] for t in targets)

def _is_valid(A):
    """Check if A covers all targets [128, 256]."""
    targets = set(range(128, 257))
    A_set = set(A)
    covered = set()
    A_list = sorted(A)
    n = len(A_list)
    for i in range(n):
        for j in range(i, n):
            s = A_list[i] + A_list[j]
            if 128 <= s <= 256:
                covered.add(s)
    return targets.issubset(covered)

def _eval(A):
    """Evaluate: return peak multiplicity, or inf if invalid."""
    targets = set(range(128, 257))
    A_list = sorted(A)
    n = len(A_list)
    counts = {t: 0 for t in targets}
    for i in range(n):
        for j in range(i, n):
            s = A_list[i] + A_list[j]
            if s in counts:
                counts[s] += 1
    for t in targets:
        if counts[t] == 0:
            return float('inf')
    return max(counts[t] for t in targets)

def construct():
    deadline = time.time() + 80  # Safety margin
    
    # Initial heuristic: try to find a good starting point
    # The seed [64,128] has peak mult that we can compute
    best_A = list(range(64, 129))
    best_score = _eval(best_A)
    
    if best_score == float('inf'):
        # Shouldn't happen with seed, but handle it
        best_A = list(range(64, 129))
        best_score = _eval(best_A)
    
    # Local search
    random.seed(42)
    all_elements = list(range(0, 129))
    
    def try_move(A, move_type, *args):
        """Try a move and return new_A and new_score, or None if invalid."""
        A_set = set(A)
        if move_type == 'add':
            x = args[0]
            if x in A_set:
                return None
            new_A = list(A_set | {x})
        elif move_type == 'remove':
            x = args[0]
            if x not in A_set:
                return None
            new_A = list(A_set - {x})
        elif move_type == 'swap':
            x_out, x_in = args[0], args[1]
            if x_out not in A_set or x_in in A_set:
                return None
            new_A = list((A_set - {x_out}) | {x_in})
        else:
            return None
        
        new_score = _eval(new_A)
        return new_A, new_score
    
    # Local search iterations
    current_A = list(best_A)
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Try random moves
        for _ in range(200):
            if time.time() >= deadline:
                break
            
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                x = random.choice(all_elements)
                result = try_move(current_A, 'add', x)
            elif move_type == 'remove':
                if len(current_A) <= 1:
                    continue
                x = random.choice(current_A)
                result = try_move(current_A, 'remove', x)
            elif move_type == 'swap':
                x_out = random.choice(current_A)
                x_in = random.choice(all_elements)
                if x_out == x_in:
                    continue
                result = try_move(current_A, 'swap', x_out, x_in)
            
            if result is None:
                continue
            
            new_A, new_score = result
            
            if new_score < current_score:
                current_A = new_A
                current_score = new_score
                improved = True
                no_improve_count = 0
                if current_score < best_score:
                    best_A = list(current_A)
                    best_score = current_score
                break  # Restart with new current
            elif new_score == current_score and new_score < float('inf'):
                # Accept neutral moves sometimes to explore
                if random.random() < 0.3:
                    current_A = new_A
                    no_improve_count += 1
                    break
            else:
                no_improve_count += 1
        
        if improved:
            continue
        
        # If no improvement, try a larger perturbation
        if no_improve_count > 100:
            # Perturb: add/remove a few random elements
            for _ in range(3):
                if random.random() < 0.5:
                    x = random.choice(all_elements)
                    if x not in current_A:
                        current_A.append(x)
                else:
                    if len(current_A) > 1:
                        x = random.choice(current_A)
                        current_A.remove(x)
            current_score = _eval(current_A)
            if current_score < best_score:
                best_A = list(current_A)
                best_score = current_score
            no_improve_count = 0
    
    return list(best_A)
