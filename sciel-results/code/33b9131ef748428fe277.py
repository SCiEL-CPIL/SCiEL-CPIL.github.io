import time
import random
import math

def calculate_score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if not U:
        return 0.0
    U = sorted(list(set(U)))
    if 0 not in U:
        U = sorted(list(set(U) | {0}))
        
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sums.add(ui + uj)
            diffs.add(ui - uj)
            
    num = len(diffs)
    den = len(sums)
    if den == 0:
        return 0.0
    ratio = num / den
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Initial candidates: Golomb rulers and variations
    initial_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 44],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
        [0, 2, 7, 15],
        [0, 1, 4, 10, 12, 17, 18, 23],
        [0, 1, 3, 7, 12, 18, 25],
        [0, 2, 5, 11, 17, 23, 31],
    ]
    
    for s in initial_sets:
        s = sorted(set(s))
        if 0 not in s:
            s = sorted(set(s) | {0})
        score = calculate_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    random.seed(42)
    
    def local_search(start_set, max_iterations=500):
        """Perform local search from a starting set."""
        current_set = sorted(set(start_set) | {0})
        current_score = calculate_score(current_set)
        no_improvement = 0
        max_no_improvement = 80
        
        while no_improvement < max_no_improvement and time.time() < deadline:
            improved = False
            n = len(current_set)
            
            # Try all possible moves: perturb, add, remove
            moves = []
            
            # Perturb moves
            for i in range(1, n):
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = current_set[i] + delta
                    if new_val >= 0 and new_val not in current_set:
                        candidate = list(current_set)
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate:
                            candidate = sorted(set(candidate) | {0})
                        moves.append((candidate, "perturb"))
            
            # Add moves
            for i in range(n):
                for delta in range(1, 8):
                    new_val = current_set[i] + delta
                    if new_val not in current_set and new_val >= 0:
                        candidate = sorted(set(current_set) | {new_val})
                        moves.append((candidate, "add"))
            
            # Remove moves
            if n > 3:
                for i in range(1, n):
                    candidate = list(current_set)
                    candidate.pop(i)
                    candidate = sorted(set(candidate))
                    if 0 not in candidate:
                        candidate = sorted(set(candidate) | {0})
                    moves.append((candidate, "remove"))
            
            # Evaluate moves and pick the best improvement
            best_move_score = current_score
            best_move = None
            
            for candidate, move_type in moves:
                s = calculate_score(candidate)
                if s > best_move_score:
                    best_move_score = s
                    best_move = candidate
            
            if best_move is not None:
                current_set = best_move
                current_score = best_move_score
                no_improvement = 0
                improved = True
            else:
                no_improvement += 1
        
        return current_set, current_score
    
    # Main search loop with multiple restarts
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Choose starting point
        if restart % 3 == 0:
            # Start from best found so far
            start_set = list(best_set)
        elif restart % 3 == 1:
            # Random Golomb-like set
            n = random.randint(4, 12)
            start_set = [0]
            last = 0
            for i in range(1, n):
                gap = random.randint(1, 6)
                last += gap
                start_set.append(last)
            start_set = sorted(start_set)
        else:
            # Perturbed version of best
            start_set = list(best_set)
            if len(start_set) > 1:
                idx = random.randint(1, len(start_set)-1)
                delta = random.randint(-4, 4)
                if delta != 0:
                    new_val = start_set[idx] + delta
                    if new_val >= 0 and new_val not in start_set:
                        start_set[idx] = new_val
                        start_set = sorted(set(start_set))
        
        result_set, result_score = local_search(start_set)
        
        if result_score > best_score:
            best_score = result_score
            best_set = result_set
    
    # Final polish: fine-tune the best set with smaller steps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(1, len(best_set)):
                for delta in [-2, -1, 1, 2]:
                    candidate = list(best_set)
                    new_val = candidate[i] + delta
                    if new_val >= 0 and new_val not in candidate:
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate:
                            candidate = sorted(set(candidate) | {0})
                        s = calculate_score(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            improved = True
                            break
                if improved:
                    break
            
            # Try adding a new element
            if not improved:
                for delta in [1, 2, 3]:
                    new_val = max(best_set) + delta
                    candidate = sorted(set(best_set) | {new_val})
                    s = calculate_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        improved = True
            
            # Try removing an element
            if not improved and len(best_set) > 3:
                for i in range(1, len(best_set)):
                    candidate = list(best_set)
                    candidate.pop(i)
                    candidate = sorted(set(candidate))
                    if 0 not in candidate:
                        candidate = sorted(set(candidate) | {0})
                    s = calculate_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        improved = True
    
    return best_set
