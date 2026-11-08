<<<<<<< SEARCH
def hill_climb(initial_set, deadline, max_iter=5000):
=======
def simulated_annealing(initial_set, deadline, max_iter=20000, T_start=1.0, T_end=0.01):
    """Simulated annealing to escape local optima"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    best_score = compute_score(U)
    best_set = U[:]
    current_score = best_score
    
    n_steps = max_iter
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule (geometric cooling)
        progress = step / max(n_steps - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        n = len(U)
        max_val = max(U) if U else 0
        
        # Generate a move: replace an element, add, or remove
        move_type = random.random()
        new_U = U[:]
        
        if move_type < 0.6:  # Replace
            i = random.randint(0, n - 1)
            new_val = random.randint(0, max_val * 2 + 10)
            if new_val in U and new_val != U[i]:
                continue
            new_U[i] = new_val
        elif move_type < 0.8:  # Add
            if len(U) < 30:
                new_val = random.randint(1, max_val * 3 + 10)
                new_U.append(new_val)
            else:
                continue
        else:  # Remove
            if len(U) > 3:
                i = random.randint(0, n - 1)
                if U[i] == 0:
                    continue
                new_U.pop(i)
            else:
                continue
        
        new_U = sorted(set(new_U))
        if 0 not in new_U:
            new_U = sorted([0] + [x for x in new_U if x > 0])
        if len(new_U) < 2:
            continue
        
        new_score = compute_score(new_U)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            U = new_U
            current_score = new_score
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_set = U[:]
    
    return best_score, best_set

def hill_climb(initial_set, deadline, max_iter=5000):
>>>>>>> REPLACE

