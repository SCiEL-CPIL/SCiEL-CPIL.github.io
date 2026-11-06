<<<<<<< SEARCH
def two_opt_polish(cycle, edge_pos, deadline):
    """Apply 2-opt to minimize max edge position."""
    if cycle is None:
        return None
    
    def cycle_score(cyc):
        max_pos = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    # Try 2-opt moves
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        n = len(best_cycle)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Reverse segment from i+1 to j
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
    
    return best_cycle
=======
def two_opt_polish(cycle, edge_pos, deadline):
    """Apply 2-opt to minimize max edge position."""
    if cycle is None:
        return None
    
    def cycle_score(cyc):
        max_pos = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    # Try 2-opt moves
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        n = len(best_cycle)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Reverse segment from i+1 to j
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
    
    # Simulated annealing polish to escape local minima
    if time.time() < deadline - 2:
        rng = random.Random(42)
        current_cycle = best_cycle[:]
        current_score = best_score
        T = 50.0
        cooling = 0.995
        min_T = 0.1
        
        while T > min_T and time.time() < deadline - 2:
            i = rng.randrange(n - 1)
            j = i + 2 + rng.randrange(n - i - 2)
            if i == 0 and j == n - 1:
                j = i + 1 + rng.randrange(n - i - 2)
                if j <= i + 1:
                    T *= cooling
                    continue
            
            new_cycle = current_cycle[:i+1] + current_cycle[i+1:j+1][::-1] + current_cycle[j+1:]
            new_score = cycle_score(new_cycle)
            
            delta = new_score - current_score
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current_cycle = new_cycle
                current_score = new_score
                if current_score < best_score:
                    best_cycle = current_cycle[:]
                    best_score = current_score
            
            T *= cooling
    
    return best_cycle
>>>>>>> REPLACE

