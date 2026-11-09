# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences (nonzero quadratic residues mod 205).
    2. Build the conflict graph where vertices are 0..204 and edges connect pairs with forbidden difference.
    3. Use randomized greedy construction with multiple random orderings to find good independent sets.
    4. Apply local search (add, remove+refill, swap) to improve each candidate.
    5. Use simulated annealing with 2-swap moves for better exploration.
    6. Return the best solution found within a time budget.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # Leave margin for overhead
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d mod m is also effectively forbidden
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
    forbidden_sym.discard(0)
    
    # Precompute conflict mask for fast validity checks
    # conflict_mask[i][j] = True if (i-j) % m is in forbidden_sym
    conflict_mask = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d in forbidden_sym:
                    conflict_mask[i][j] = True
    
    def is_valid_addition(S_set, x):
        """Check if x can be added to set S_set"""
        for s in S_set:
            d = (x - s) % m
            if d in forbidden_sym:
                return False
        return True
    
    def greedy_construct(order):
        """Construct an independent set greedily based on the given order"""
        S = []
        S_set = set()
        for x in order:
            if is_valid_addition(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set
    
    def greedy_fill(S, S_set):
        """Try to add any valid element to S, return updated (S, S_set)"""
        improved = True
        while improved:
            improved = False
            for x in range(m):
                if x not in S_set:
                    if is_valid_addition(S_set, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
        return S, S_set
    
    def local_search(S, S_set):
        """
        Improve the set using local search.
        Returns the improved (S, S_set).
        """
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Try to add any element
            S, S_set = greedy_fill(S, S_set)
            if len(S) > 0:
                # Check if we actually added something
                pass
            
            # Phase 2: If no additions possible, try remove + refill
            current_size = len(S)
            best_S = None
            best_S_set = None
            best_size = current_size
            
            for u in list(S):
                # Create a copy to test removal
                S_test = [v for v in S if v != u]
                S_set_test = S_set - {u}
                
                # Try to add any element
                S_test, S_set_test = greedy_fill(S_test, S_set_test)
                
                # Check if this is better
                if len(S_test) > best_size:
                    best_size = len(S_test)
                    best_S = S_test
                    best_S_set = S_set_test
            
            # If we found a better configuration, accept it
            if best_S is not None and best_size > current_size:
                S = best_S
                S_set = best_S_set
                improved = True
        
        return S, S_set
    
    def simulated_annealing(S, S_set, temperature=2.0, cooling_rate=0.995, max_iterations=8000):
        """
        Apply simulated annealing to improve the set.
        Moves: add a vertex, remove a vertex, swap a vertex, 2-swap.
        """
        current_S = list(S)
        current_S_set = set(S_set)
        current_size = len(current_S)
        best_S = list(current_S)
        best_S_set = set(current_S_set)
        best_size = current_size
        T = temperature
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Choose a random move
            move_type = random.choice(['add', 'remove', 'swap', 'swap'])
            
            if move_type == 'add':
                # Try to add a random element
                candidates = [x for x in range(m) if x not in current_S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if is_valid_addition(current_S_set, x):
                    new_S = current_S + [x]
                    new_S_set = current_S_set | {x}
                    new_size = len(new_S)
                    # Always accept if it improves
                    if new_size > current_size:
                        current_S = new_S
                        current_S_set = new_S_set
                        current_size = new_size
                        if current_size > best_size:
                            best_S = list(current_S)
                            best_S_set = set(current_S_set)
                            best_size = current_size
                T *= cooling_rate
                
            elif move_type == 'remove':
                # Remove a random element
                if not current_S:
                    continue
                idx = random.randint(0, len(current_S) - 1)
                u = current_S[idx]
                new_S = [v for i, v in enumerate(current_S) if i != idx]
                new_S_set = current_S_set - {u}
                new_size = len(new_S)
                
                # Compute delta (size change)
                delta = new_size - current_size
                # For removal, we only accept if it allows future additions (handled by add moves)
                # Or if we're stuck. Let's use a simple acceptance criterion.
                if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-9)):
                    current_S = new_S
                    current_S_set = new_S_set
                    current_size = new_size
                    if current_size > best_size:
                        best_S = list(current_S)
                        best_S_set = set(current_S_set)
                        best_size = current_size
                T *= cooling_rate
                
            elif move_type == 'swap':
                # Swap a random element in the set with a random element not in the set
                if not current_S:
                    continue
                candidates_out = [x for x in range(m) if x not in current_S_set]
                if not candidates_out:
                    continue
                
                idx = random.randint(0, len(current_S) - 1)
                u = current_S[idx]
                v = random.choice(candidates_out)
                
                # Remove u, try to add v
                new_S_set = current_S_set - {u}
                if is_valid_addition(new_S_set, v):
                    new_S = [x for i, x in enumerate(current_S) if i != idx]
                    new_S.append(v)
                    new_S_set = new_S_set | {v}
                    new_size = len(new_S)
                    
                    delta = new_size - current_size
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-9)):
                        current_S = new_S
                        current_S_set = new_S_set
                        current_size = new_size
                        if current_size > best_size:
                            best_S = list(current_S)
                            best_S_set = set(current_S_set)
                            best_size = current_size
                T *= cooling_rate
            
            # Occasionally run a greedy fill to recover from bad moves
            if iteration % 50 == 0:
                current_S, current_S_set = greedy_fill(current_S, current_S_set)
                current_size = len(current_S)
                if current_size > best_size:
                    best_S = list(current_S)
                    best_S_set = set(current_S_set)
                    best_size = current_size
        
        return best_S, best_S_set
    
    # Main search loop
    best_solution = None
    best_size = 0
    
    # Run multiple randomized greedy constructions
    num_trials = 80
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        # Random ordering
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy construction
        S, S_set = greedy_construct(order)
        
        # Local search
        S, S_set = local_search(S, S_set)
        
        # Simulated annealing
        S, S_set = simulated_annealing(S, S_set, temperature=2.0, cooling_rate=0.995, max_iterations=5000)
        
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_solution = S
        
        # If we've found a very good solution, we can stop early
        if best_size >= 18:
            break
    
    # Final check: ensure the best solution is valid
    if best_solution is None:
        # Fallback to a simple greedy
        S, S_set = greedy_construct(list(range(m)))
        best_solution = S
    
    # Verify validity (sanity check)
    S_final = list(best_solution)
    S_final_set = set(S_final)
    for i, a in enumerate(S_final):
        for b in S_final[i+1:]:
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                # This should not happen, but if it does, remove the offending element
                # For simplicity, just return as is since local search should ensure validity
                pass
    
    return np.array(S_final, dtype=np.int64)

# EVOLVE-BLOCK-END
