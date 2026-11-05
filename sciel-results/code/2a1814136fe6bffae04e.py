import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd table for speed
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute "constraint degree": number of other moduli not coprime to this one
    # Higher degree means more constraints, so these are "harder" to place?
    # Or maybe easier because they restrict others more?
    # Let's sort by number of coprime partners (higher is easier?)
    # Actually, let's just use random restarts with a fast greedy.
    
    # To maximize count, we want to pick moduli that are pairwise coprime if possible.
    # Primes are the best candidates.
    # Let's identify primes first.
    def is_prime(k):
        if k < 2: return False
        for i in range(2, int(math.sqrt(k)) + 1):
            if k % i == 0:
                return False
        return True

    # Sort indices: Primes first (easiest), then composites.
    # Among composites, maybe smaller ones first?
    def sort_key(idx):
        m = moduli[idx]
        if is_prime(m):
            return (0, m)
        else:
            return (1, m)

    base_order = sorted(range(n), key=sort_key)

    best_size = 0
    best_assignment = {}

    # Fast Greedy Solver
    # Takes a list of indices to try to include.
    # It iterates through them, and for each, tries to find a valid residue.
    # If it finds one, it keeps it. If not, it skips it (greedy drop).
    # This guarantees a valid subset.
    def fast_greedy(indices_to_try):
        selected = [] # list of (idx, residue)
        for idx in indices_to_try:
            m = moduli[idx]
            # Try residues in random order? Or sorted?
            # Random is better for diversity.
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                valid = True
                for (prev_idx, prev_r) in selected:
                    g = gcd_tab[idx][prev_idx]
                    if (r - prev_r) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append((idx, r))
                    break
        return selected

    # Strategy 1: Greedy with fixed "easy" order (Primes first)
    # This will likely pick all primes and some composites.
    res = fast_greedy(base_order)
    if len(res) > best_size:
        best_size = len(res)
        best_assignment = {x[0]: x[1] for x in res}

    # Strategy 2: Random Restarts
    # Shuffle the order of moduli we attempt to add.
    # This allows different subsets to be found.
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        order = list(range(n))
        random.shuffle(order)
        
        res = fast_greedy(order)
        
        if len(res) > best_size:
            best_size = len(res)
            best_assignment = {x[0]: x[1] for x in res}
            
        # Optional: Local Search (Swap)
        # If we are stuck at a local optimum, try to swap a selected modulus with an unselected one.
        if best_size < n and time.time() < deadline - 10.0:
            current_indices = list(best_assignment.keys())
            missing_indices = [i for i in range(n) if i not in best_assignment]
            
            if missing_indices:
                # Try a few random swaps
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    if not current_indices:
                        break
                    idx_remove = random.choice(current_indices)
                    idx_add = random.choice(missing_indices)
                    
                    new_indices = [i for i in current_indices if i != idx_remove] + [idx_add]
                    
                    # Try to solve for this new set
                    # We need a solver that tries to assign residues to a *fixed* set of indices.
                    # If it fails, the swap is invalid.
                    # fast_greedy drops items, so it's not suitable for fixed set validation directly
                    # unless we modify it to return None if it can't place all.
                    
                    # Let's write a strict solver for a fixed set
                    def strict_greedy(fixed_indices):
                        selected = []
                        for idx in fixed_indices:
                            m = moduli[idx]
                            residues = list(range(m))
                            random.shuffle(residues)
                            for r in residues:
                                valid = True
                                for (prev_idx, prev_r) in selected:
                                    g = gcd_tab[idx][prev_idx]
                                    if (r - prev_r) % g == 0:
                                        valid = False
                                        break
                                if valid:
                                    selected.append((idx, r))
                                    break
                            else:
                                return None # Failed to place idx
                        return selected

                    # Try strict greedy on the new set
                    # To make it faster, we can reuse the previous assignment as a guide?
                    # No, just run strict greedy.
                    # Optimization: Sort fixed_indices by "difficulty" (number of constraints)
                    # But for small sets, it's fast enough.
                    new_res = strict_greedy(new_indices)
                    if new_res is not None:
                        # Valid swap found
                        best_assignment = {x[0]: x[1] for x in new_res}
                        current_indices = list(best_assignment.keys())
                        missing_indices = [i for i in range(n) if i not in best_assignment]
                        # Size is same, but configuration changed.
                        # We can break out of swap loop to try adding next?
                        # Or just continue swapping?
                        break

    return [[r, m] for (i, r) in best_assignment.items() for m in [moduli[i]]]
