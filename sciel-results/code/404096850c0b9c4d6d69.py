import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute prime factors (omega) for ordering heuristics
    def get_omega(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    omega_cache = [0] * (MAX_MOD + 1)
    for m in moduli:
        omega_cache[m] = get_omega(m)

    best_result = [[1, 2]]  # fallback
    best_count = 1

    def find_residue_fast(m_idx, selected):
        """
        Find a valid residue for moduli[m_idx] given a list of selected [r, m] pairs.
        Uses a boolean array to mark forbidden residues.
        """
        m = moduli[m_idx]
        # forbidden[r] = True if residue r is invalid
        forbidden = [False] * m
        
        for (r_i, m_i) in selected:
            # Find index of m_i in moduli list for gcd lookup
            # Since moduli is sorted and small, we can use a dict or just math.gcd directly
            # But we have gcd_table. We need the index of m_i.
            # To avoid O(N) search for index, we can store index in selected tuples or use a dict.
            # Let's modify selected to store indices for speed in the main loop.
            pass # This logic will be adapted in the caller to pass indices.

    def try_add_residue_with_indices(m_idx, selected_indices):
        """
        selected_indices: list of (r, m_idx)
        """
        m = moduli[m_idx]
        forbidden = [False] * m
        
        for (r_i, m_i_idx) in selected_indices:
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                # Residues r such that (r - r_i) % g == 0 are forbidden
                # r = r_i + k*g
                # We need r in [0, m)
                # First forbidden residue in [0, m) is r_i % m ? No.
                # We need r in [0, m) such that r % g == r_i % g.
                # Let target_rem = r_i % g
                # r can be target_rem, target_rem + g, target_rem + 2g, ...
                target_rem = r_i % g
                # Mark all r in [0, m) where r % g == target_rem
                for r in range(target_rem, m, g):
                    forbidden[r] = True
        
        # Find first unforbidden residue
        # Randomize start to break symmetry? Or just first?
        # First is deterministic and fast.
        for r in range(m):
            if not forbidden[r]:
                return r
        return None

    def greedy_pass(order):
        """
        order: list of indices of moduli to try
        returns list of [r, m]
        """
        selected = [] # list of [r, m]
        selected_indices = [] # list of (r, m_idx)
        used = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            if idx in used:
                continue
            
            r = try_add_residue_with_indices(idx, selected_indices)
            if r is not None:
                selected.append([r, moduli[idx]])
                selected_indices.append((r, idx))
                used.add(idx)
        
        return selected

    # Define orders
    indices = list(range(n_mod))
    
    # Order 1: Random
    # Order 2: Descending Omega (most complex first)
    order_omega_desc = sorted(indices, key=lambda i: -omega_cache[moduli[i]])
    # Order 3: Ascending Omega (simplest first)
    order_omega_asc = sorted(indices, key=lambda i: omega_cache[moduli[i]])
    # Order 4: Ascending Modulus
    order_mod_asc = sorted(indices, key=lambda i: moduli[i])
    # Order 5: Descending Modulus
    order_mod_desc = sorted(indices, key=lambda i: -moduli[i])

    # Run fixed heuristics first
    for order in [order_omega_desc, order_omega_asc, order_mod_asc, order_mod_desc]:
        result = greedy_pass(order)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result

    # Run randomized restarts
    # We can also try local search: swap one element to allow another
    def local_search(initial_result):
        """
        Try to improve a given result by swapping or adding.
        initial_result: list of [r, m]
        """
        if not initial_result:
            return []
        
        current = [[r, m] for r, m in initial_result]
        current_indices = []
        # Rebuild indices
        mod_to_idx = {m: i for i, m in enumerate(moduli)}
        for r, m in current:
            current_indices.append((r, mod_to_idx[m]))
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to add any unused modulus
            used_mods = set(m for _, m in current)
            for idx in range(n_mod):
                if moduli[idx] in used_mods:
                    continue
                r = try_add_residue_with_indices(idx, current_indices)
                if r is not None:
                    current.append([r, moduli[idx]])
                    current_indices.append((r, idx))
                    improved = True
                    break # Restart scan from beginning? Or continue? 
                    # Continuing might be faster if we just append.
                    # But the order of checking matters. Let's restart to be safe?
                    # No, just continue the loop.
            
            # If no addition, try a swap: remove one, add another
            if not improved and time.time() < deadline - 10.0:
                # Try removing each element and see if we can add more
                # This is expensive. Maybe just try removing the "least constrained" or random ones.
                # Let's try removing a random element and re-greedy the rest?
                # Too expensive.
                # Instead, try a 1-swap: remove i, add j.
                # We only do this if we are stuck.
                # For now, let's just rely on the addition phase and randomized restarts.
                pass

        return current

    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random order
        random.shuffle(indices)
        # Mix in some heuristic bias?
        # 50% random, 50% heuristic
        if restart % 2 == 0:
            order = indices
        else:
            # Randomized heuristic: sort by omega + noise
            order = sorted(indices, key=lambda i: -omega_cache[moduli[i]] + random.uniform(-0.5, 0.5))
            
        result = greedy_pass(order)
        
        # Local improvement (addition only)
        # Rebuild indices for local search
        mod_to_idx = {m: i for i, m in enumerate(moduli)}
        result_indices = [(r, mod_to_idx[m]) for r, m in result]
        
        # Quick addition pass
        used_mods = set(m for _, m in result)
        for idx in range(n_mod):
            if moduli[idx] in used_mods:
                continue
            r = try_add_residue_with_indices(idx, result_indices)
            if r is not None:
                result.append([r, moduli[idx]])
                result_indices.append((r, idx))
                used_mods.add(moduli[idx])
        
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
            
            # If we found a very good result, maybe do a more intensive local search?
            # For now, just keep going.

    return best_result
