import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Identify prime powers for initial bias
    def is_prime_power(n):
        if n < 2:
            return False
        # Check if n is a power of a prime
        for p in range(2, int(math.sqrt(n)) + 1):
            if n % p == 0:
                temp = n
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    return True
                else:
                    return False
        return True # n is prime

    prime_powers = [m for m in moduli_list if is_prime_power(m)]
    other_moduli = [m for m in moduli_list if m not in prime_powers]
    
    best_count = 0
    best_result = None
    
    def is_valid_assignment(chosen, m_new, r_new):
        """Check if adding (r_new, m_new) to chosen is valid."""
        for (r, m) in chosen:
            g = gcd_table[m_new][m]
            if (r_new - r) % g == 0:
                return False
        return True

    def find_valid_residue(chosen, m):
        """Find a valid residue for m given chosen."""
        for r in range(m):
            if is_valid_assignment(chosen, m, r):
                return r
        return None

    def greedy_construct(order):
        chosen = []
        for m in order:
            r = find_valid_residue(chosen, m)
            if r is not None:
                chosen.append((r, m))
        return chosen

    def backtrack_search(chosen, remaining, deadline):
        """
        Recursive backtracking to add more moduli.
        chosen: list of (r, m)
        remaining: list of moduli to consider
        """
        if time.time() > deadline:
            return chosen
        
        if not remaining:
            return chosen
        
        # Heuristic: sort remaining by number of constraints or just random
        # To speed up, try to add one that is 'easy' (few conflicts)
        # Or just pick random from remaining
        
        # Try to add the first one in remaining
        m = remaining[0]
        rest = remaining[1:]
        
        # Try all residues for m
        found = False
        # Randomize residue order to break symmetry
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            if is_valid_assignment(chosen, m, r):
                new_chosen = chosen + [(r, m)]
                result = backtrack_search(new_chosen, rest, deadline)
                if len(result) > len(chosen):
                    return result
                # If we didn't improve, we might have hit a dead end deeper, 
                # but for now we just return the best found in this branch.
                # Actually, if result is same size, we failed to add m effectively?
                # No, result might be same size if m couldn't be added later?
                # Backtracking should return the max size set from this state.
                found = True
                break # Just take the first valid residue for now to keep it fast? 
                # No, we need to explore. But time is tight.
        
        if not found:
            # m cannot be added with any residue
            return backtrack_search(chosen, rest, deadline)
            
        return chosen # Fallback, shouldn't reach if found is True and we return result

    # Strategy: 
    # 1. Greedy with prime powers first.
    # 2. Local search / Backtracking on the remainder.
    
    num_restarts = 1000
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Order: Prime powers first (shuffled), then others (shuffled)
        pp = prime_powers[:]
        om = other_moduli[:]
        random.shuffle(pp)
        random.shuffle(om)
        order = pp + om
        
        chosen = greedy_construct(order)
        
        # Try to improve with local search: 
        # If we have unused moduli, try to swap one used for one unused?
        # Or just try to add unused ones with backtracking?
        
        # Simple improvement: try to add any unused modulus that fits
        used_moduli = set(m for _, m in chosen)
        unused = [m for m in moduli_list if m not in used_moduli]
        random.shuffle(unused)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for m in unused:
                if time.time() > deadline:
                    break
                r = find_valid_residue(chosen, m)
                if r is not None:
                    chosen.append((r, m))
                    used_moduli.add(m)
                    unused.remove(m)
                    improved = True
                    break # Restart unused list scan
            
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
            
        # If we are stuck at a low count, maybe try a different heuristic?
        # The prime power greedy should get ~42.
        
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result
