import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n_moduli = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * (max_modulus + 1) for _ in range(max_modulus + 1)]
    for i in range(2, max_modulus + 1):
        for j in range(i, max_modulus + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_compatible(r_new, m_new, selected):
        """Check if (r_new, m_new) is compatible with all selected pairs."""
        for r_k, m_k in selected:
            g = gcd_table[m_new][m_k]
            if g > 1 and (r_new - r_k) % g == 0:
                return False
        return True
    
    def greedy_build(order, randomize_residue=False):
        """Build a solution greedily using the given order of moduli."""
        selected = []
        for m in order:
            # Try residues in a specific order
            if randomize_residue:
                residues = list(range(m))
                random.shuffle(residues)
            else:
                residues = range(m)
            
            for r in residues:
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    break
        return selected
    
    def local_search(selected, max_iters=2000):
        """Improve solution via residue swaps and additions."""
        selected = list(selected)
        best_len = len(selected)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new modulus
            selected_moduli = set(m for r, m in selected)
            remaining = [m for m in moduli if m not in selected_moduli]
            
            for m in remaining:
                if time.time() > deadline:
                    break
                for r in range(m):
                    if is_compatible(r, m, selected):
                        selected.append((r, m))
                        best_len = len(selected)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try residue swaps: change residue of one selected pair
            for i, (r_old, m_old) in enumerate(selected):
                if time.time() > deadline:
                    break
                for r_new in range(m_old):
                    if r_new == r_old:
                        continue
                    # Check if new residue is compatible with all others
                    compatible = True
                    for j, (r_k, m_k) in enumerate(selected):
                        if j == i:
                            continue
                        g = gcd_table[m_old][m_k]
                        if g > 1 and (r_new - r_k) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        # This swap doesn't change count, but might enable future additions
                        # Only do it if it helps (we'd need to check, which is expensive)
                        # For now, skip pure swaps unless they enable additions
                        pass
            
            if not improved:
                # Try removing a pair and re-adding different ones
                # This is more expensive, so only do it occasionally
                if it % 10 == 0 and len(selected) > 2:
                    # Try removing the last added element and see if we can add two
                    removed = selected.pop()
                    added_count = 0
                    for m in moduli:
                        if time.time() > deadline:
                            break
                        if m == removed[1]:
                            continue
                        if m in set(mk for rk, mk in selected):
                            continue
                        for r in range(m):
                            if is_compatible(r, m, selected):
                                selected.append((r, m))
                                added_count += 1
                                break
                        if added_count >= 2:
                            break
                    
                    if added_count < 1:
                        selected.append(removed)  # Put it back
                    # If we added at least 1, we're better off
                    if len(selected) > best_len:
                        best_len = len(selected)
                        improved = True
        
        return selected
    
    # Multiple greedy strategies
    best_solution = []
    best_len = 0
    
    # Strategy 1: Sorted by number of divisors (descending) - hard moduli first
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    div_counts = {m: num_divisors(m) for m in moduli}
    prime_flags = {m: is_prime(m) for m in moduli}
    
    orders = []
    # Primes first (easiest to satisfy)
    orders.append(sorted(moduli, key=lambda m: (not prime_flags[m], -div_counts[m])))
    # Then composites
    orders.append(sorted(moduli, key=lambda m: (prime_flags[m], -div_counts[m])))
    # Descending divisors
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    # Ascending divisors
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    # Natural order
    orders.append(sorted(moduli))
    # Reverse natural
    orders.append(sorted(moduli, reverse=True))
    # Primes only first, then composites in reverse
    primes = [m for m in moduli if prime_flags[m]]
    composites = [m for m in moduli if not prime_flags[m]]
    orders.append(primes + composites)
    orders.append(composites + primes)
    
    # Random orders
    random.seed(42)
    for _ in range(50):
        shuffled = moduli.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy_build(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Local search from best solution
    if best_solution:
        improved = local_search(best_solution)
        if len(improved) > best_len:
            best_len = len(improved)
            best_solution = improved
    
    # More random restarts with local search
    for _ in range(100):
        if time.time() > deadline:
            break
        shuffled = moduli.copy()
        random.shuffle(shuffled)
        sol = greedy_build(shuffled, randomize_residue=True)
        if len(sol) > best_len:
            sol = local_search(sol, max_iters=500)
            if len(sol) > best_len:
                best_len = len(sol)
                best_solution = sol
        elif len(sol) == best_len and random.random() < 0.3:
            # Sometimes try local search even if same length
            sol2 = local_search(sol, max_iters=300)
            if len(sol2) > best_len:
                best_len = len(sol2)
                best_solution = sol2
    
    # Backtracking from best solution to try to add more
    def backtrack_add(selected, remaining_moduli, depth=0):
        """Try to add more moduli via backtracking."""
        nonlocal best_solution, best_len
        
        if time.time() > deadline:
            return
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_solution = list(selected)
        
        if not remaining_moduli or depth > 30:
            return
        
        for idx, m in enumerate(remaining_moduli):
            if time.time() > deadline:
                return
            
            # Pruning
            if len(selected) + len(remaining_moduli) - idx <= best_len:
                break
            
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    backtrack_add(selected, remaining_moduli[idx+1:], depth+1)
                    selected.pop()
                    break  # Only try one residue per modulus for speed
    
    if best_solution:
        selected_moduli = set(m for r, m in best_solution)
        remaining = [m for m in moduli if m not in selected_moduli]
        # Sort remaining by "difficulty" - fewer options first
        remaining.sort(key=lambda m: -div_counts[m])
        current = list(best_solution)
        backtrack_add(current, remaining)
    
    return [[r, m] for r, m in best_solution]
