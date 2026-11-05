import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcd_table = {}
    for i in range(n_moduli):
        for j in range(i+1, n_moduli):
            m1 = moduli[i]
            m2 = moduli[j]
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def are_compatible(r1, m1, r2, m2):
        """Check if two residue classes are disjoint (compatible)."""
        g = gcd_table.get((m1, m2), math.gcd(m1, m2))
        return (r1 - r2) % g != 0
    
    def count_compatible(r, m, selected):
        """Count how many selected classes are compatible with (r,m)."""
        count = 0
        for r_k, m_k in selected:
            if are_compatible(r, m, r_k, m_k):
                count += 1
        return count
    
    def is_fully_compatible(r, m, selected):
        """Check if (r,m) is compatible with ALL selected classes."""
        for r_k, m_k in selected:
            if not are_compatible(r, m, r_k, m_k):
                return False
        return True
    
    def greedy_build(order, random_residues=False, seed=None):
        """Build a solution greedily following the given order."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        selected = []
        for m in order:
            best_r = None
            best_score = -1
            if random_residues:
                candidates = list(range(m))
                rng.shuffle(candidates)
                for r in candidates:
                    if is_fully_compatible(r, m, selected):
                        # Score by how many future moduli this residue conflicts with
                        score = 0
                        for m2 in order:
                            if m2 == m:
                                continue
                            if m2 in [x[1] for x in selected]:
                                continue
                            g = gcd_table.get((m, m2), math.gcd(m, m2))
                            if r % g == 0:  # This residue might conflict
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_r = r
            else:
                for r in range(m):
                    if is_fully_compatible(r, m, selected):
                        # Prefer residues that minimize future conflicts
                        score = 0
                        for m2 in order:
                            if m2 == m:
                                continue
                            if m2 in [x[1] for x in selected]:
                                continue
                            g = gcd_table.get((m, m2), math.gcd(m, m2))
                            if r % g == 0:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_r = r
            if best_r is not None:
                selected.append((best_r, m))
        return selected
    
    # Local search: try to improve by swapping residues or adding/removing moduli
    def local_search(initial_solution, max_iters=1000):
        """Improve solution via local search (swap residues, add moduli)."""
        rng = random.Random()
        selected = list(initial_solution)
        selected_set = set(m for r, m in selected)
        best = list(selected)
        best_len = len(selected)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 1: Add a missing modulus
            missing = [m for m in moduli if m not in selected_set]
            if missing:
                rng.shuffle(missing)
                for m in missing[:20]:  # Try up to 20 missing moduli
                    for r in range(m):
                        if is_fully_compatible(r, m, selected):
                            new_sel = selected + [(r, m)]
                            if len(new_sel) > best_len:
                                best = new_sel
                                best_len = len(new_sel)
                                selected = new_sel
                                selected_set.add(m)
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2: Change residue of an existing modulus
            if selected:
                idx = rng.randrange(len(selected))
                r_old, m_old = selected[idx]
                for r_new in range(m_old):
                    if r_new == r_old:
                        continue
                    # Check if new residue is compatible with all others
                    compatible = True
                    for i, (r_k, m_k) in enumerate(selected):
                        if i == idx:
                            continue
                        if not are_compatible(r_new, m_old, r_k, m_k):
                            compatible = False
                            break
                    if compatible:
                        # Accept if it doesn't reduce length (same length, maybe better structure)
                        # For now, just accept to explore
                        selected[idx] = (r_new, m_old)
                        improved = True
                        break
            
            # Try 3: Remove a modulus and try a different one
            if not improved and len(selected) > 1:
                idx = rng.randrange(len(selected))
                r_removed, m_removed = selected.pop(idx)
                selected_set.discard(m_removed)
                
                # Try to add a different modulus
                missing = [m for m in moduli if m not in selected_set]
                if missing:
                    rng.shuffle(missing)
                    for m_new in missing[:10]:
                        for r_new in range(m_new):
                            if is_fully_compatible(r_new, m_new, selected):
                                selected.append((r_new, m_new))
                                selected_set.add(m_new)
                                improved = True
                                break
                        if improved:
                            break
                
                if not improved:
                    # Restore
                    selected.append((r_removed, m_removed))
                    selected_set.add(m_removed)
            
            if not improved:
                break
        
        return best
    
    # Simulated annealing style search
    def simulated_annealing(initial_solution, max_iters=5000):
        """Simulated annealing to find better solutions."""
        rng = random.Random()
        selected = list(initial_solution)
        best = list(selected)
        best_len = len(selected)
        current_temp = 1.0
        cooling = 0.999
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate a neighbor
            neighbor = list(selected)
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a random missing modulus
                missing = [m for m in moduli if m not in [x[1] for x in neighbor]]
                if missing:
                    m = rng.choice(missing)
                    valid_rs = [r for r in range(m) if is_fully_compatible(r, m, neighbor)]
                    if valid_rs:
                        r = rng.choice(valid_rs)
                        neighbor.append((r, m))
            
            elif move_type < 0.6:
                # Remove a random modulus
                if neighbor:
                    idx = rng.randrange(len(neighbor))
                    neighbor.pop(idx)
            
            elif move_type < 0.85:
                # Change a random residue
                if neighbor:
                    idx = rng.randrange(len(neighbor))
                    r_old, m_old = neighbor[idx]
                    valid_rs = [r for r in range(m_old) if is_fully_compatible(r, m_old, [x for i, x in enumerate(neighbor) if i != idx])]
                    if valid_rs:
                        r_new = rng.choice(valid_rs)
                        neighbor[idx] = (r_new, m_old)
            
            else:
                # Swap: remove one, add another
                if neighbor:
                    idx = rng.randrange(len(neighbor))
                    r_removed, m_removed = neighbor.pop(idx)
                    missing = [m for m in moduli if m not in [x[1] for x in neighbor]]
                    if missing:
                        m_new = rng.choice(missing)
                        valid_rs = [r for r in range(m_new) if is_fully_compatible(r, m_new, neighbor)]
                        if valid_rs:
                            r_new = rng.choice(valid_rs)
                            neighbor.append((r_new, m_new))
            
            delta = len(neighbor) - len(selected)
            if delta >= 0 or rng.random() < math.exp(delta / max(current_temp, 0.001)):
                selected = neighbor
                if len(selected) > best_len:
                    best = list(selected)
                    best_len = len(selected)
            
            current_temp *= cooling
        
        return best
    
    # Generate diverse initial solutions
    best_solution = []
    best_len = 0
    
    # Multiple greedy strategies
    orders = []
    
    # Sort by various criteria
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
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    div_counts = {m: num_divisors(m) for m in moduli}
    
    # Primes first (they have fewest conflicts)
    orders.append(sorted(moduli, key=lambda m: (not is_prime(m), m)))
    # Most composite first
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    # Least composite first
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    # Natural order
    orders.append(sorted(moduli))
    # Reverse natural
    orders.append(sorted(moduli, reverse=True))
    
    # Random orders
    rng = random.Random(42)
    for _ in range(50):
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    for i, order in enumerate(orders):
        if time.time() > deadline:
            break
        
        # Try both deterministic and random residue selection
        for use_random in [False, True]:
            sol = greedy_build(order, random_residues=use_random, seed=i if use_random else None)
            if len(sol) > best_len:
                best_len = len(sol)
                best_solution = sol
    
    # Apply local search and simulated annealing to best solutions
    if best_len > 0:
        # Local search
        improved = local_search(best_solution, max_iters=2000)
        if len(improved) > best_len:
            best_len = len(improved)
            best_solution = improved
        
        # Simulated annealing
        sa_result = simulated_annealing(best_solution, max_iters=10000)
        if len(sa_result) > best_len:
            best_len = len(sa_result)
            best_solution = sa_result
    
    # Try more diverse initial solutions with local search
    rng2 = random.Random(123)
    for i in range(20):
        if time.time() > deadline:
            break
        
        shuffled = moduli.copy()
        rng2.shuffle(shuffled)
        sol = greedy_build(shuffled, random_residues=True, seed=1000+i)
        
        # Quick local search
        improved = local_search(sol, max_iters=500)
        if len(improved) > best_len:
            best_len = len(improved)
            best_solution = improved
    
    return [[r, m] for r, m in best_solution]
