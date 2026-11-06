import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    moduli = list(range(2, 129))
    N = len(moduli)  # 127 moduli: 2 to 128 inclusive

    # Precompute gcd table
    gcd_table = [[0]*129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def check_pair_disjoint(r1, m1, r2, m2):
        """Check if residue class r1 mod m1 is disjoint from r2 mod m2.
        They are disjoint iff r1 mod g != r2 mod g where g = gcd(m1, m2)."""
        g = gcd_table[m1][m2]
        if g == 1:
            return True  # gcd=1 means they are always disjoint (by CRT, they intersect but we want disjoint... wait)
        # Actually: two residue classes r1 mod m1 and r2 mod m2 are DISJOINT iff r1 mod g != r2 mod g
        # where g = gcd(m1, m2). If g=1, then r1 mod 1 = 0 and r2 mod 1 = 0, so they are NEVER disjoint?
        # Let me reconsider: CRT says the system x ≡ r1 (mod m1), x ≡ r2 (mod m2) has a solution
        # iff r1 ≡ r2 (mod gcd(m1,m2)). So the classes INTERSECT iff r1 mod g == r2 mod g.
        # Therefore they are DISJOINT iff r1 mod g != r2 mod g.
        # If g=1, r1 mod 1 = 0, r2 mod 1 = 0, so r1 mod g == r2 mod g, meaning they INTERSECT.
        # So if g=1, they are NOT disjoint.
        return (r1 % g) != (r2 % g)

    def count_conflicts_for_residue(r, m, existing):
        """Count how many existing pairs conflict with (r, m)."""
        count = 0
        for r_k, m_k in existing:
            g = gcd_table[m][m_k]
            if (r % g) == (r_k % g):
                count += 1
        return count

    def find_valid_residue(m, existing):
        """Find a valid residue for modulus m given existing list of (r, m) pairs."""
        for r in range(m):
            if count_conflicts_for_residue(r, m, existing) == 0:
                return r
        return None

    def try_add_unselected(sol, order=None):
        """Try to add any unselected modulus with a valid residue."""
        if order is None:
            order = moduli
        selected_set = set(m for _, m in sol)
        new_sol = list(sol)
        for m in order:
            if m in selected_set:
                continue
            r = find_valid_residue(m, new_sol)
            if r is not None:
                new_sol.append([r, m])
                selected_set.add(m)
        return new_sol

    def local_search_minconflicts(sol, time_budget=None):
        """MinConflicts-style local search to improve residue assignments and add moduli."""
        if time_budget is None:
            time_budget = deadline - 2.0
        
        sol = list(sol)
        selected_moduli = [m for _, m in sol]
        selected_set = set(selected_moduli)
        
        # Build conflict count for each existing pair
        def get_conflict_counts(pairs):
            counts = [0] * len(pairs)
            for i in range(len(pairs)):
                r_i, m_i = pairs[i]
                for j in range(len(pairs)):
                    if i == j:
                        continue
                    r_j, m_j = pairs[j]
                    g = gcd_table[m_i][m_j]
                    if (r_i % g) == (r_j % g):
                        counts[i] += 1
            return counts
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            counts = get_conflict_counts(sol)
            if max(counts) == 0:
                break  # No conflicts
            
            # Find variable with max conflicts
            max_conf = max(counts)
            conflicted = [i for i in range(len(sol)) if counts[i] > 0]
            if not conflicted:
                break
            
            i = random.choice(conflicted)
            r_old, m_old = sol[i]
            
            # Try all residues for this modulus, pick the one with min conflicts
            best_r = r_old
            best_conf = counts[i]
            for r in range(m_old):
                conf = count_conflicts_for_residue(r, m_old, [(sol[j][0], sol[j][1]) for j in range(len(sol)) if j != i])
                if conf < best_conf:
                    best_conf = conf
                    best_r = r
                    if conf == 0:
                        break
            
            sol[i] = [best_r, m_old]
        
        # Try to add more moduli
        new_sol = try_add_unselected(sol)
        if len(new_sol) > len(sol):
            sol = new_sol
        
        return sol

    def local_search_swaps(sol, time_budget=None):
        """Local search with swap operators."""
        if time_budget is None:
            time_budget = deadline - 2.0
        
        sol = list(sol)
        
        for _ in range(200):
            if time.time() > time_budget:
                break
            improved = False
            
            # Operator 1: try adding unselected moduli
            new_sol = try_add_unselected(sol)
            if len(new_sol) > len(sol):
                sol = new_sol
                improved = True
            
            if not improved:
                # Operator 2: 1-for-1 swap
                for i in range(len(sol)):
                    if time.time() > time_budget:
                        break
                    sol_copy = sol[:i] + sol[i+1:]
                    new_sol = try_add_unselected(sol_copy)
                    if len(new_sol) > len(sol):
                        sol = new_sol
                        improved = True
                        break
            
            if not improved:
                # Operator 3: 2-for-1 swap
                for i in range(len(sol)):
                    if time.time() > time_budget:
                        break
                    for j in range(i+1, len(sol)):
                        if time.time() > time_budget:
                            break
                        sol_copy = sol[:i] + sol[i+1:j] + sol[j+1:]
                        new_sol = try_add_unselected(sol_copy)
                        if len(new_sol) > len(sol):
                            sol = new_sol
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return sol

    # ---- Main optimization ----
    
    best_solution = []
    
    # 1. Greedy with various orderings
    def construct_greedy_order(order):
        """Greedy: try to add moduli in the given order, picking a valid residue."""
        selected = []
        for m in order:
            if not selected:
                selected.append([0, m])
                continue
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([r, m])
        return selected
    
    # Precompute constraint score
    def constraint_score(m):
        score = 0
        for other in moduli:
            if other != m and gcd(m, other) > 1:
                score += 1
        return score
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: -x.bit_count()),
        sorted(moduli, key=lambda x: -constraint_score(x)),
        sorted(moduli, key=lambda x: constraint_score(x)),
        sorted(moduli, key=lambda x: x),
        sorted(moduli, key=lambda x: -x),
    ]
    
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        sol = construct_greedy_order(order)
        sol = local_search_minconflicts(sol, deadline - 3.0)
        sol = local_search_swaps(sol, deadline - 2.0)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # 2. Randomized restarts
    for _ in range(300):
        if time.time() > deadline - 5.0:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = construct_greedy_order(random_order)
        sol = local_search_minconflicts(sol, deadline - 3.0)
        sol = local_search_swaps(sol, deadline - 2.0)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # 3. Final polish on best solution
    if time.time() < deadline - 3.0:
        best_solution = local_search_minconflicts(best_solution, deadline - 1.0)
        best_solution = local_search_swaps(best_solution, deadline - 0.5)
    
    # Final safety: ensure non-empty
    if not best_solution:
        best_solution = [[0, 2]]
    
    return best_solution
