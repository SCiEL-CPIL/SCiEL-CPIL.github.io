"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    """
    Return a list of [residue, modulus] pairs with distinct moduli in [2,128],
    canonical residues 0<=r<m, such that every pair of infinite residue classes
    is disjoint (verified by exact gcd criterion). Maximize the number of classes.
    
    Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does not divide (r1 - r2).
    """
    start_time = time.time()
    deadline = 85.0  # 5s margin before 90s kill
    
    def check_pair(r1, m1, r2, m2):
        """Check if two residue classes are disjoint."""
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def check_all(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                if not check_pair(r1, m1, r2, m2):
                    return False
        return True
    
    # Seed solution: powers of 2 with residue 2^k - 1
    def seed_solution():
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    best = seed_solution()
    best_count = len(best)
    
    all_moduli = list(range(2, 129))
    n_mods = len(all_moduli)
    mod_to_idx = {m: i for i, m in enumerate(all_moduli)}
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Key insight: two moduli with gcd=1 can NEVER be in a valid set
    # because gcd=1 divides everything, so (r1-r2) % 1 == 0 always
    # So we need a set of moduli where every pair has gcd > 1
    
    # For each prime p, find all moduli divisible by p
    # Within such a set, we can potentially have a valid configuration
    
    # Strategy: For each prime p, consider the subgraph of moduli all divisible by p
    # Then find maximum valid set within that subgraph using bitmask-based search
    
    # Precompute for each pair of moduli (i,j) with gcd>1:
    # the set of residues r for modulus i that are compatible with residue s for modulus j
    # i.e., (r - s) % gcd(m_i, m_j) != 0
    
    # For efficiency, precompute compatibility as bitmasks
    # For modulus m_i, residue r is a bit in a bitmask of size m_i
    
    # But moduli go up to 128, so bitmasks are at most 128 bits = 2 Python ints
    
    # Let's use a different approach: for each modulus m, precompute which residues
    # are "blocked" by each other modulus
    
    # Actually, let's use a cleaner approach:
    # For a fixed set of moduli S, we need to assign residues r_m for each m in S
    # such that for all m, n in S: (r_m - r_n) % gcd(m,n) != 0
    
    # This is a CSP. Let's solve it with backtracking + constraint propagation.
    
    # For each prime p, build the candidate set of moduli divisible by p
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    def solve_csp(moduli_list, time_limit):
        """
        Solve the CSP: assign residues to each modulus in moduli_list
        such that all pairs are disjoint.
        Returns list of [residue, modulus] or None.
        """
        t0 = time.time()
        n = len(moduli_list)
        if n == 0:
            return []
        
        # Order moduli by increasing size (more constrained first)
        # Actually, order by decreasing number of constraints (more divisors = more constraints)
        def constraint_count(m):
            """Number of other moduli in moduli_list that share a factor with m."""
            count = 0
            for m2 in moduli_list:
                if m2 != m and math.gcd(m, m2) > 1:
                    count += 1
            return count
        
        ordered = sorted(moduli_list, key=lambda m: -constraint_count(m))
        
        # Backtracking with forward checking
        assignment = {}  # modulus -> residue
        used_residues = {}  # modulus -> set of used residues (for forward checking)
        
        def backtrack(idx):
            if time.time() - t0 > time_limit:
                return False
            if idx == n:
                return True
            
            m = ordered[idx]
            # Try residues in order of least constraining first
            # For each residue r, check compatibility with all assigned moduli
            for r in range(m):
                # Check compatibility with all already-assigned moduli
                ok = True
                for m2, r2 in assignment.items():
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False
        
        if backtrack(0):
            return [[assignment[m], m] for m in ordered]
        return None
    
    # For each prime, get the set of moduli divisible by it
    # Then try to solve the CSP on that set
    
    for p in primes:
        if time.time() > deadline - 5:
            break
        
        candidates = [m for m in all_moduli if m % p == 0]
        if len(candidates) <= best_count:
            continue
        
        # Try to solve on the full candidate set
        # But this might be too large; try subsets
        
        # First, try the full set with a time limit
        result = solve_csp(candidates, time_limit=2.0)
        if result is not None and len(result) > best_count:
            best = result
            best_count = len(result)
        
        if time.time() > deadline - 10:
            break
        
        # If the full set didn't work, try to find a large subset
        # Use a greedy approach to build a subset, then solve
        if len(candidates) > best_count + 2:
            # Greedy: pick moduli one by one, preferring those with more shared factors
            def shared_factor_score(m, current_set):
                """Score how well m fits with current_set."""
                score = 0
                for m2 in current_set:
                    g = math.gcd(m, m2)
                    if g > 1:
                        score += g
                return score
            
            current_set = []
            remaining = list(candidates)
            
            while remaining and len(current_set) < min(len(candidates), best_count + 5):
                if time.time() > deadline - 5:
                    break
                # Pick the modulus with highest shared factor score
                best_m = None
                best_score = -1
                for m in remaining:
                    s = shared_factor_score(m, current_set) if current_set else 0
                    if s > best_score:
                        best_score = s
                        best_m = m
                if best_m is not None:
                    current_set.append(best_m)
                    remaining.remove(best_m)
            
            if len(current_set) > best_count:
                result = solve_csp(current_set, time_limit=3.0)
                if result is not None and len(result) > best_count:
                    best = result
                    best_count = len(result)
    
    # Also try combinations of moduli not all sharing a single prime
    # But this is harder. Let's try a different approach:
    # Use the fact that if all moduli are even, they all share factor 2
    
    # Try all even moduli
    if time.time() < deadline - 10:
        even_moduli = [m for m in all_moduli if m % 2 == 0]
        result = solve_csp(even_moduli, time_limit=5.0)
        if result is not None and len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Try all multiples of 2 or 3 (union)
    if time.time() < deadline - 5:
        mod_2_or_3 = [m for m in all_moduli if m % 2 == 0 or m % 3 == 0]
        result = solve_csp(mod_2_or_3, time_limit=5.0)
        if result is not None and len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Final local search: try to improve best by adding moduli
    def try_add_modulus(current_classes, m, time_limit):
        """Try to add modulus m to current_classes with some residue."""
        t0 = time.time()
        for r in range(m):
            if time.time() - t0 > time_limit:
                return None
            if all(check_pair(r, m, r2, m2) for r2, m2 in current_classes):
                return [r, m]
        return None
    
    def local_search_improve(initial_classes, time_limit):
        """Improve a solution by trying to add moduli."""
        classes = [list(c) for c in initial_classes]
        used_moduli = set(m for _, m in classes)
        t0 = time.time()
        improved = True
        
        while improved and time.time() - t0 < time_limit:
            improved = False
            for m in range(2, 129):
                if time.time() - t0 > time_limit:
                    break
                if m in used_moduli:
                    continue
                new_class = try_add_modulus(classes, m, 0.1)
                if new_class is not None:
                    classes.append(new_class)
                    used_moduli.add(m)
                    improved = True
        
        return classes
    
    if time.time() < deadline - 3:
        improved = local_search_improve(best, time_limit=5.0)
        if len(improved) > best_count:
            best = improved
            best_count = len(improved)
    
    return best
# EVOLVE-BLOCK-END
