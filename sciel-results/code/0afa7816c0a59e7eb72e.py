"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _gcd(a, b):
    return math.gcd(a, b)

def _is_disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    """
    Uses Backtracking with Forward Checking to find a large set of pairwise disjoint residue classes.
    """
    start_time = time.time()
    time_limit = 55.0
    
    moduli = list(range(2, 129))
    # Precompute gcds
    gcd_table = {}
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            gcd_table[(moduli[i], moduli[j])] = _gcd(moduli[i], moduli[j])
            
    # Sort moduli by "constrainedness" - those with more small gcds are harder?
    # Or just try to add in order of increasing modulus?
    # Actually, let's try to find a large set first, then optimize.
    
    # Heuristic: Start with a set of powers of 2, then try to add others.
    # But let's do a systematic backtracking.
    
    # State: list of (r, m)
    best_solution = []
    best_size = 0
    
    # Order moduli to maximize success
    # Moduli with many divisors are more constrained?
    # Let's just iterate and use backtracking.
    
    def backtrack(current_set, remaining_moduli):
        nonlocal best_solution, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = list(current_set)
            
        if not remaining_moduli:
            return
            
        # Pick a modulus from remaining
        # Heuristic: Pick the one with the most conflicts with current_set?
        # Or just the first one.
        # To speed up, we can try to add ALL possible moduli if they fit.
        
        # Try to add each remaining modulus
        # We can prune if we can't add any more?
        
        # Optimization: If we can add a modulus, we should try to add it.
        # If we can't, we skip it.
        # This is greedy. Backtracking helps if skipping one allows adding two others?
        # Actually, adding a modulus never makes it harder to add others?
        # No, adding a modulus ADDS constraints. So it makes it HARDER to add others.
        # So we should be careful.
        
        # Strategy: Try to add a modulus. If it fails, try the next.
        # If we reach the end, we have a maximal set for this path.
        
        # To find the MAXIMUM set, we need to explore different choices.
        # But 127 moduli is too many for full backtracking.
        
        # Alternative: Local Search on the set of moduli.
        # But we already have that.
        
        # Let's try a different heuristic:
        # 1. Find a large set of moduli $S$ such that $\gcd(m, n) > 1$ for all $m, n \in S$.
        # 2. Solve CSP for $S$.
        
        pass

    # Since full backtracking is too slow, we use a two-phase approach:
    # Phase 1: Find a large pairwise non-coprime set $S$ using greedy + local search on the moduli only.
    # Phase 2: Solve the residue assignment CSP for $S$ using backtracking.
    
    # Phase 1: Maximize $|S|$ subject to $\gcd(m, n) > 1$.
    # This is Max Clique in the graph $G$ where $E$ connects non-coprime pairs.
    # Greedy Max Clique:
    def greedy_max_clique():
        vertices = list(range(2, 129))
        # Order by degree (descending)
        def degree(v):
            return sum(1 for u in vertices if u != v and _gcd(v, u) > 1)
        
        vertices.sort(key=degree, reverse=True)
        
        clique = []
        for v in vertices:
            # Check if v can be added to clique
            can_add = True
            for u in clique:
                if _gcd(v, u) == 1:
                    can_add = False
                    break
            if can_add:
                clique.append(v)
        
        return clique

    # Local search on the clique
    def local_search_clique(clique):
        clique = set(clique)
        improved = True
        while improved:
            improved = False
            # Try to add
            for v in range(2, 129):
                if v not in clique:
                    if all(_gcd(v, u) > 1 for u in clique):
                        clique.add(v)
                        improved = True
            # Try to swap
            # Remove one, add two?
            # Keep it simple for now.
        return list(clique)

    S = greedy_max_clique()
    S = local_search_clique(S)
    
    # Phase 2: Solve CSP for S
    # Variables: residues $r_m$ for $m \in S$.
    # Domain: $0 \dots m-1$.
    # Constraints: $r_m \not\equiv r_n \pmod{\gcd(m, n)}$.
    
    # Backtracking solver for CSP
    def solve_csp(moduli_set):
        # Order variables by domain size (smallest first) or constraint degree
        # Domain size is $m$. Smaller $m$ is better.
        vars = sorted(moduli_set, key=lambda x: x)
        
        assignment = {}
        
        def backtrack(idx):
            nonlocal start_time, time_limit
            if time.time() - start_time > time_limit:
                return None
            if idx == len(vars):
                return assignment.copy()
                
            m = vars[idx]
            # Try residues
            # Heuristic: Try residues that are "central" or random?
            # Let's try all, but with pruning.
            for r in range(m):
                # Check constraints with already assigned variables
                valid = True
                for j in range(idx):
                    m_j = vars[j]
                    r_j = assignment[m_j]
                    g = _gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        valid = False
                        break
                if valid:
                    assignment[m] = r
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    del assignment[m]
            return None
            
        return backtrack(0)

    # Try to solve for S
    solution = solve_csp(S)
    
    if solution:
        return [[r, m] for m, r in solution.items()]
    else:
        # If CSP fails, try removing some moduli from S
        # This is a fallback
        # For now, return the best we can get by greedy adding
        pass

    # Fallback: Greedy addition of full pairs
    # This is similar to the seed but with a better order
    chosen = []
    # Try moduli in order of increasing "constraint"
    # Sort by number of divisors?
    def num_divisors(n):
        return sum(1 for i in range(1, int(math.isqrt(n)) + 1) if n % i == 0) * 2 - (1 if math.isqrt(n)**2 == n else 0)
        
    # Actually, just use the seed logic if CSP fails
    moduli = list(range(2, 129))
    random.shuffle(moduli)
    for m in moduli:
        for r in range(m):
            if all(_is_disjoint(r, m, r2, m2) for r2, m2 in chosen):
                chosen.append((r, m))
                break
                
    return [[r, m] for r, m in chosen]

# EVOLVE-BLOCK-END
