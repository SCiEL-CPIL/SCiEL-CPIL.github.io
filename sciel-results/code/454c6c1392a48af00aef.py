"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Construct a maximum set of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two residue classes (r1, m1) and (r2, m2) are disjoint iff
    r1 != r2 (mod gcd(m1, m2)).
    
    Strategy:
    1. The best single-prime family is p=11, giving 11 classes with
       moduli 11, 22, ..., 121 and residues 0, 1, ..., 10.
    2. Try to extend this base solution by adding more moduli via
       a smart backtracking search that prunes early.
    3. If extension fails, return the base solution.
    """
    START = time.time()
    DEADLINE = START + 80  # Safety margin
    
    MAX_MOD = 128
    
    # Base solution: prime 11 family
    base_solution = [(i, 11 * (i + 1)) for i in range(11)]
    used_moduli = set(m for _, m in base_solution)
    
    # Available moduli not in base solution
    available = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
    
    # Precompute gcd with each base modulus for each available modulus
    # This speeds up the constraint checking
    base_gcds = {}
    for m in available:
        gcds = [math.gcd(m, bm) for _, bm in base_solution]
        base_gcds[m] = gcds
    
    def can_assign_residue(m, solution):
        """
        Check if there exists a residue r in [0, m) such that for all
        (r_i, m_i) in solution, r % gcd(m, m_i) != r_i % gcd(m, m_i).
        Returns the first valid residue or None.
        """
        if time.time() > DEADLINE:
            return None
        
        m_gcds = base_gcds.get(m, [math.gcd(m, bm) for _, bm in solution])
        
        # For each existing class, compute the forbidden residue mod g
        # We need r such that for all i: r % g_i != r_i % g_i
        # This is a system of modular inequalities.
        # Try residues 0 to m-1, but with early termination.
        
        for r in range(m):
            ok = True
            for idx, (r_i, m_i) in enumerate(solution):
                g = m_gcds[idx] if idx < len(m_gcds) else math.gcd(m, m_i)
                if r % g == r_i % g:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def try_extend(solution, available_moduli):
        """
        Recursively try to add more moduli to the solution.
        Uses greedy ordering: try moduli that are most constrained first.
        """
        if time.time() > DEADLINE:
            return solution
        
        if not available_moduli:
            return solution
        
        # Sort available moduli by number of prime factors (more factors = more constraints)
        # Actually, let's just try them in order and use backtracking
        # To keep it fast, we limit the depth and breadth
        
        for idx in range(len(available_moduli)):
            if time.time() > DEADLINE:
                return solution
            
            m = available_moduli[idx]
            
            # Quick check: can we assign a residue?
            r = can_assign_residue(m, solution)
            if r is not None:
                new_solution = solution + [(r, m)]
                remaining = available_moduli[:idx] + available_moduli[idx+1:]
                extended = try_extend(new_solution, remaining)
                if len(extended) > len(solution):
                    return extended
        
        return solution
    
    # Limit the search to keep it fast
    # Only try a subset of available moduli to avoid exponential blowup
    # Sort by modulus size - smaller moduli might be easier to fit
    available_sorted = sorted(available, key=lambda m: m)
    
    # Try extending with a limited set
    # Use a time-limited search
    extended = try_extend(base_solution, available_sorted[:50])
    
    if len(extended) > len(base_solution):
        result = [[r, m] for r, m in extended]
        # Verify the result
        n = len(result)
        valid = True
        for i in range(n):
            if time.time() > DEADLINE:
                break
            for j in range(i + 1, n):
                r1, m1 = result[i]
                r2, m2 = result[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return result
    
    # Return base solution
    return [[r, m] for r, m in base_solution]
# EVOLVE-BLOCK-END
