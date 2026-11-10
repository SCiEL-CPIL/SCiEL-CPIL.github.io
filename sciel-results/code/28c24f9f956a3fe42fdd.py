import math
import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    # Step 1: Select moduli M
    # Greedy: sort by 1/m desc, add if non-coprime with all existing
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0/m)
    
    M = []
    for m in moduli:
        compatible = True
        for m2 in M:
            if math.gcd(m, m2) == 1:
                compatible = False
                break
        if compatible:
            M.append(m)
            
    # Step 2: Assign residues
    # Use backtracking with forward checking
    # Order M by increasing number of constraints? Or just increasing m?
    # Let's try increasing m.
    
    def solve_residues(mods):
        n = len(mods)
        if n == 0:
            return {}
        
        # Heuristic: order by most constrained first (smallest m? or most neighbors?)
        # For now, just use the given order.
        residues = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            m = mods[idx]
            # Try residues
            # Optimization: only need to check against previous assigned
            for r in range(m):
                conflict = False
                for j in range(idx):
                    m2 = mods[j]
                    r2 = residues[j]
                    g = math.gcd(m, m2)
                    if (r % g) == (r2 % g):
                        conflict = True
                        break
                if not conflict:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            return False
        
        # Simple backtracking might be slow if n is large (100+).
        # But the constraints are loose.
        # Let's add a time check.
        
        # To speed up, we can use a greedy pass first, then backtrack only if needed.
        # Greedy pass:
        for i in range(n):
            m = mods[i]
            found = False
            for r in range(m):
                conflict = False
                for j in range(i):
                    m2 = mods[j]
                    r2 = residues[j]
                    if r2 is None: continue
                    g = math.gcd(m, m2)
                    if (r % g) == (r2 % g):
                        conflict = True
                        break
                if not conflict:
                    residues[i] = r
                    found = True
                    break
            if not found:
                # Greedy failed, try backtracking from here
                # This is complex to implement inline.
                # Let's just use a randomized local search for residues.
                pass
                
        # Check if greedy succeeded
        if all(r is not None for r in residues):
            return {mods[i]: residues[i] for i in range(n)}
            
        # If greedy failed, use local search
        # Initialize with random valid residues or greedy partial
        current_res = [0] * n
        # Try to fix conflicts
        for _ in range(1000):
            # Pick a random index with a conflict
            conflicts = []
            for i in range(n):
                for j in range(i):
                    if (current_res[i] % math.gcd(mods[i], mods[j])) == (current_res[j] % math.gcd(mods[i], mods[j])):
                        conflicts.append(i)
                        conflicts.append(j)
                        break
            if not conflicts:
                return {mods[i]: current_res[i] for i in range(n)}
            
            idx = random.choice(conflicts)
            m = mods[idx]
            # Try random new residue
            r_new = random.randint(0, m - 1)
            # Check if it reduces conflicts
            # Simple: just set it and hope
            current_res[idx] = r_new
            
        return None

    # Try to solve
    solution = solve_residues(M)
    
    if solution is None:
        # Fallback: remove lowest weight moduli until solvable
        # Or just return empty
        return []
        
    return [[m, r] for m, r in solution.items()]

