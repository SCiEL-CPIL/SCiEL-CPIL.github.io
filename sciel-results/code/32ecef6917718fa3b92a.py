import math
import time
import random
from collections import defaultdict

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN

    all_mods = list(range(2, MAX_MOD + 1))
    
    # Pre-compute gcd table
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)

    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0

    def _solve_mrv(moduli, t_limit, seed_val):
        random.seed(seed_val)
        n = len(moduli)
        if n == 0:
            return []
        
        # active_mods: list of indices into moduli that are not yet assigned
        # assigned: dict from index to residue
        assigned = {}
        unassigned = set(range(n))
        
        def backtrack():
            if time.time() > t_limit:
                return False
            if not unassigned:
                return True
            
            # MRV Heuristic: Pick the unassigned modulus with the fewest valid residues
            best_idx = None
            best_count = float('inf')
            best_residues = None
            
            for idx in unassigned:
                m = moduli[idx]
                valid_residues = []
                for r in range(m):
                    ok = True
                    for j, rj in assigned.items():
                        if not fast_disjoint(r, m, rj, moduli[j]):
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)
                
                if len(valid_residues) < best_count:
                    best_count = len(valid_residues)
                    best_idx = idx
                    best_residues = valid_residues
                    if best_count == 0:
                        break # No valid residues for this modulus, dead end
                    if best_count == 1:
                        break # Found a forced move, no need to check others
                
            if best_count == 0:
                return False
            
            # Try residues for the chosen modulus
            # LCV Heuristic: Try the residue that constrains other variables least?
            # For simplicity, we try them in random order or sorted order.
            # Let's try random order to break symmetry.
            random.shuffle(best_residues)
            
            for r in best_residues:
                assigned[best_idx] = r
                unassigned.remove(best_idx)
                
                if backtrack():
                    return True
                
                del assigned[best_idx]
                unassigned.add(best_idx)
            
            return False

        if backtrack():
            return [(assigned[i], moduli[i]) for i in range(n)]
        return None

    best_pairs = []
    best_len = 0
    
    # Try to solve the full set with MRV
    # Use a few seeds to handle randomness in tie-breaking or shuffling
    for seed in range(5):
        if time.time() > deadline - 1.0:
            break
        pairs = _solve_mrv(all_mods, t0 + 5.0, seed)
        if pairs is not None:
            if len(pairs) > best_len:
                best_pairs = pairs
                best_len = len(pairs)
            if best_len == len(all_mods):
                break

    # If full set failed, try greedy removal
    # Start with all moduli, try to solve. If fail, remove one and try again?
    # This is expensive. Instead, use a greedy construction that builds up.
    
    # Greedy construction with MRV-like selection?
    # Actually, if the full set is not solvable, we need to find the largest subset.
    # We can use a heuristic: Sort moduli by "difficulty" and add them one by one.
    
    # Difficulty heuristic: Moduli with more divisors are harder?
    # Or moduli that are composite?
    
    # Let's try a greedy approach where we add moduli in a specific order and keep the best found so far.
    # We'll use the _solve_mrv on the subset.
    
    # To find a good subset, we can try removing "hard" moduli.
    # But we don't know which are hard without solving.
    
    # Alternative: Randomized Greedy
    # 1. Shuffle moduli.
    # 2. Iterate through shuffled moduli.
    # 3. For each modulus, try to add it to the current set by solving the CSP for the new set.
    # 4. If it fails, skip it.
    
    # This is O(N * Time_Solve). Time_Solve might be large.
    
    # Let's stick to the MRV solver on the full set first.
    # If it fails, we try a greedy build.
    
    if best_len < len(all_mods):
        # Greedy build
        for seed in range(50):
            if time.time() > deadline - 0.5:
                break
            random.seed(seed)
            mods = all_mods[:]
            random.shuffle(mods)
            
            current_set = []
            for m in mods:
                # Try to add m to current_set
                # Create a list of moduli: current_set + [m]
                # Solve CSP for this list
                # This is expensive if we do it for every m.
                
                # Better: Just try to find a residue for m that is disjoint from all in current_set.
                # This is the greedy approach from the previous version.
                found = False
                r_candidates = list(range(m))
                random.shuffle(r_candidates)
                for r in r_candidates:
                    ok = True
                    for (r2, m2) in current_set:
                        if not fast_disjoint(r, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        current_set.append((r, m))
                        found = True
                        break
                if not found:
                    continue
            
            if len(current_set) > best_len:
                best_pairs = current_set
                best_len = len(current_set)
                if best_len == len(all_mods):
                    break

    if not best_pairs:
        best_pairs = [(0, 2)]
        
    return [[r, m] for r, m in best_pairs]
