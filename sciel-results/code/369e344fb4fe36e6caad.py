import math
import time

def construct():
    moduli = list(range(20, 201))
    weights = {m: 1.0 / m for m in moduli}
    sorted_moduli = sorted(moduli, key=lambda m: -weights[m])
    
    gcds = {}
    for i in range(len(sorted_moduli)):
        for j in range(i + 1, len(sorted_moduli)):
            m1, m2 = sorted_moduli[i], sorted_moduli[j]
            g = math.gcd(m1, m2)
            if g > 1:
                gcds[(m1, m2)] = g
                gcds[(m2, m1)] = g

    best_score = 0.0
    best_residues = {}
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for safety

    # Precompute potential conflicts for faster checking?
    # For each m, list of (m_other, gcd)
    conflicts = {m: [] for m in sorted_moduli}
    for (m1, m2), g in gcds.items():
        conflicts[m1].append((m2, g))
        conflicts[m2].append((m1, g))

    def is_compatible(m, r, current_moduli, current_residues):
        for m_other, g in conflicts[m]:
            if m_other in current_residues:
                if r % g == current_residues[m_other] % g:
                    return False
        return True

    def solve(idx, score, current_moduli, current_residues):
        nonlocal best_score, best_residues
        
        if time.time() - start_time > time_limit:
            return

        if idx == len(sorted_moduli):
            if score > best_score:
                best_score = score
                best_residues = dict(current_residues)
            return

        m = sorted_moduli[idx]
        
        # Bound
        # Simple bound: score + sum of weights of remaining
        # This is O(N) each time. Precompute suffix sums.
        
        # Try Include
        # Find valid residue
        found = False
        for r in range(m):
            if is_compatible(m, r, current_moduli, current_residues):
                current_moduli.append(m)
                current_residues[m] = r
                solve(idx + 1, score + weights[m], current_moduli, current_residues)
                current_moduli.pop()
                del current_residues[m]
                found = True
                break # Only try one residue for B&B speed? 
                # Trying all residues increases branching factor significantly.
                # We should pick the "best" residue.
                # For now, first valid is a heuristic.
        
        # Try Exclude
        solve(idx + 1, score, current_moduli, current_residues)

    # Precompute suffix sums for bound
    suffix_sums = [0] * (len(sorted_moduli) + 1)
    for i in range(len(sorted_moduli) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[sorted_moduli[i]]

    # Modify solve to use bound
    def solve_bnb(idx, score, current_moduli, current_residues):
        nonlocal best_score, best_residues
        
        if time.time() - start_time > time_limit:
            return

        # Bound
        if score + suffix_sums[idx] <= best_score:
            return

        if idx == len(sorted_moduli):
            if score > best_score:
                best_score = score
                best_residues = dict(current_residues)
            return

        m = sorted_moduli[idx]
        
        # Try Include
        # Heuristic: Try residues that are "least conflicting"
        # Or just first valid.
        for r in range(m):
            if is_compatible(m, r, current_moduli, current_residues):
                current_moduli.append(m)
                current_residues[m] = r
                solve_bnb(idx + 1, score + weights[m], current_moduli, current_residues)
                current_moduli.pop()
                del current_residues[m]
                break

        # Try Exclude
        solve_bnb(idx + 1, score, current_moduli, current_residues)

    solve_bnb(0, 0.0, [], {})
    
    # If no solution found (shouldn't happen), fallback to greedy
    if not best_residues:
        # Greedy fallback
        selected = []
        for m in range(20, 201):
            for r in range(m):
                conflict = False
                for (m2, r2) in selected:
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        conflict = True
                        break
                if not conflict:
                    selected.append((m, r))
                    break
        return [[m, r] for m, r in selected]
        
    return [[m, best_residues[m]] for m in best_residues]
