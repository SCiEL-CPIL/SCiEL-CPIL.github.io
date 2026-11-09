import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _count_conflicts(residues, moduli):
    """Count the number of conflicting pairs."""
    n = len(residues)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not _check_pair(residues[i], moduli[i], residues[j], moduli[j]):
                conflicts += 1
    return conflicts

def _local_search(residues, moduli, deadline):
    """
    Local search to minimize conflicts.
    Returns the best residues found.
    """
    n = len(residues)
    best_residues = residues[:]
    best_conflicts = _count_conflicts(best_residues, moduli)
    
    if best_conflicts == 0:
        return best_residues

    current_residues = residues[:]
    current_conflicts = best_conflicts
    
    # Precompute gcds for speed
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Identify conflicting pairs
    def get_conflicts(res):
        confs = []
        for i in range(n):
            for j in range(i + 1, n):
                if res[i] % gcds[i][j] == res[j] % gcds[i][j]:
                    confs.append((i, j))
        return confs

    current_confs = get_conflicts(current_residues)
    
    step = 0
    while time.time() < deadline:
        if not current_confs:
            return current_residues
            
        # Pick a random conflict
        i, j = random.choice(current_confs)
        
        # Try to fix by changing residue of i or j
        # Prefer changing the one with more conflicts or smaller modulus (more flexibility?)
        # Actually, just try changing i first
        old_r_i = current_residues[i]
        
        # Find a new residue for i that resolves conflict with j
        # and ideally doesn't create new conflicts
        best_new_r = None
        best_score = float('inf')
        
        # Sample a few random residues or try all if modulus is small
        # Modulus is at most 128, so trying all is fine
        for new_r in range(moduli[i]):
            if new_r == old_r_i:
                continue
            # Check if it resolves conflict with j
            if new_r % gcds[i][j] == current_residues[j] % gcds[i][j]:
                continue
            
            # Estimate new conflicts
            # We remove the conflict (i,j) if it was there.
            # We check against all other k
            score = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if new_r % gcds[i][k] == current_residues[k] % gcds[i][k]:
                    score += 1
            
            if score < best_score:
                best_score = score
                best_new_r = new_r
                
        if best_new_r is not None:
            current_residues[i] = best_new_r
            # Recalculate conflicts
            # Optimization: Only update conflicts involving i
            # But for simplicity and given n=64, full recalc is O(n^2) = 4096 ops.
            # 10000 iterations * 4096 = 40M ops. Python might be slow.
            # Let's optimize conflict counting.
            
            # Actually, let's just count conflicts involving i and j
            # This is tricky. Let's stick to full count but limit iterations or optimize inner loop.
            # With n=64, n^2/2 = 2016 pairs.
            # 2016 gcd checks per iteration.
            # 90 seconds allows for ~100k-200k iterations easily in C, but Python is slower.
            # Let's assume 10k iterations is safe.
            
            current_confs = get_conflicts(current_residues)
            current_conflicts = len(current_confs)
            
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_residues = current_residues[:]
                if best_conflicts == 0:
                    return best_residues
        else:
            # Try changing j
            old_r_j = current_residues[j]
            best_new_r = None
            best_score = float('inf')
            for new_r in range(moduli[j]):
                if new_r == old_r_j:
                    continue
                if new_r % gcds[i][j] == current_residues[i] % gcds[i][j]:
                    continue
                
                score = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if new_r % gcds[j][k] == current_residues[k] % gcds[j][k]:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_new_r = new_r

            if best_new_r is not None:
                current_residues[j] = best_new_r
                current_confs = get_conflicts(current_residues)
                current_conflicts = len(current_confs)
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_residues = current_residues[:]
                    if best_conflicts == 0:
                        return best_residues
            else:
                # Stuck, random restart or random move
                idx = random.randint(0, n - 1)
                current_residues[idx] = random.randint(0, moduli[idx] - 1)
                current_confs = get_conflicts(current_residues)
                current_conflicts = len(current_confs)
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_residues = current_residues[:]

    return best_residues

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Strategy: Use all even moduli
    moduli = [m for m in range(2, MAXIMUM + 1) if m % 2 == 0]
    
    # Initial residues: m/2
    residues = [m // 2 for m in moduli]
    
    best_residues = None
    best_conflicts = float('inf')
    
    # Run local search with a few random restarts if needed
    # But m/2 is a good start.
    
    # Check initial conflicts
    init_conflicts = _count_conflicts(residues, moduli)
    if init_conflicts == 0:
        return [[r, m] for r, m in zip(residues, moduli)]
        
    # Run local search
    result_residues = _local_search(residues, moduli, deadline)
    
    # Verify result
    final_conflicts = _count_conflicts(result_residues, moduli)
    
    if final_conflicts == 0:
        return [[r, m] for r, m in zip(result_residues, moduli)]
    
    # If local search didn't find a perfect solution, try a greedy backtracking on a subset
    # Or just return the best found, but it must be valid.
    # The task requires a valid list. If we can't find a valid one for all 64, we must reduce the set.
    
    # Greedy construction on the set of even moduli
    # Sort moduli by "constraint tightness" or just iterate
    solution = []
    used_moduli = set()
    
    # Try to add moduli one by one
    # We already have a near-solution. Let's try to repair it by dropping conflicting moduli.
    # Or simpler: Greedy from scratch with the good residue hints?
    
    # Let's try a greedy approach that prioritizes moduli with fewer constraints?
    # Actually, the local search result is likely very close.
    # Let's identify which moduli are involved in conflicts and drop them if necessary.
    
    # Better: Re-run greedy on the even moduli, using the local search residues as a bias?
    # No, let's just do a robust greedy that backtracks if stuck.
    
    # Given the high probability of success for even moduli, if local search fails,
    # it's likely due to a local minimum.
    
    # Let's try a simple greedy:
    # Sort moduli. For each, find a residue that works with all previous.
    # If none, skip.
    
    greedy_solution = []
    for m in moduli:
        found = False
        for r in range(m):
            ok = True
            for r_prev, m_prev in greedy_solution:
                if r % gcd(m, m_prev) == r_prev % gcd(m, m_prev):
                    ok = False
                    break
            if ok:
                greedy_solution.append([r, m])
                found = True
                break
        if not found:
            # Try random residues?
            # Or just skip
            pass
    
    if len(greedy_solution) > 0:
        # Check if this greedy solution is better than the local search one (which might be invalid)
        # Local search result is valid only if conflicts==0.
        # If local search failed (conflicts>0), we must use a valid subset.
        # The greedy solution is valid by construction.
        
        # Compare sizes? Local search likely found 64 or close.
        # If local search found 0 conflicts, we return it.
        # If not, we return the greedy solution.
        
        # But wait, if local search found 0 conflicts, we already returned.
        # So here we know local search failed.
        # Is greedy better than the partial local search?
        # The local search result is NOT a valid solution if conflicts > 0.
        # So we MUST return a valid solution.
        # Greedy is valid.
        
        return greedy_solution
        
    # Fallback: Return a known small valid set
    # e.g. [(0, 2)]
    return [[0, 2]]

if __name__ == "__main__":
    res = construct()
    print(len(res))
