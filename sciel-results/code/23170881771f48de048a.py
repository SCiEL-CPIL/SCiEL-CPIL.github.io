"""Optimized odd covering system solver for moduli dividing 51975."""
import time
import random
import math

BASE = 51975

def _get_divisors():
    divs = []
    for i in range(1, int(math.isqrt(BASE)) + 1):
        if BASE % i == 0:
            if i > 1:
                divs.append(i)
            other = BASE // i
            if other != i and other > 1:
                divs.append(other)
    divs.sort()
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    moduli = _get_divisors()
    n = len(moduli)
    
    # Precompute for each modulus index, for each residue, the list of positions
    # This is memory intensive but allows fast gain calculations
    # For modulus m, residue r covers positions r, r+m, r+2m, ... < BASE
    # Number of positions: (BASE - 1 - r) // m + 1 if r < BASE else 0
    
    # For efficiency, store as lists for small moduli, compute on fly for large
    pos_lists = [None] * n
    for i, m in enumerate(moduli):
        if m <= 200:  # Cache small moduli
            pos_lists[i] = [list(range(r, BASE, m)) for r in range(m)]
    
    def build_coverage(residues):
        covered = bytearray(BASE)
        for i, m in enumerate(moduli):
            r = residues[i]
            if pos_lists[i] is not None:
                for pos in pos_lists[i][r]:
                    covered[pos] = 1
            else:
                for pos in range(r, BASE, m):
                    covered[pos] = 1
        return covered
    
    def count_uncovered(covered):
        return sum(1 for x in covered if x == 0)
    
    # Initial greedy solution
    def greedy(order):
        covered = bytearray(BASE)
        residues = [0] * n
        for i in order:
            m = moduli[i]
            best_r = 0
            best_gain = -1
            for r in range(m):
                if pos_lists[i] is not None:
                    gain = sum(1 for pos in pos_lists[i][r] if covered[pos] == 0)
                else:
                    gain = sum(1 for pos in range(r, BASE, m) if covered[pos] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[i] = best_r
            if pos_lists[i] is not None:
                for pos in pos_lists[i][best_r]:
                    covered[pos] = 1
            else:
                for pos in range(best_r, BASE, m):
                    covered[pos] = 1
        return residues
    
    best_residues = None
    best_score = BASE + 1
    
    # Try greedy with different orders
    orders = [
        list(range(n)),           # increasing
        list(range(n-1, -1, -1)), # decreasing
        sorted(range(n), key=lambda i: moduli[i]),  # same as increasing
    ]
    
    for order in orders:
        r = greedy(order)
        cov = build_coverage(r)
        score = count_uncovered(cov)
        if score < best_score:
            best_score = score
            best_residues = r[:]
    
    # Random starts
    for _ in range(30):
        r = [random.randint(0, moduli[i]-1) for i in range(n)]
        cov = build_coverage(r)
        score = count_uncovered(cov)
        if score < best_score:
            best_score = score
            best_residues = r[:]
    
    # Simulated annealing with efficient delta calculation
    # For a move (i, old_r -> new_r), the delta in uncovered count is:
    # - Positions that were only covered by modulus i with old_r become uncovered
    # - Positions that were uncovered become covered
    # We need to track for each position which moduli cover it
    
    # Alternative: maintain coverage count array
    # cov_count[pos] = number of moduli covering position pos
    # When we change modulus i from old_r to new_r:
    #   For pos in old_r class: cov_count[pos] -= 1, if it becomes 0, uncovered increases
    #   For pos in new_r class: cov_count[pos] += 1, if it was 0, uncovered decreases
    
    def init_cov_count(residues):
        cov_count = [0] * BASE
        for i, m in enumerate(moduli):
            r = residues[i]
            if pos_lists[i] is not None:
                for pos in pos_lists[i][r]:
                    cov_count[pos] += 1
            else:
                for pos in range(r, BASE, m):
                    cov_count[pos] += 1
        return cov_count
    
    current_residues = best_residues[:]
    current_cov_count = init_cov_count(current_residues)
    current_uncovered = sum(1 for c in current_cov_count if c == 0)
    
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.001
    
    while time.time() < deadline and temp > min_temp:
        i = random.randint(0, n - 1)
        m = moduli[i]
        old_r = current_residues[i]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
        
        # Compute delta
        delta = 0
        
        # Remove old residue class
        if pos_lists[i] is not None:
            for pos in pos_lists[i][old_r]:
                current_cov_count[pos] -= 1
                if current_cov_count[pos] == 0:
                    delta += 1  # becomes uncovered
        else:
            for pos in range(old_r, BASE, m):
                current_cov_count[pos] -= 1
                if current_cov_count[pos] == 0:
                    delta += 1
        
        # Add new residue class
        if pos_lists[i] is not None:
            for pos in pos_lists[i][new_r]:
                if current_cov_count[pos] == 0:
                    delta -= 1  # becomes covered
                current_cov_count[pos] += 1
        else:
            for pos in range(new_r, BASE, m):
                if current_cov_count[pos] == 0:
                    delta -= 1
                current_cov_count[pos] += 1
        
        new_uncovered = current_uncovered + delta
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_residues[i] = new_r
            current_uncovered = new_uncovered
            if current_uncovered < best_score:
                best_score = current_uncovered
                best_residues = current_residues[:]
        else:
            # Revert
            if pos_lists[i] is not None:
                for pos in pos_lists[i][new_r]:
                    current_cov_count[pos] -= 1
                    if current_cov_count[pos] == 0:
                        pass  # was uncovered, now still uncovered after revert
                for pos in pos_lists[i][old_r]:
                    current_cov_count[pos] += 1
            else:
                for pos in range(new_r, BASE, m):
                    current_cov_count[pos] -= 1
                for pos in range(old_r, BASE, m):
                    current_cov_count[pos] += 1
        
        temp *= cooling
    
    # Final local search
    if time.time() < deadline:
        # Rebuild cov_count from best_residues
        best_cov_count = init_cov_count(best_residues)
        best_uncovered = sum(1 for c in best_cov_count if c == 0)
        
        for i in range(n):
            if time.time() > deadline:
                break
            m = moduli[i]
            old_r = best_residues[i]
            for new_r in range(m):
                if new_r == old_r:
                    continue
                if time.time() > deadline:
                    break
                
                # Compute delta
                delta = 0
                if pos_lists[i] is not None:
                    for pos in pos_lists[i][old_r]:
                        best_cov_count[pos] -= 1
                        if best_cov_count[pos] == 0:
                            delta += 1
                    for pos in pos_lists[i][new_r]:
                        if best_cov_count[pos] == 0:
                            delta -= 1
                        best_cov_count[pos] += 1
                else:
                    for pos in range(old_r, BASE, m):
                        best_cov_count[pos] -= 1
                        if best_cov_count[pos] == 0:
                            delta += 1
                    for pos in range(new_r, BASE, m):
                        if best_cov_count[pos] == 0:
                            delta -= 1
                        best_cov_count[pos] += 1
                
                new_uncovered = best_uncovered + delta
                
                if new_uncovered < best_score:
                    best_score = new_uncovered
                    best_residues[i] = new_r
                    best_uncovered = new_uncovered
                else:
                    # Revert
                    if pos_lists[i] is not None:
                        for pos in pos_lists[i][new_r]:
                            best_cov_count[pos] -= 1
                        for pos in pos_lists[i][old_r]:
                            best_cov_count[pos] += 1
                    else:
                        for pos in range(new_r, BASE, m):
                            best_cov_count[pos] -= 1
                        for pos in range(old_r, BASE, m):
                            best_cov_count[pos] += 1
    
    return [[moduli[i], best_residues[i]] for i in range(n)]
