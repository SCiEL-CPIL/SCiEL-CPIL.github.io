import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    all_moduli = list(range(20, 201))
    n = len(all_moduli)
    
    # Precompute gcd table for speed
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = all_moduli[i]
        for j in range(n):
            gcd_table[i][j] = math.gcd(mi, all_moduli[j])
    
    def get_valid_residues(m_idx, selected):
        """Return list of residues r in [0, m) compatible with all selected (m_idx, r)."""
        m = all_moduli[m_idx]
        valid = []
        for r in range(m):
            ok = True
            for sel_idx, r_old in selected:
                g = gcd_table[m_idx][sel_idx]
                if g > 1 and r % g == r_old % g:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def score_of(selected):
        return sum(1.0 / all_moduli[m_idx] for m_idx, r in selected)
    
    def greedy(order, rng=None):
        """Process moduli in given order, pick a valid residue for each."""
        selected = []
        for m_idx in order:
            if time.time() > deadline:
                break
            valid_rs = get_valid_residues(m_idx, selected)
            if valid_rs:
                if rng is not None:
                    r = rng.choice(valid_rs)
                else:
                    r = valid_rs[0]
                selected.append((m_idx, r))
        return selected
    
    best_score = 0.0
    best_solution = []
    
    rng = random.Random(42)
    
    # Initial greedy in increasing order (deterministic)
    order = list(range(n))
    sel = greedy(order)
    best_score = score_of(sel)
    best_solution = sel
    
    # Random restarts with shuffled order and random residue choice
    for trial in range(1000):
        if time.time() > deadline:
            break
        order = list(range(n))
        rng.shuffle(order)
        sel = greedy(order, rng)
        s = score_of(sel)
        if s > best_score:
            best_score = s
            best_solution = sel
    
    return [[all_moduli[m_idx], r] for m_idx, r in best_solution]
