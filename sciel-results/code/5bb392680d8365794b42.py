import time
from math import gcd
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    moduli = list(range(20, 201))
    n = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd for all pairs
    gcd_table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Sort moduli by 1/m descending (i.e., m ascending)
    sorted_mods = sorted(moduli)  # 20, 21, ..., 200
    sorted_idx = [mod_idx[m] for m in sorted_mods]
    
    # Precompute 1/m as float for quick upper bound
    inv_m = [1.0 / m for m in sorted_mods]
    # Suffix sum of 1/m for upper bound: suffix[i] = sum(inv_m[i:])
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + inv_m[i]
    
    def feasible(m_idx, m, existing):
        """Check if there exists a residue r in [0, m) that doesn't conflict with existing.
        existing: list of (m_idx2, m2, r2)
        Returns True if feasible, False otherwise.
        """
        if not existing:
            return True
        m_val = m
        # For each existing (m2, r2), forbidden residues are r ≡ r2 (mod gcd(m, m2))
        # Build a boolean array of forbidden residues
        forbidden = np.zeros(m_val, dtype=np.uint8)
        for (mi2, m2, r2) in existing:
            g = gcd_table[m_idx][mi2]
            if g == m_val:
                # All residues forbidden
                return False
            # Mark residues r where r ≡ r2 (mod g)
            start = r2 % g
            forbidden[start::g] = 1
        # Check if any residue is not forbidden
        if np.any(forbidden == 0):
            return True
        return False
    
    def find_residues(m_idx, m, existing):
        """Find all valid residues for m given existing set."""
        m_val = m
        forbidden = np.zeros(m_val, dtype=np.uint8)
        for (mi2, m2, r2) in existing:
            g = gcd_table[m_idx][mi2]
            if g == m_val:
                return []
            start = r2 % g
            forbidden[start::g] = 1
        valid = np.where(forbidden == 0)[0]
        return valid.tolist()
    
    # Branch and bound: decide for each modulus (in order of 1/m desc) whether to include it
    # State: position in sorted_mods, current set of included classes
    # Upper bound: current_score + suffix_sum[pos]
    
    best_score_frac = Fraction(0, 1)
    best_set = []
    
    def branch_and_bound(pos, current_set, current_score_float, current_score_frac):
        nonlocal best_score_frac, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound
        ub = current_score_float + suffix_sum[pos]
        if ub <= float(best_score_frac) + 1e-12:
            return
        
        if pos == n:
            if current_score_frac > best_score_frac:
                best_score_frac = current_score_frac
                best_set = [x[:] for x in current_set]
            return
        
        m = sorted_mods[pos]
        m_idx = sorted_idx[pos]
        
        # Try including m
        valid_residues = find_residues(m_idx, m, current_set)
        if valid_residues:
            # Pick a residue - use a heuristic: try multiple and see which leads to better results
            # For speed, try a few residues
            max_trials = min(len(valid_residues), 5)
            if max_trials > 0:
                # Try the first few residues
                for r in valid_residues[:max_trials]:
                    new_set = current_set + [[m, r]]
                    new_score_f = current_score_float + inv_m[pos]
                    new_score_frac = current_score_frac + Fraction(1, m)
                    branch_and_bound(pos + 1, new_set, new_score_f, new_score_frac)
                    if time.time() > deadline:
                        return
        
        # Try excluding m
        branch_and_bound(pos + 1, current_set, current_score_float, current_score_frac)
    
    # Initial greedy to get a good lower bound
    def greedy_init():
        current_set = []
        current_score_f = 0.0
        current_score_frac = Fraction(0, 1)
        for pos in range(n):
            m = sorted_mods[pos]
            m_idx = sorted_idx[pos]
            valid_residues = find_residues(m_idx, m, current_set)
            if valid_residues:
                r = valid_residues[0]
                current_set.append([m, r])
                current_score_f += inv_m[pos]
                current_score_frac += Fraction(1, m)
        return current_set, current_score_f, current_score_frac
    
    init_set, init_f, init_frac = greedy_init()
    best_score_frac = init_frac
    best_set = [x[:] for x in init_set]
    
    # Run branch and bound
    branch_and_bound(0, [], 0.0, Fraction(0, 1))
    
    # If we didn't improve much, try a different greedy with different residue choices
    if time.time() < deadline - 10:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            current_set = []
            current_score_f = 0.0
            current_score_frac = Fraction(0, 1)
            for pos in range(n):
                m = sorted_mods[pos]
                m_idx = sorted_idx[pos]
                valid_residues = find_residues(m_idx, m, current_set)
                if valid_residues:
                    # Pick a different residue based on trial
                    idx = trial % len(valid_residues)
                    r = valid_residues[idx]
                    current_set.append([m, r])
                    current_score_f += inv_m[pos]
                    current_score_frac += Fraction(1, m)
            if current_score_frac > best_score_frac:
                best_score_frac = current_score_frac
                best_set = [x[:] for x in current_set]
    
    return best_set
