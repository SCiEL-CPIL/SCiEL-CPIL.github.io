import math
import random
import time

# Precompute gcds
MAX_M = 128
MODULI = list(range(2, MAX_M + 1))
N = len(MODULI)
GCD = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        GCD[i][j] = math.gcd(MODULI[i], MODULI[j])

def _find_valid_residue_fast(m_idx, existing_pairs):
    """
    Find a valid residue for MODULI[m_idx] given existing_pairs.
    existing_pairs is a list of (r, m_idx) tuples.
    """
    m = MODULI[m_idx]
    # Check if any constraints exist
    has_constraint = False
    for r_i, m_i_idx in existing_pairs:
        if GCD[m_idx][m_i_idx] > 1:
            has_constraint = True
            break
            
    if not has_constraint:
        return 0 # Or random

    forbidden = [False] * m
    for r_i, m_i_idx in existing_pairs:
        g = GCD[m_idx][m_i_idx]
        if g > 1:
            rem = r_i % g
            for r in range(rem, m, g):
                forbidden[r] = True
                
    for r in range(m):
        if not forbidden[r]:
            return r
    return None

def _greedy_pass_fast(order):
    """
    order is a list of indices into MODULI.
    Returns list of (r, m_idx) tuples.
    """
    pairs = []
    for m_idx in order:
        r = _find_valid_residue_fast(m_idx, pairs)
        if r is not None:
            pairs.append((r, m_idx))
    return pairs

def _improve_fast(pairs, deadline):
    """
    Try to add any missing moduli.
    pairs is a list of (r, m_idx).
    """
    current_set = set(m_idx for _, m_idx in pairs)
    all_indices = set(range(N))
    missing = list(all_indices - current_set)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        random.shuffle(missing)
        for m_idx in missing:
            if m_idx in current_set:
                continue
            r = _find_valid_residue_fast(m_idx, pairs)
            if r is not None:
                pairs.append((r, m_idx))
                current_set.add(m_idx)
                changed = True
                # Remove from missing? Not strictly necessary if we check current_set
                # But to avoid re-checking, we can manage the list.
                # Since we shuffle, we might check same ones.
                # It's okay for small N.
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_pairs = []
    best_len = 0
    
    # 1. Deterministic passes
    # Natural order
    order = list(range(N))
    pairs = _greedy_pass_fast(order)
    pairs = _improve_fast(pairs, deadline)
    if len(pairs) > best_len:
        best_pairs = pairs
        best_len = len(pairs)
        
    # Reverse order
    order = list(range(N-1, -1, -1))
    pairs = _greedy_pass_fast(order)
    pairs = _improve_fast(pairs, deadline)
    if len(pares) > best_len:
        best_pairs = pairs
        best_len = len(pairs)
        
    # 2. Randomized passes
    while time.time() < deadline - 2.0:
        random.shuffle(order)
        pairs = _greedy_pass_fast(order)
        pairs = _improve_fast(pairs, deadline)
        if len(pairs) > best_len:
            best_pairs = pairs
            best_len = len(pairs)
            
    # Convert to required format: list of [residue, modulus]
    result = [[r, MODULI[m_idx]] for r, m_idx in best_pairs]
    return result
