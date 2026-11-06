import time
import random
import numpy as np

BASE = 51975
_DIVS = []
for a in range(4):
    for b in range(3):
        for c in range(2):
            for d in range(2):
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N_DIVS = len(_DIVS)

# Precompute positions for each (m, r)
# _POS[i][r] is a numpy array of indices in 0..BASE-1
_POS = []
for m in _DIVS:
    lst = []
    for r in range(m):
        lst.append(np.arange(r, BASE, m, dtype=np.int32))
    _POS.append(lst)

def construct():
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # Initialize residues
    # Try a few random initializations or a structured one
    # Structured: r_m = 0 for all m
    residues = [0] * N_DIVS
    
    # Calculate initial coverage
    cover_count = np.zeros(BASE, dtype=np.int16)
    for i, m in enumerate(_DIVS):
        r = residues[i]
        cover_count[_POS[i][r]] += 1
    
    uncovered = np.sum(cover_count == 0)
    
    best_uncovered = uncovered
    best_residues = residues[:]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random modulus index
        i = rng.randrange(N_DIVS)
        m = _DIVS[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = rng.randrange(m)
        if new_r == old_r:
            continue
            
        # Calculate delta in uncovered count
        # Remove old class
        old_pos = _POS[i][old_r]
        new_pos = _POS[i][new_r]
        
        delta_uncovered = 0
        
        # For old positions:
        # If cover_count[pos] becomes 0, uncovered increases by 1
        # cover_count[pos] is at least 1 (since it was covered by this class)
        # We need to check if cover_count[pos] == 1
        # If yes, it becomes 0, so uncovered + 1
        # We can count how many pos in old_pos have cover_count[pos] == 1
        # But we must be careful if new_pos overlaps with old_pos?
        # The classes for a fixed modulus are disjoint. So old_pos and new_pos are disjoint.
        # So we can update independently.
        
        # Update for old_pos
        # count how many have cover_count == 1
        # We can use numpy: np.sum(cover_count[old_pos] == 1)
        # But we also need to update cover_count.
        
        # To be safe and fast, let's just do the updates and count.
        # However, doing numpy sum is fast.
        
        c_old = cover_count[old_pos]
        # Number of elements in old_pos that are exclusively covered by this class
        # i.e., cover_count[pos] == 1
        lost_covered = np.sum(c_old == 1)
        
        # Update cover_count for old_pos
        cover_count[old_pos] -= 1
        
        # Update for new_pos
        c_new = cover_count[new_pos]
        # Number of elements in new_pos that are currently uncovered
        # i.e., cover_count[pos] == 0
        gained_covered = np.sum(c_new == 0)
        
        # Update cover_count for new_pos
        cover_count[new_pos] += 1
        
        delta_uncovered = lost_covered - gained_covered
        
        # Acceptance
        if delta_uncovered < 0 or rng.random() < np.exp(-delta_uncovered / T):
            residues[i] = new_r
            uncovered += delta_uncovered
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues[:]
        else:
            # Revert
            cover_count[old_pos] += 1
            cover_count[new_pos] -= 1
            # uncovered remains same
            
        T *= cooling
        
    # Return best classes
    classes = []
    for i, m in enumerate(_DIVS):
        classes.append([m, best_residues[i]])
        
    return classes
