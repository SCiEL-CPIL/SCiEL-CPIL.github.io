"""
Edge-colourings of K16 in which every K4 sees five colours.

Strategy:
1. Start from a random 15-colouring or a structured one.
2. Use local search to fix violations (ensure every K4 has >= 5 colours).
3. Use local search to minimize the number of distinct colours used.
4. Return the best valid colouring found before the deadline.
"""

import time
import random
import sys
import numpy as np

DEADLINE = 118.0  # safety margin before 120s kill

def _edge_index(i, j):
    """Map pair (i,j), 0<=i<j<16, to index in the 120-element list."""
    # i * 15 - i * (i - 1) // 2 + j - 1 - i is the standard formula?
    # Let's verify:
    # i=0: j-1. 0..14. Correct.
    # i=1: 15 - 0 + j - 1 - 1 = 13 + j. j=2 -> 15. j=15 -> 28.
    # Total edges for i=1 is 14. 15..28. Correct.
    # General: index = sum_{k=0}^{i-1} (15-k) + (j - i - 1)
    # sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)/2
    # So: 15*i - i*(i-1)//2 + j - i - 1
    return i * 15 - i * (i - 1) // 2 + j - i - 1

def _precompute_k4s():
    """Return list of 6-tuples of edge indices for each K4."""
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    edges = [
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ]
                    k4s.append(tuple(edges))
    return k4s

_K4S = _precompute_k4s()
_K4S_ARRAY = np.array(_K4S, dtype=np.int32)  # shape (1820, 6)
_NUM_K4S = len(_K4S)

def _count_violations_fast(colours_arr):
    """Vectorized violation count using numpy. colours_arr is a numpy array of shape (120,)."""
    k4_colours = colours_arr[_K4S_ARRAY]  # (1820, 6)
    sorted_cols = np.sort(k4_colours, axis=1)  # (1820, 6)
    changes = (sorted_cols[:, 1:] != sorted_cols[:, :-1]).astype(np.int32)
    distinct_counts = 1 + changes.sum(axis=1)
    violations = np.sum(distinct_counts < 5)
    return int(violations)

def _num_colours_used(colours):
    return len(set(colours))

def _local_search_repair(colours, deadline):
    """Fix violations by changing edge colours."""
    current = list(colours)
    current_arr = np.array(current, dtype=np.int32)
    current_violations = _count_violations_fast(current_arr)
    rng = random.Random(42)
    
    # Track used colours to pick from
    used_colours = set(current)
    
    while time.time() < deadline:
        if current_violations == 0:
            break
        
        # Pick a random edge
        idx = rng.randrange(120)
        old_c = current[idx]
        
        # Try a random new colour. 
        # To reduce violations, we want to introduce a NEW colour into a K4 that has few colours.
        # Changing an edge in a violating K4 to a colour not in that K4 helps.
        # Simple heuristic: pick a random colour from 0..14 (or slightly larger range if stuck)
        # Let's stick to 0..14 first.
        new_c = rng.randrange(15)
        
        if new_c == old_c:
            continue
            
        current[idx] = new_c
        current_arr[idx] = new_c
        new_violations = _count_violations_fast(current_arr)
        
        if new_violations < current_violations:
            current_violations = new_violations
            used_colours.add(new_c)
        else:
            # Revert
            current[idx] = old_c
            current_arr[idx] = old_c
            
    return current, current_violations

def _local_search_minimize(colours, deadline):
    """Minimize number of colours used while maintaining validity."""
    current = list(colours)
    current_arr = np.array(current, dtype=np.int32)
    current_colours = _num_colours_used(current)
    rng = random.Random(123)
    
    used_colours = list(set(current))
    
    while time.time() < deadline:
        if len(used_colours) <= 5:
            break
            
        # Try to merge a colour into another
        # Pick a random colour to eliminate
        c_elim = rng.choice(used_colours)
        # Pick a random target colour
        c_target = rng.choice([c for c in used_colours if c != c_elim])
        
        # Apply mapping
        trial = list(current)
        for i in range(120):
            if trial[i] == c_elim:
                trial[i] = c_target
                
        trial_arr = np.array(trial, dtype=np.int32)
        v = _count_violations_fast(trial_arr)
        
        if v == 0:
            # Valid, accept
            current = trial
            current_arr = trial_arr
            current_colours = _num_colours_used(current)
            used_colours = list(set(current))
        else:
            # Try swapping colours of two edges? Or just revert.
            # Let's try a small perturbation: change one edge of c_elim to c_target, see if it helps locally?
            # No, merging is a global change. If it fails, it's likely bad.
            # Let's try random edge swaps to escape local minima in the colour count space.
            idx1 = rng.randrange(120)
            idx2 = rng.randrange(120)
            if idx1 == idx2:
                continue
            c1 = current[idx1]
            c2 = current[idx2]
            if c1 == c2:
                continue
            current[idx1] = c2
            current[idx2] = c1
            current_arr[idx1] = c2
            current_arr[idx2] = c1
            v = _count_violations_fast(current_arr)
            if v == 0:
                current_colours = _num_colours_used(current)
                used_colours = list(set(current))
            else:
                # Revert
                current[idx1] = c1
                current[idx2] = c2
                current_arr[idx1] = c1
                current_arr[idx2] = c2

    return current, current_colours

def construct():
    t0 = time.time()
    
    # Start with a random 15-colouring
    # Using a structured start might be better, but random is safer for diversity
    # Let's try (i+j)%15 first, it's fast to generate
    # But (i+j)%15 might have structural violations.
    # Let's use random initialization with 15 colours.
    rng = random.Random(0)
    colours = [rng.randrange(15) for _ in range(120)]
    
    # Phase 1: Repair violations
    deadline_repair = t0 + 30.0
    colours, v = _local_search_repair(colours, deadline_repair)
    
    if v > 0:
        # If we couldn't repair with 15 colours, try with more colours?
        # Or just return the best effort? 
        # The task requires a VALID colouring.
        # If we can't get valid, we should fall back to a known valid one.
        # (i+j)%17 is known valid.
        colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        v = _count_violations_fast(np.array(colours, dtype=np.int32))
        assert v == 0, "Fallback construction must be valid"
        
    # Phase 2: Minimize colours
    deadline_min = t0 + DEADLINE
    colours, num_c = _local_search_minimize(colours, deadline_min)
    
    # Final verification
    if _count_violations_fast(np.array(colours, dtype=np.int32)) != 0:
        # Fallback
        colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return colours
