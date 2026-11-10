"""Rewritten program using a vectorized/brute-force approach with early termination."""
import numpy as np

def construct():
    p = 1009
    # Precompute a mapping from residue to one root (smallest root for determinism)
    # We need roots for residues 1..1008 (excluding 0)
    # For each x in 1..p-1, r = x*x % p. If r != 0, we can store the smallest x that gives r.
    # But actually, we just need ANY root. The task says "Each residue is certified by its root".
    # The condition is ri**2 mod 1009 == s+i.
    # We want to minimize s such that for i=0..4, (s+i) is a quadratic residue mod 1009 and s+i != 0 mod 1009.
    # Since s >= 1 and s <= 1004, s+i ranges from 1 to 1008 (since max s=1004, max s+4=1008).
    # So we never hit 0 mod 1009. Good.
    
    # Create an array root_of_residue of size p, initialized to -1
    root_of_residue = np.full(p, -1, dtype=np.int64)
    
    # For x in 1..p-1:
    # r = (x*x) % p
    # if r != 0 and root_of_residue[r] == -1: root_of_residue[r] = x
    # We want the smallest x? Or any x? The task doesn't specify which root, just that a root exists.
    # But to be safe and deterministic, let's pick the smallest x for each residue.
    
    x_vals = np.arange(1, p)
    residues = (x_vals * x_vals) % p
    # Filter out residue 0
    mask = residues != 0
    valid_residues = residues[mask]
    valid_x = x_vals[mask]
    
    # For each unique residue, we want the smallest x
    # Since x_vals is sorted, the first occurrence of a residue will have the smallest x.
    # We can use a loop or a vectorized approach.
    # A simple loop is fine since p is small (1009).
    for x, r in zip(valid_x, valid_residues):
        if root_of_residue[r] == -1:
            root_of_residue[r] = x
    
    # Now search for the smallest s in 1..1004 such that
    # root_of_residue[s], root_of_residue[s+1], ..., root_of_residue[s+4] are all != -1
    # Since s max is 1004, s+4 max is 1008, which is < 1009.
    
    # We can vectorize the check:
    # For each s, check if min(root_of_residue[s:s+5]) != -1
    # But we want the FIRST s where all are valid.
    
    # Let's just iterate s from 1 to 1004.
    # For each s, check if all 5 entries are valid.
    
    for s in range(1, 1005):
        # Check s, s+1, s+2, s+3, s+4
        # All must have root_of_residue[.] != -1
        if (root_of_residue[s] != -1 and
            root_of_residue[s+1] != -1 and
            root_of_residue[s+2] != -1 and
            root_of_residue[s+3] != -1 and
            root_of_residue[s+4] != -1):
            roots = [int(root_of_residue[s+i]) for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')
