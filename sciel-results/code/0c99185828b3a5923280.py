"""
Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009.
"""
import time

_P = 1009
_M = 5

def construct():
    # Deadline guard (though this task is very fast, it's good practice per directive)
    start_time = time.time()
    
    # 1. Precompute quadratic residues and one root for each
    # is_residue[i] is True if i is a non-zero quadratic residue mod _P
    # root_of[i] stores a witness x such that x*x % _P == i
    is_residue = [False] * _P
    root_of = [0] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and not is_residue[r]:
            is_residue[r] = True
            root_of[r] = x
    
    # 2. Search for the minimum s
    # We require s, s+1, ..., s+4 to be residues.
    # Since s >= 1 and s <= 1004, the values s..s+4 are in [1, 1008].
    # We just check if they are marked as residues.
    for s in range(1, _P - _M + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        # Using a loop is clear, but slicing or all() is faster
        if all(is_residue[s + i] for i in range(_M)):
            roots = [root_of[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
    
    # Fallback (should not happen for p=1009, m=5)
    raise RuntimeError('No run found')
