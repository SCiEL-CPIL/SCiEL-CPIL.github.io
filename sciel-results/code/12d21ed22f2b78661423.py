import time
import numpy as np

def construct():
    p = 1009
    deadline = time.time() + 85.0
    
    # Precompute quadratic residues and their roots
    # residues: set of non-zero QRs
    # root_map: dictionary mapping residue to one of its roots (1 <= root < p)
    residues = set()
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # Exclude zero residue as per "zero residues ... are excluded"
            residues.add(r)
            # Store the root. Since we iterate x from 1, the first root found is the smallest.
            if r not in root_map:
                root_map[r] = x
        
        if time.time() > deadline:
            # This should practically never happen for p=1009, but safe to include
            return {'start': 1004, 'roots': [1, 1, 1, 1, 1]} # Fallback, likely invalid but prevents crash

    # Search for the minimum s
    # Constraint: 1 <= s <= 1004
    # We need s, s+1, s+2, s+3, s+4 to be in residues
    # Note: s+i must be a valid residue in 1..1008.
    # If s=1004, s+4=1008. All values are < 1009, so no wraparound issues for mod p arithmetic 
    # in the sense that the numbers themselves are the residues.
    
    # Create a boolean array indicating which values (1..1008) are quadratic residues
    # is_residue[i] is True if i is a QR, False otherwise
    is_residue = np.zeros(p, dtype=bool)
    for r in residues:
        is_residue[r] = True
    
    # Find the minimal s such that s, s+1, s+2, s+3, s+4 are all residues
    # We can use a sliding window approach or bitwise operations
    # For simplicity and clarity, we'll use a direct check with numpy
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback if we hit deadline during search (very unlikely)
            pass
        
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if all(is_residue[s+i] for i in range(5)):
            # Ensure roots exist (they must, if in residues)
            # roots[i] is a root for s+i
            r0 = root_map[s]
            r1 = root_map[s+1]
            r2 = root_map[s+2]
            r3 = root_map[s+3]
            r4 = root_map[s+4]
            
            return {'start': s, 'roots': [r0, r1, r2, r3, r4]}
            
    # If no run found (should not happen for p=1009, m=5)
    # The problem states "Early runs ...", implying existence.
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
