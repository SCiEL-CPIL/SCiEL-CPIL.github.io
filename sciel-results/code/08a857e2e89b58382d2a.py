import numpy as np

def construct():
    p = 1009
    # Generate all non-zero squares mod p
    x_vals = np.arange(1, p)
    squares = (x_vals * x_vals) % p
    
    # Create a boolean mask for residues
    is_residue = np.zeros(p, dtype=bool)
    is_residue[squares] = True
    
    # We need to find the smallest s such that is_residue[s:s+5] are all True.
    # s ranges from 1 to p-5 (1004).
    # We can check this efficiently.
    
    # Optimization: 
    # A run of 5 consecutive residues.
    # We can iterate s from 1 to p-5.
    # Given p=1009, this loop is very small.
    # But we can also use convolution or just simple slicing.
    
    # Let's stick to a simple loop for clarity but using the numpy boolean array.
    # Accessing numpy scalars is slower than python lists, so maybe converting to a python list of bools is faster for the scan?
    # is_res_list = is_residue.tolist()
    
    # Actually, the precomputation is the main cost.
    # Generating 1008 squares is fast.
    
    # Let's find the roots as well.
    # We need the root for each residue in the final window.
    # We can build the root map in parallel or after finding s.
    # Since we need to minimize s, we find s first.
    
    # To find s quickly:
    # We can look for the first index i > 0 such that is_residue[i:i+5].all()
    
    # Note: is_residue[0] is False (0 is not in 1..p-1 squares? 
    # x in 1..p-1. x^2 mod p. If x=p/2? No p is odd.
    # 0 is a residue (0^2), but we exclude 0.
    # Our squares are from x=1..1008.
    # So is_residue[0] will be False unless some x^2 % p == 0, which implies x is multiple of p, not in range.
    # So is_residue[0] is False.
    
    # The task requires 1 <= s.
    
    for s in range(1, p - 4):
        if is_residue[s] and is_residue[s+1] and is_residue[s+2] and is_residue[s+3] and is_residue[s+4]:
            # Found s. Now we need roots.
            # We can compute roots on the fly for these 5 numbers.
            # Or we can have stored them.
            # Since we didn't store them in the numpy array, we need to find them.
            # We can iterate x again? That's O(p).
            # Or we can just use pow(v, (p+1)//4, p)? No, p=1009=4k+1. Tonelli-Shanks or just brute force for 5 numbers is fine.
            # Brute force for 5 numbers: 5 * 1009 operations. Negligible.
            
            roots = []
            targets = [s, s+1, s+2, s+3, s+4]
            # We can build a small map for these specific targets
            target_set = set(targets)
            root_map = {}
            for x in range(1, p):
                v = (x * x) % p
                if v in target_set:
                    if v not in root_map:
                        root_map[v] = x
                    if len(root_map) == 5:
                        break
            # It's possible we don't find all 5 if we break early? 
            # The loop goes 1..p-1. We will find all 5 eventually.
            # But we can just iterate fully if needed, or break when found.
            # Actually, just iterating 1..p-1 once is fine.
            
            for t in targets:
                roots.append(root_map[t])
                
            return {'start': s, 'roots': roots}
            
    raise RuntimeError("No run found")
