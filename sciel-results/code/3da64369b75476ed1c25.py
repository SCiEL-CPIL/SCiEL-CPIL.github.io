import time
def construct():
    start_time = time.time()
    _P = 1009
    _M = 5
    
    # Use a list of booleans for clarity and potentially faster access than bit shifts on huge ints
    is_residue = [False] * _P
    root_map = {}
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_residue[r] = True
            if r not in root_map:
                root_map[r] = x
                
    for s in range(1, _P - _M + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        if all(is_residue[s + i] for i in range(_M)):
            roots = [root_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No baseline run found')
