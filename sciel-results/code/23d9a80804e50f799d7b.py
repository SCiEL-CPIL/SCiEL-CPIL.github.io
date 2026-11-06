import time
def construct():
    p = 1009
    deadline = time.time() + 50.0
    
    residues = set()
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        residues.add(r)
        if r not in roots:
            roots[r] = x
    
    for s in range(1, p - 4):
        if time.time() > deadline:
            pass
            
        valid = True
        for i in range(5):
            val = s + i
            if val == 0 or val not in residues:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
