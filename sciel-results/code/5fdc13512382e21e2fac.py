import time
from itertools import product

def construct():
    """Return the 132 blocks of the Steiner system S(5,6,12)."""
    
    # Points are 0..10 and 11 (infinity)
    # Field is GF(11)
    p = 11
    
    def apply_trans(x, a, b, c, d):
        if x == 11: # infinity
            if c == 0:
                return 11
            else:
                return pow(a, -1, p) * c % p # a/c mod p
        # x is 0..10
        denom = c * x + d
        if denom % p == 0:
            return 11
        else:
            num = a * x + b
            return num * pow(denom, -1, p) % p

    # Seed block: {0, 1, 3, 4, 5, 9}
    seed = [0, 1, 3, 4, 5, 9]
    
    blocks_set = set()
    
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a == 0:
                    if c == 0:
                        continue
                    if (b * c) % p == p - 1:
                        for d in range(p):
                            new_block = []
                            for x in seed:
                                new_block.append(apply_trans(x, a, b, c, d))
                            blocks_set.add(tuple(sorted(new_block)))
                else:
                    inv_a = pow(a, -1, p)
                    for c_val in range(p):
                        for b_val in range(p):
                            d = (1 + b_val * c_val) * inv_a % p
                            new_block = []
                            for x in seed:
                                new_block.append(apply_trans(x, a, b_val, c_val, d))
                            blocks_set.add(tuple(sorted(new_block)))

    blocks = [list(b) for b in blocks_set]
    
    # Sort blocks lexicographically for canonical output
    blocks.sort()
    
    return blocks
