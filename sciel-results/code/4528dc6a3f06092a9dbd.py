"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _independence_number(triples):
    """Compute the independence number: largest subset with no complete triple."""
    n = 16
    if not triples:
        return n
    
    # Precompute masks
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
        
    max_indep = 0
    for mask in range(1 << n):
        if bin(mask).count('1') <= max_indep:
            continue
            
        is_independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_independent = False
                break
        
        if is_independent:
            max_indep = bin(mask).count('1')
            
    return max_indep

def _is_valid(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Algebraic design using GF(4)^2 representation
    # Points 0..15 correspond to (x, y) where x, y in GF(4)
    # GF(4) = {0, 1, alpha, alpha^2} with alpha^2 = alpha + 1
    
    # Map GF(4) elements to integers 0..3
    # 0 -> 0, 1 -> 1, alpha -> 2, alpha^2 -> 3
    
    # GF(4) addition table (index: a + b)
    gf4_add = [
        [0, 1, 2, 3],  # 0 + x
        [1, 0, 3, 2],  # 1 + x
        [2, 3, 0, 1],  # alpha + x
        [3, 2, 1, 0],  # alpha^2 + x
    ]
    
    # GF(4) multiplication table (index: a * b)
    gf4_mul = [
        [0, 0, 0, 0],  # 0 * x
        [0, 1, 2, 3],  # 1 * x
        [0, 2, 3, 1],  # alpha * x
        [0, 3, 1, 2],  # alpha^2 * x
    ]
    
    # Point (x, y) maps to integer x * 4 + y
    def point_id(x, y):
        return x * 4 + y
    
    def gf4_add_pts(a, b):
        return gf4_add[a][b]
    
    def gf4_mul_pts(a, b):
        return gf4_mul[a][b]
    
    # Generate all affine lines in GF(4)^2
    # An affine line is {p, p + d, p + 2d} where d != 0
    # In GF(4), the scalar multiples of d are d and 2d = d + d (since char 2)
    # Wait, in GF(4) with char 2, 1 + 1 = 0, so 2 = 0? No.
    # GF(4) has characteristic 2. So 1 + 1 = 0.
    # The nonzero elements are {1, alpha, alpha^2}.
    # For a fixed direction d, the line through p in direction d is {p, p+d, p+alpha*d}.
    # Since alpha*d is another nonzero multiple, this gives 3 distinct points.
    
    triples = []
    seen = set()
    
    # Iterate over all points p and directions d
    for px in range(4):
        for py in range(4):
            p = point_id(px, py)
            for dx in range(4):
                for dy in range(4):
                    if dx == 0 and dy == 0:
                        continue
                    # Line: {p, p+d, p+alpha*d}
                    # Compute p + d
                    q1x = gf4_add[px][dx]
                    q1y = gf4_add[py][dy]
                    q1 = point_id(q1x, q1y)
                    
                    # Compute alpha * d
                    adx = gf4_mul[2][dx]  # 2 represents alpha
                    ady = gf4_mul[2][dy]
                    
                    # Compute p + alpha*d
                    q2x = gf4_add[px][adx]
                    q2y = gf4_add[py][ady]
                    q2 = point_id(q2x, q2y)
                    
                    # The triple is {p, q1, q2}
                    t = tuple(sorted([p, q1, q2]))
                    if t not in seen:
                        # Check linearity: any two triples share at most 1 point
                        valid = True
                        for existing in triples:
                            common = set(t) & set(existing)
                            if len(common) > 1:
                                valid = False
                                break
                        if valid:
                            triples.append(list(t))
                            seen.add(t)
    
    # Verify validity
    if not _is_valid(triples):
        # Fallback to random
        triples = []
        all_triples = list(combinations(range(16), 3))
        random.shuffle(all_triples)
        for t in all_triples:
            t_set = set(t)
            valid = True
            for existing in triples:
                if len(t_set & set(existing)) > 1:
                    valid = False
                    break
            if valid:
                triples.append(list(t))
    
    # Now try to improve by adding more triples
    current_score = _independence_number(triples)
    
    # Try to add random triples that don't violate linearity
    while time.time() < deadline - 5.0:
        improved = False
        for _ in range(1000):
            t = list(random.sample(range(16), 3))
            t_set = set(t)
            valid = True
            for existing in triples:
                if len(t_set & set(existing)) > 1:
                    valid = False
                    break
            if valid:
                # Check if it improves the score
                new_triples = triples + [t]
                new_score = _independence_number(new_triples)
                if new_score < current_score:
                    triples = new_triples
                    current_score = new_score
                    improved = True
                    break
        if not improved:
            break
    
    return triples
# EVOLVE-BLOCK-END
