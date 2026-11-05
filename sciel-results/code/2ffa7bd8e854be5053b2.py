import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    best_result = [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]
    best_count = len(best_result)
    
    def check_creates_k222(existing_frozensets, new_triple):
        """Returns True if adding new_triple creates a K(2,2,2)."""
        a, b, c = new_triple
        new_fs = frozenset(new_triple)
        remaining = [v for v in range(n) if v not in (a, b, c)]
        
        for x, y, z in combinations(remaining, 3):
            verts = [a, b, c, x, y, z]
            # Generate all 15 perfect matchings
            # a pairs with one of the other 5
            for pa in [b, c, x, y, z]:
                rest = [v for v in verts if v not in (a, pa)]
                # 3 matchings of 4 elements
                matchings = [
                    [(rest[0], rest[1]), (rest[2], rest[3])],
                    [(rest[0], rest[2]), (rest[1], rest[3])],
                    [(rest[0], rest[3]), (rest[1], rest[2])],
                ]
                for pairs2 in matchings:
                    # Check if {a,b,c} is a transversal
                    # a is in pair0=(a,pa). b and c must be in different pairs among pairs2
                    b_loc = -1
                    c_loc = -1
                    for pi in range(2):
                        if b in pairs2[pi]:
                            b_loc = pi
                        if c in pairs2[pi]:
                            c_loc = pi
                    if b_loc == -1 or c_loc == -1 or b_loc == c_loc:
                        continue
                    if b == pa or c == pa:
                        continue
                    
                    # {a,b,c} is a transversal. Check all 8 transversals.
                    p0 = (a, pa)
                    p1 = tuple(pairs2[0])
                    p2 = tuple(pairs2[1])
                    
                    ok = True
                    for u in p0:
                        for v in p1:
                            for w in p2:
                                if u == v or v == w or u == w:
                                    continue
                                fs = frozenset((u, v, w))
                                if fs == new_fs:
                                    continue
                                if fs not in existing_frozensets:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        return True
        return False
    
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        current_set = set()
        current_triples = []
        
        for tri in shuffled:
            if time.time() > deadline - 5:
                break
            if check_creates_k222(current_set, tri):
                continue
            current_set.add(frozenset(tri))
            current_triples.append(list(tri))
        
        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_result = [list(t) for t in current_triples]
    
    return best_result
