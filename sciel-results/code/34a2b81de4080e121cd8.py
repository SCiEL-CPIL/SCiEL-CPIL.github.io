"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 29
    # Paley graph of order 29: colour 0 if (j-i) mod 29 is a quadratic residue (nonzero), else 1.
    squares = set()
    for x in range(1, 29):
        squares.add((x * x) % 29)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % 29
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    
    # Verify: neither colour class contains K_{2,2,2}
    def contains_k222(adj):
        # adj is a set of edges (frozenset of pairs) for one colour
        # Check all 6-subsets for K_{2,2,2}
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            # Try all partitions of 6 into 3 pairs
            # There are 15 ways to partition 6 elements into 3 unordered pairs
            a, b, c, d, e, f = combo
            partitions = [
                [(a,b),(c,d),(e,f)],
                [(a,c),(b,d),(e,f)],
                [(a,d),(b,c),(e,f)],
                [(a,b),(c,e),(d,f)],
                [(a,c),(b,e),(d,f)],
                [(a,e),(b,c),(d,f)],
                [(a,b),(c,f),(d,e)],
                [(a,c),(b,f),(d,e)],
                [(a,f),(b,c),(d,e)],
                [(a,d),(b,e),(c,f)],
                [(a,e),(b,d),(c,f)],
                [(a,f),(b,d),(c,e)],
                [(a,d),(b,f),(c,e)],
                [(a,e),(b,f),(c,d)],
                [(a,f),(b,e),(c,d)],
            ]
            for part in partitions:
                ok = True
                for (u, v) in part:
                    if frozenset((u, v)) not in adj:
                        ok = False
                        break
                    # Check all cross edges
                if not ok:
                    continue
                # Check all 12 cross edges
                pairs = part
                for idx1 in range(3):
                    for idx2 in range(idx1 + 1, 3):
                        u1, v1 = pairs[idx1]
                        u2, v2 = pairs[idx2]
                        for u in (u1, v1):
                            for v in (u2, v2):
                                if frozenset((u, v)) not in adj:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    return True
        return False
    
    adj0 = set()
    adj1 = set()
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0.add(frozenset((i, j)))
            else:
                adj1.add(frozenset((i, j)))
            idx += 1
    
    if contains_k222(adj0) or contains_k222(adj1):
        # Fallback to n=17 Paley
        n = 17
        squares17 = set()
        for x in range(1, 17):
            squares17.add((x * x) % 17)
        colors = [0 if (j - i) % 17 in squares17 else 1 for i in range(17) for j in range(i + 1, 17)]
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
