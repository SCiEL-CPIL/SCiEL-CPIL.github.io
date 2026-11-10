"""Maximize pairwise-disjoint infinite residue classes with distinct moduli 2..128."""
import math

def construct():
    # Maximum possible = 127 (one per modulus 2..128).
    # Disjointness of [r1,m1] and [r2,m2]: gcd(m1,m2) must NOT divide (r1-r2).
    # Greedy: for each modulus, scan residues 0..m-1, accept the first that is
    # disjoint from all already-chosen classes. A valid residue always exists.
    chosen = []
    for m in range(2, 129):
        for r in range(m):
            ok = True
            for (rj, mj) in chosen:
                g = math.gcd(m, mj)
                if g > 1 and ((r - rj) % g) == 0:
                    ok = False
                    break
            if ok:
                chosen.append([r, m])
                break
    return chosen
