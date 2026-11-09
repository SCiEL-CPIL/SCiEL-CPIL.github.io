"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Pre-compute modular inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # Group subsets by size
    # subsets_by_size[s] = list of (residue, terms_list) for subsets of size s
    subsets_by_size = [[] for _ in range(11)]  # index 0 unused
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                x = i + 1
                terms.append(x)
                s = (s + inv[x]) % 101
                size += 1
        subsets_by_size[size].append((s, terms))
    
    # For each size, try to cover uncovered residues
    # best[r] = smallest subset (by size) that sums to r
    best = [None] * 101
    uncovered = set(range(101))
    
    for size in range(1, 11):
        if not uncovered:
            break
        if time.time() > deadline:
            break
        for (residue, terms) in subsets_by_size[size]:
            if residue in uncovered:
                best[residue] = terms
                uncovered.discard(residue)
        if not uncovered:
            break
    
    # If any residue is still uncovered (shouldn't happen), use any available
    # But we know all 101 residues are covered by the 1023 subsets
    # As a fallback, use the seed approach
    if uncovered:
        representations = {}
        for mask in range(1, 1 << 10):
            terms = []
            s = 0
            for i in range(10):
                if mask & (1 << i):
                    x = i + 1
                    terms.append(x)
                    s = (s + inv[x]) % 101
            if s not in representations:
                representations[s] = terms
        for r in uncovered:
            if r in representations:
                best[r] = representations[r]
            else:
                best[r] = [1]  # fallback, shouldn't happen
    
    return [best[r] for r in range(101)]
# EVOLVE-BLOCK-END
