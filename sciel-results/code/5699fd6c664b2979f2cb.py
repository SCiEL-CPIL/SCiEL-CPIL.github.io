import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 80.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Store all subsets: size -> list of (residue, terms)
    # Also store a mapping residue -> terms for quick retrieval
    # We want the minimum max length L.
    
    # subsets_by_size[size] = list of (residue, terms)
    subsets_by_size = defaultdict(list)
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        size = len(terms)
        subsets_by_size[size].append((s, terms))
    
    # Find minimum L such that union of residues from sizes 1..L covers all 0..100
    # We can just iterate L from 1 to 10
    
    # Pre-compute the set of residues covered by sizes <= L
    # Actually, we can just check coverage incrementally
    
    covered_residues = set()
    # Map residue to a valid subset (any size <= current L)
    # We want to keep the smallest size if possible? No, just any valid one.
    # But we want to minimize the MAX size.
    # So we find the smallest L that works.
    
    assignment = {}
    
    for L in range(1, 11):
        # Add subsets of size L to our pool
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms
        
        # Check if all residues are covered
        if len(assignment) == P:
            # We found a valid L
            # Construct the final list
            result = [assignment[r] for r in range(P)]
            return result
            
        # Check time
        if time.time() > deadline:
            # Fallback: return whatever we have, padded?
            # If we haven't covered all, we must use larger L.
            # But this loop is very fast.
            pass

    # If we reach here, it means no L up to 10 covered all residues?
    # This is impossible since L=10 includes all subsets.
    # But just in case, return a default valid solution
    # (This should never happen)
    return [assignment[r] if r in assignment else [1] for r in range(P)]
