import itertools
import random
import time

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Build adjacency list
    # Edge u->v if suffix of u (len n-1) == prefix of v (len n-1)
    adj = [[] for _ in range(len(perms))]
    for u in perms:
        suffix = u[1:]
        # Find all v such that v[:n-1] == suffix
        # This is equivalent to finding v where v[0] is the dropped element
        # and v[1:] is the suffix? No.
        # v[0] = x, v[1] = u[1], ..., v[n-1] = u[n-1].
        # So v is determined by the element x that is NOT in suffix?
        # No, v must be a permutation.
        # v[1:] is fixed to u[1:].
        # v[0] must be the element missing from u[1:].
        # So there is EXACTLY ONE v for each u?
        # No. u[1:] has n-1 elements.
        # The missing element is unique.
        # So v is unique?
        # Yes, for a given u, there is only one v such that v[1:] == u[1:].
        # Wait, the De Bruijn graph of order n-1 has nodes of length n-1.
        # Here nodes are length n.
        # The condition is overlap of n-1.
        # So yes, it's a directed graph where each node has out-degree 1?
        # No.
        # u = (a1, ..., an).
        # v = (b1, ..., bn).
        # Overlap n-1: a2...an == b1...b{n-1}.
        # So b1=a2, b2=a3, ..., b{n-1}=an.
        # bn is the remaining element.
        # So v is uniquely determined by u.
        # This is a single cycle?
        # If so, the path is fixed.
        # But we need to visit ALL permutations.
        # If each node has out-degree 1, we can only follow the cycle.
        # This doesn't cover all nodes unless it's a single Hamiltonian cycle.
        # The De Bruijn graph of permutations (permutation graph) is different.
        
        # Actually, the standard De Bruijn graph for superpermutations uses nodes of length n-1.
        # Edges are length n.
        # We want a path in the De Bruijn graph of order n-1 that visits all edges?
        # No, visits all nodes (permutations of n-1)?
        # If we visit all nodes of length n-1, we cover all permutations of length n?
        # No.
        
        # Let's stick to the canonical construction first, as it's known to work and is fast.
        # The issue is why it scores 0.52.
        # I will output the canonical construction.
        
    S = [1]
    for m in range(2, n + 1):
        seen = set()
        perms_order = []
        for i in range(len(S) - (m - 1) + 1):
            window = tuple(S[i:i + m - 1])
            if window not in seen:
                seen.add(window)
                perms_order.append(window)
                
        new_S = []
        for P in perms_order:
            block = list(P) + [m] + list(P)
            max_possible_overlap = min(len(new_S), len(block) - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new_S[-k:] == block[:k]:
                    overlap = k
                    break
            new_S.extend(block[overlap:])
        S = new_S
        
    return "".join(map(str, S))
