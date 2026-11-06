import itertools
from collections import defaultdict

def construct():
    # 1. Generate all 4-subsets of [1, 16]
    elements = range(1, 17)
    all_combos = list(itertools.combinations(elements, 4))
    all_sets = [set(c) for c in all_combos]
    n = len(all_sets)
    
    # Map set to index for fast lookup
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    universe = set(elements)

    # 2. Precompute sunflower partners
    # partners[i] is a dict: {j: set_of_k}
    # Or a 2D list of sets. 1820x1820 list of sets might be memory heavy but ok.
    # 1820*1820 = 3.3M entries. If each is a set object, overhead is high.
    # Better: list of lists. partners[i][j] is a list of indices.
    # We only need to store for i < j, and access symmetrically.
    # Let's use a list of dicts to save memory, or just a 2D array of None/sets.
    # Given N=1820, a 2D list of size 1820x1820 containing empty lists initially is fine.
    
    partners = [[None for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        partners[i][i] = set() # Self, though we won't use it
        si = all_sets[i]
        for j in range(i + 1, n):
            sj = all_sets[j]
            I = si & sj
            len_I = len(I)
            k_needed = 4 - len_I
            if k_needed < 0:
                continue
            
            # Forbidden elements: (si \ I) U (sj \ I)
            # Free elements: Universe \ (si U sj)
            # Note: I is subset of si U sj, so free elements are disjoint from I.
            # We need to pick k_needed elements from free.
            union_ij = si | sj
            free = universe - union_ij
            
            if len(free) < k_needed:
                continue
                
            # Generate combinations of free of size k_needed
            # The resulting set C = I U comb
            # We need to find its index.
            # To speed up lookup, we can convert I to a frozenset or tuple?
            # set_to_idx keys are sets.
            
            p_set = set()
            if k_needed == 0:
                # C must be I. But |I|=4. C=I=si=sj.
                # But si != sj (i != j). So no such C?
                # If si != sj and si U sj = I? No, I = si & sj.
                # If k_needed=0, C=I.
                # If I=si, then si subset sj. Since sizes equal, si=sj. Contradiction.
                # So if i!=j, k_needed cannot be 0 with a valid distinct C?
                # Wait. If si & sj = I.
                # If C = I, then C must be disjoint from (si\I) U (sj\I).
                # C=I is disjoint from si\I? Yes.
                # So C=I is a valid candidate geometrically.
                # But C must be in all_sets.
                # If C=I, and I=si, then C=si. But C must be distinct from si and sj.
                # So if I=si, C=si is not allowed.
                # So we just check set_to_idx.
                c_set = I
                idx = set_to_idx.get(c_set)
                if idx is not None and idx != i and idx != j:
                    p_set.add(idx)
            else:
                for comb in itertools.combinations(free, k_needed):
                    c_set = I | set(comb)
                    idx = set_to_idx.get(c_set)
                    if idx is not None and idx != i and idx != j:
                        p_set.add(idx)
            
            partners[i][j] = p_set
            partners[j][i] = p_set

    # 3. Greedy Set Packing
    available = set(range(n))
    chosen = []
    # deg[c] = number of pairs (a, b) in chosen such that c is in partners[a][b]
    deg = {c: 0 for c in available}
    
    while available:
        # Find candidate with minimum degree
        # If min degree > 0, we cannot add any more without conflict
        c_min = min(available, key=lambda x: deg[x])
        
        if deg[c_min] > 0:
            break
        
        # Add c_min to chosen
        chosen.append(c_min)
        available.remove(c_min)
        
        # Update degrees
        # New pairs are (c_min, s) for s in chosen[:-1] (the ones before c_min)
        # Actually, just iterate over all s in chosen except c_min
        for s in chosen[:-1]:
            # Get partners of (c_min, s)
            p_set = partners[c_min][s]
            for c in p_set:
                if c in available:
                    deg[c] += 1
                    
    return [sorted(all_sets[i]) for i in chosen]
