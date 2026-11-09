"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number_fast(triples, n=16):
    """Fast independence number computation using bitmask DP."""
    if not triples:
        return n
    
    # Precompute triple bitmasks
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Use a greedy + verification approach for speed
    # First, try to find large independent sets by checking from high size down
    # But use a smarter method: for each size, check fewer candidates
    
    # Precompute which triples contain each point
    point_in_triples = [[] for _ in range(n)]
    for i, tm in enumerate(triple_masks):
        for p in range(n):
            if tm & (1 << p):
                point_in_triples[p].append(i)
    
    # Check from n down to 1
    for size in range(n, 0, -1):
        # For size 16, just check the full set
        if size == n:
            mask = (1 << n) - 1
            for tm in triple_masks:
                if (tm & mask) == tm:
                    break
            else:
                return size
            continue
        
        # For large sizes, use a heuristic: try removing points that are in many triples
        if size >= 10:
            # Try a few random-ish subsets of this size
            # Actually, let's just do the full check but break early
            count = 0
            for subset in combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is fully contained
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return size
                count += 1
                # Safety: don't check too many
                if count > 5000:
                    break
            # If we broke early, continue to next size
            continue
        
        # For smaller sizes, full check
        for subset in combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return size
    
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    pair_set = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                if pair in pair_set:
                    return False
                pair_set.add(pair)
    return True

def _algebraic_sts15_plus_point():
    """
    Construct a linear triple system on 16 points using the STS(15) from F2^4
    (PG(3,2) lines) plus point 0, with additional triples to cover point 0.
    
    The STS(15): triples {a, b, a^b} for nonzero a,b in F2^4, a < b.
    This gives C(15,2)/3 = 35 triples, each pair of nonzero points in exactly one triple.
    
    For point 0: we add triples {0, a, b} where a and b are chosen so that
    the resulting system remains linear. Since {a, b, a^b} is already a triple,
    we can't add {0, a, b} if a^b is also involved. Instead, we need to be careful.
    
    Actually, in a linear system, each pair can be in at most one triple. The STS(15)
    already uses all pairs among {1..15}. So we cannot add ANY triple containing two
    nonzero points that are already paired in the STS(15). 
    
    Therefore, to add triples with point 0, we must use pairs (0, a) and (0, b) which
    are new pairs. But {0, a, b} uses pairs (0,a), (0,b), and (a,b). The pair (a,b)
    is already in the STS(15)! So we CANNOT add any triple {0, a, b} for a,b nonzero.
    
    This means point 0 is isolated in the STS(15) extension. We need a different approach.
    
    Alternative: Remove some triples from the STS(15) to free up pairs, then add
    triples involving point 0.
    
    Better approach: Use a different construction entirely.
    """
    # Let's use a different algebraic structure
    # F2^4 has 16 elements. Consider the 5-dimensional structure or use
    # a different decomposition.
    
    # Actually, let's use the following: partition the 15 nonzero elements into
    # 5 groups of 3, and create triples within and across groups.
    
    # A cleaner approach: use the affine geometry AG(4,2) and take specific
    # 3-point subsets from parallel classes of lines.
    
    # In AG(4,2), there are 5 parallel classes of lines (each class has 8 parallel lines
    # covering all 16 points). Each line has 3 points... wait, in AG(4,2) lines have 2 points.
    
    # Let me reconsider. In F2^4:
    # - A 1-dimensional subspace (line through origin) has 2 nonzero points: {a, a} no, {a}
    #   Actually a 1-dim subspace is {0, a} for nonzero a.
    # - A 2-dimensional subspace (plane through origin) has 4 points: {0, a, b, a^b}
    
    # The triples {a, b, a^b} for a,b nonzero, a != b, a^b != 0 are the lines of PG(3,2).
    # There are C(15,2)/3 = 35 such triples, forming an STS(15).
    
    # For a 16-point system, let's try a different algebraic construction:
    # Use Z_16 or a product structure.
    
    # Let me try: represent points as (i, j) where i in 0..3, j in 0..3.
    # Create triples based on arithmetic patterns.
    
    triples = []
    pair_used = set()
    
    def add_triple(a, b, c):
        t = tuple(sorted((a, b, c)))
        if len(set(t)) < 3:
            return False
        pairs = [(min(t[0],t[1]), max(t[0],t[1])),
                 (min(t[0],t[2]), max(t[0],t[2])),
                 (min(t[1],t[2]), max(t[1],t[2]))]
        for p in pairs:
            if p in pair_used:
                return False
        for p in pairs:
            pair_used.add(p)
        triples.append(list(t))
        return True
    
    # Construction based on 4x4 grid with modular arithmetic
    # Points are (row, col) mapped to row*4 + col
    
    # Type 1: Horizontal lines in 4x4 grid (but these are 4 points, not 3)
    # Use: for each row, take 3 of the 4 points
    # This won't work well for linearity.
    
    # Let's use a known construction: the unique 2-(16, 4, 2) design or similar.
    
    # Actually, let me use a simpler but effective approach:
    # Start with the STS(15) on points 1-15, then find a maximum matching
    # of pairs involving point 0 that don't conflict.
    
    # Since all pairs among 1-15 are used, we need to REMOVE some triples from
    # the STS(15) to free up pairs, then add triples with point 0.
    
    # Strategy: Remove k triples from STS(15), freeing up 3k pairs among 1-15.
    # Then add k triples of the form {0, a, b} using freed pairs (a,b).
    # Net change: remove 3k pairs, add 3k pairs (k of which are (0,a) or (0,b)).
    
    # Let's try removing 5 triples and adding 5 triples with point 0.
    
    # First, generate the STS(15)
    sts15 = []
    seen = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                sts15.append(t)
    
    # Now, greedily select triples to remove and replace with {0, a, b}
    # We need to find 5 triples from STS(15) such that their 15 pairs can be
    # rearranged into 5 pairs (a_i, b_i) where we form {0, a_i, b_i}.
    
    # Actually, each removed triple {a, b, c} frees pairs (a,b), (a,c), (b,c).
    # To add {0, x, y}, we need pair (x,y) to be free.
    
    # Let's try: remove triples that share points in a way that creates free pairs.
    
    # Simpler: just use a randomized approach to build the system.
    
    # Let me try a completely different construction:
    # Use the 16 points as elements of Z_2 x Z_2 x Z_2 x Z_2.
    # Define triples as {x, y, z} where x+y+z = 0 (i.e., z = x+y in F2^4).
    # This is exactly the STS(15) on nonzero points, with 0 isolated.
    
    # To include 0, let's use a different equation.
    # Consider triples {x, y, z} where x+y+z = v for some fixed nonzero v.
    # For v = 1 (say), this gives triples not containing 0 necessarily.
    
    # Actually, let me just use a well-known construction for 16 points.
    
    # The Witt design S(3,4,8) has 8 points. Not directly applicable.
    
    # Let me try: use 4 parallel classes of a resolvable design.
    
    # For now, let's use a practical approach:
    # 1. Start with STS(15) on points 1-15
    # 2. Remove a few triples to free pairs
    # 3. Add triples involving point 0
    
    # Let's remove 3 triples that form a "triangle" in some sense
    # and add 3 triples with point 0.
    
    # Actually, the simplest correct approach:
    # Use the fact that we don't need a Steiner system, just a linear system.
    # We can have fewer than 35 triples.
    
    # Let's build a system with point 0 included:
    # Use triples {0, a, a^b} for various a, b, but ensure linearity.
    
    # New plan: use a greedy algorithm to build a good linear system.
    
    return None  # Placeholder, we'll use a different approach in construct()

def _build_algebraic_system():
    """
    Build a linear triple system on 16 points using algebraic structure.
    
    Approach: Use F2^4 structure. The 15 nonzero vectors form an STS(15) via
    triples {a, b, a^b}. To include point 0, we'll use a modified construction.
    
    Key insight: Instead of the standard STS(15), we can use a different
    3-uniform hypergraph on 16 points with good independence properties.
    
    Construction:
    - Label points 0-15 as elements of F2^4.
    - For each 2-dimensional subspace (plane) through the origin, it contains
      4 points: {0, a, b, a^b}. The 4 choose 3 = 4 triples from this plane
      would be {0,a,b}, {0,a,a^b}, {0,b,a^b}, {a,b,a^b}. But these share pairs.
    
    - Instead, for each 2-dim subspace, take just the triple {a, b, a^b}
      (the nonzero part). This gives the STS(15).
    
    - To include 0: for each 1-dim subspace {0, a}, we can't form a triple
      with just 2 points.
    
    Alternative: Use a different set of triples.
    
    Let me try: for each pair of distinct nonzero vectors a, b with a < b,
    the triple is {a, b, a^b}. This is the STS(15) on {1..15}.
    
    Now, to add point 0, we observe that we can REPLACE some triples.
    For example, replace {a, b, a^b} with {0, a, b} if the pairs (0,a) and (0,b)
    are not yet used and (a,b) is freed.
    
    Let's do this systematically:
    - Start with STS(15): 35 triples on points 1-15.
    - We want to add triples involving point 0.
    - Each triple {0, a, b} uses pairs (0,a), (0,b), (a,b).
    - Pair (a,b) is used in STS(15) triple {a, b, a^b}.
    - So to add {0, a, b}, we must first remove {a, b, a^b}.
    - After removing {a, b, a^b}, pairs (a,b), (a,a^b), (b,a^b) are free.
    - We add {0, a, b}, using (a,b). Pairs (a,a^b) and (b,a^b) remain free.
    - We could add more triples using these freed pairs.
    
    Let's try to add 5 triples involving point 0 by removing 5 STS triples.
    """
    # Generate STS(15)
    sts_triples = []
    seen = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                sts_triples.append(t)
    
    # Build pair-to-triple map
    pair_to_idx = {}
    for i, t in enumerate(sts_triples):
        for k in range(3):
            for l in range(k+1, 3):
                pair = (min(t[k], t[l]), max(t[k], t[l]))
                pair_to_idx[pair] = i
    
    # Greedily add triples with point 0
    triples = [list(t) for t in sts_triples]
    used_pairs = set(pair_to_idx.keys())
    
    # Try to add triples {0, a, b} for various a, b
    # We need pair (a,b) to be free, so we must remove the triple containing (a,b)
    
    added = 0
    # Try all pairs (a, b) with 1 <= a < b <= 15
    for a in range(1, 16):
        for b in range(a+1, 16):
            if added >= 5:
                break
            if a == 0 or b == 0:
                continue
            
            # Check if pair (a,b) is currently used
            pair_ab = (a, b)
            if pair_ab not in used_pairs:
                # Pair is free, add {0, a, b}
                # Check pairs (0,a) and (0,b) are free
                pair_0a = (0, a)
                pair_0b = (0, b)
                if pair_0a not in used_pairs and pair_0b not in used_pairs:
                    triples.append([0, a, b])
                    used_pairs.add(pair_0a)
                    used_pairs.add(pair_0b)
                    used_pairs.add(pair_ab)  # Already should be free
                    added += 1
            else:
                # Pair (a,b) is used in some triple. Remove that triple and add {0,a,b}
                idx = pair_to_idx.get(pair_ab)
                if idx is not None and idx < len(triples):
                    old_triple = triples[idx]
                    # Check if old_triple is still in the list (not already removed)
                    if old_triple in triples:
                        # Check pairs (0,a) and (0,b) are free
                        pair_0a = (0, a)
                        pair_0b = (0, b)
                        if pair_0a not in used_pairs and pair_0b not in used_pairs:
                            # Remove old triple
                            triples[idx] = [0, a, b]
                            # Update used pairs
                            # Remove pairs from old triple
                            for k in range(3):
                                for l in range(k+1, 3):
                                    p = (min(old_triple[k], old_triple[l]), max(old_triple[k], old_triple[l]))
                                    if p != pair_ab:
                                        used_pairs.discard(p)
                                    # pair_ab is being reused
                            # Add new pairs
                            used_pairs.add(pair_0a)
                            used_pairs.add(pair_0b)
                            # pair_ab is still used (in new triple)
                            added += 1
                            # Update pair_to_idx for the new triple
                            for k in range(3):
                                for l in range(k+1, 3):
                                    p = (min(triples[idx][k], triples[idx][l]), max(triples[idx][k], triples[idx][l]))
                                    pair_to_idx[p] = idx
                            # Clear old pair mappings
                            for k in range(3):
                                for l in range(k+1, 3):
                                    p = (min(old_triple[k], old_triple[l]), max(old_triple[k], old_triple[l]))
                                    if p != pair_ab:
                                        if p in pair_to_idx and pair_to_idx[p] == idx:
                                            del pair_to_idx[p]
    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    best_triples = None
    best_indep = 16
    
    # Try the algebraic construction
    alg = _build_algebraic_system()
    if alg and _is_linear(alg):
        indep = _independence_number_fast(alg, 16)
        best_indep = indep
        best_triples = alg
    
    # Also try a simple structured construction:
    # Use 4 groups of 4 points, with triples within and between groups
    for seed in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        triples = []
        pair_used = set()
        
        def try_add(a, b, c):
            nonlocal pair_used
            t = tuple(sorted((a, b, c)))
            if len(set(t)) < 3:
                return False
            pairs = [(min(t[0],t[1]), max(t[0],t[1])),
                     (min(t[0],t[2]), max(t[0],t[2])),
                     (min(t[1],t[2]), max(t[1],t[2]))]
            for p in pairs:
                if p in pair_used:
                    return False
            for p in pairs:
                pair_used.add(p)
            triples.append(list(t))
            return True
        
        # Generate random linear triples
        attempts = 0
        while len(triples) < 50 and attempts < 5000:
            attempts += 1
            a, b, c = rng.sample(range(16), 3)
            try_add(a, b, c)
        
        if triples and _is_linear(triples):
            indep = _independence_number_fast(triples, 16)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples
    
    if best_triples is None:
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
