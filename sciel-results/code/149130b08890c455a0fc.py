import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 115.0

    # We know S(5,6,12) is the unique Witt design with 132 blocks.
    # Use the standard algebraic construction via S(4,5,11) on 11 points,
    # which has 66 blocks, then extend to 12 points.
    
    # The S(4,5,11) (Steiner system of order 5 on 11 points) can be
    # constructed from the projective line over GF(11) or equivalently
    # from the Mathieu group M11 action. A concrete construction:
    # The 66 blocks of S(4,5,11) are the orbits of the PSL(2,11) action.
    # 
    # A well-known explicit construction uses the "short" and "long" 
    # blocks from the extended binary Golay code, but here we use the
    # classical construction from the 11-point projective geometry.
    #
    # Actually, the simplest deterministic construction: the 66 blocks of
    # S(4,5,11) can be generated as follows. Label points 0..10.
    # The blocks are all 5-subsets that sum to 0 mod 11 in a certain
    # sense... No, let me use a known explicit list or algorithm.
    #
    # Better: use the fact that S(4,5,11) is the unique Steiner system.
    # We can construct it via backtracking with strong symmetry breaking.
    # Fix the first block as {0,1,2,3,4}. Then by the action of the
    # stabilizer of this block (which is S_2 × S_3 acting on the remaining
    # points in a specific way), we can further break symmetry.
    #
    # Actually, for speed, let me just use the known algebraic construction.
    # The 132 blocks of S(5,6,12) can be explicitly listed or generated.
    #
    # Fastest approach: hardcode or generate via known formula.
    # The Witt design S(5,6,12) blocks can be generated from the extended
    # binary Golay code. The 6-weights of the Golay code give the blocks.
    #
    # Let me use the standard construction:
    # S(5,6,12) = { B ∪ {∞} : B ∈ S(4,5,11) } ∪ { complement of B in {0..10} : B ∈ S(4,5,11) }
    # where S(4,5,11) is the Steiner system on points 0..10.
    #
    # For S(4,5,11), I'll use a deterministic backtracking with aggressive
    # symmetry breaking that should complete in milliseconds.

    points11 = list(range(11))
    all4 = list(combinations(points11, 4))
    all5 = list(combinations(points11, 5))
    n4 = len(all4)  # 330
    n5 = len(all5)  # 462
    
    four_idx = {s: i for i, s in enumerate(all4)}
    five_to_fours = {f: frozenset(combinations(f, 4)) for f in all5}
    
    # For each 4-subset, which 5-subsets contain it
    four_to_fives = [[] for _ in range(n4)]
    for f in all5:
        for s in five_to_fours[f]:
            four_to_fives[four_idx[s]].append(f)
    
    # Bitmasks: for each 5-subset, mask of which 4-subsets it covers
    five_mask = [0] * n5
    five_to_idx = {f: i for i, f in enumerate(all5)}
    for i, f in enumerate(all5):
        m = 0
        for s in five_to_fours[f]:
            m |= (1 << four_idx[s])
        five_mask[i] = m
    
    # Index 5-subsets by their mask for quick lookup
    # Precompute: for each 4-subset index, list of 5-subset indices containing it
    four_to_fives_idx = [[] for _ in range(n4)]
    for i, f in enumerate(all5):
        for s in five_to_fours[f]:
            four_to_fives_idx[four_idx[s]].append(i)
    
    full_mask = (1 << n4) - 1
    
    def solve_s4511():
        """Determine S(4,5,11) via backtracking with symmetry breaking."""
        chosen = []
        covered = 0
        
        def backtrack(covered, chosen):
            if time.time() > deadline:
                return None
            if covered == full_mask:
                return chosen[:]
            if len(chosen) >= 66:
                return None
            
            # Find uncovered 4-subset with fewest valid candidates
            best_col = -1
            best_cands = []
            min_c = 999999
            
            for col in range(n4):
                if (covered >> col) & 1:
                    continue
                cands = []
                for fi in four_to_fives_idx[col]:
                    if five_mask[fi] & covered == 0:
                        cands.append(fi)
                if len(cands) < min_c:
                    min_c = len(cands)
                    best_col = col
                    best_cands = cands
                    if min_c == 0:
                        return None
                    if min_c == 1:
                        break
            
            if best_col == -1:
                return None
            
            # Sort by number of new 4-subsets covered
            best_cands.sort(key=lambda fi: bin(five_mask[fi] & ~covered).count('1'), reverse=True)
            
            for fi in best_cands:
                new_covered = covered | five_mask[fi]
                result = backtrack(new_covered, chosen + [fi])
                if result is not None:
                    return result
            return None
        
        # Symmetry breaking: fix first block as {0,1,2,3,4}
        first = (0, 1, 2, 3, 4)
        first_idx = five_to_idx[first]
        init_covered = five_mask[first_idx]
        result = backtrack(init_covered, [first_idx])
        
        if result is None:
            # Try without symmetry breaking
            result = backtrack(0, [])
        
        return result
    
    s4511 = solve_s4511()
    
    if s4511 is not None and len(s4511) == 66:
        # Convert indices to actual blocks
        blocks11 = [all5[i] for i in s4511]
        
        # Extend to S(5,6,12) on points 0..11
        blocks12 = []
        for b in blocks11:
            bset = set(b)
            # Block with point 11
            blocks12.append(tuple(sorted(bset | {11})))
            # Complement block (no point 11)
            comp = tuple(sorted(set(range(11)) - bset))
            blocks12.append(comp)
        
        # Verify
        if len(set(blocks12)) == 132:
            # Check no 5-subset appears in two blocks
            five_counts = {}
            valid = True
            for b in blocks12:
                for s in combinations(b, 5):
                    five_counts[s] = five_counts.get(s, 0) + 1
                    if five_counts[s] > 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [list(b) for b in blocks12]
    
    # Fallback: greedy packing
    points = list(range(12))
    all6 = list(combinations(points, 6))
    block_5sets = {b: set(combinations(b, 5)) for b in all6}
    
    covered = set()
    result = []
    import random
    shuffled = all6[:]
    random.shuffle(shuffled)
    for b in shuffled:
        if time.time() > deadline:
            break
        if not (block_5sets[b] & covered):
            result.append(b)
            covered.update(block_5sets[b])
    
    return [list(b) for b in result]
