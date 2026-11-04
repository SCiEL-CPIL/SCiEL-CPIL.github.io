"""Greedy covering with fast gain computation, iterative pruning, and deadline guard."""
import itertools
import time

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    all_tsubs_set = set(all_tsubs)

    # Precompute for each point x and each t-1 combination, the t-subset formed
    # Actually, for gain computation we need: given a partial block P (size < k),
    # for each candidate x not in P, count how many uncovered t-subsets are covered
    # by adding x to P. A t-subset S is covered by P∪{x} iff S ⊆ P∪{x} and x∈S.
    # So S must contain x and S\{x} ⊆ P.
    # Number of such S = number of (t-1)-subsets of P that, together with x, form an uncovered t-subset.
    
    uncovered = set(all_tsubs)
    blocks = []
    
    # Index: for each (t-1)-tuple, store which t-subsets contain it (as sorted tuples)
    # Actually, for gain we need: given partial P, for candidate x, count uncovered S such that x∈S and S\{x}⊆P.
    # Precompute: for each point x, a dict mapping (t-1)-subsets of other points to the list of t-subsets containing x and that (t-1)-subset.
    # That's too heavy. Instead, for each candidate x and partial P, the t-subsets covered are:
    #   {S : x∈S, S\{x}⊆P, S uncovered}
    # The number of (t-1)-subsets of P is C(|P|, t-1), which is manageable for small |P|.
    
    while uncovered:
        if time.monotonic() > deadline:
            break
        # Start block with first uncovered t-subset
        seed = min(uncovered)
        block = list(seed)
        
        while len(block) < k:
            if time.monotonic() > deadline:
                break
            # Find best x to add
            best_x = -1
            best_gain = -1
            block_set = set(block)
            # Precompute (t-1)-subsets of block
            t1_subs = list(itertools.combinations(block, t-1))
            for x in points:
                if x in block_set:
                    continue
                # Count uncovered t-subsets S where x∈S and S\{x}⊆block
                gain = 0
                for s in t1_subs:
                    candidate = tuple(sorted(s + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                break
            block.append(best_x)
        
        if len(block) < k:
            # Can't complete block, just use what we have (shouldn't happen if k<=v)
            break
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    # Pruning pass: remove redundant blocks
    if time.monotonic() < deadline - 5:
        # For each t-subset, count how many blocks cover it
        cover_count = {ts: 0 for ts in all_tsubs_set}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts] += 1
        
        # Remove blocks that are not essential
        kept = []
        for b in blocks:
            essential = False
            for ts in itertools.combinations(b, t):
                if cover_count[ts] == 1:
                    essential = True
                    break
            if essential:
                kept.append(b)
            else:
                for ts in itertools.combinations(b, t):
                    cover_count[ts] -= 1
        
        if len(kept) < len(blocks):
            blocks = kept
    
    # Verify and return
    if not uncovered:
        return blocks
    # If deadline hit before completion, return what we have (may be invalid)
    return blocks
