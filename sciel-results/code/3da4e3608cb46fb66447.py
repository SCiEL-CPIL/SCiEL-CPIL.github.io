import itertools
import random

def construct(v, k, t):
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    uncovered = set(all_triples)
    blocks = []
    
    # Precompute: for each triple, list of all k-sets containing it
    # Actually, we'll generate on the fly
    
    while uncovered:
        # Pick a seed triple
        seed = min(uncovered)  # or random.choice(list(uncovered))
        
        best_block = None
        best_count = -1
        
        # Generate all (k-t)-sets from the remaining points
        remaining = [x for x in range(v) if x not in seed]
        for ext in itertools.combinations(remaining, k - t):
            block = tuple(sorted(seed + ext))
            # Count uncovered triples in this block
            count = 0
            for triple in itertools.combinations(block, t):
                if triple in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = block
        
        blocks.append(list(best_block))
        # Remove covered triples
        for triple in itertools.combinations(best_block, t):
            uncovered.discard(triple)
    
    return blocks
