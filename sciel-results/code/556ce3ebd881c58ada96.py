"""Max K(2,2,2)-free triple system on 16 points via greedy + local search."""
import time
import random
import itertools
import sys

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    random.seed(42)
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute: for each triple, which other triples form a K(2,2,2) completion
    # A K(2,2,2) has 3 disjoint pairs, 8 transversal triples.
    # For a given triple t, it participates in K(2,2,2) if there exist 2 other points
    # such that the 8 transversals include t.
    # 
    # Better approach: enumerate all K(2,2,2) configurations (sets of 8 triples)
    # and for each triple, record which K(2,2,2)s it belongs to.
    
    # Enumerate all ways to choose 3 disjoint pairs from 16 points
    # Then for each such choice, the 8 transversal triples form a "block"
    
    # This is expensive: C(16,2)*C(14,2)*C(12,2)/6 is large but manageable
    # Actually, let's think differently.
    
    # For each triple t = {a,b,c}, a K(2,2,2) containing t requires:
    # - a is in one pair, b in another, c in a third (or some permutation)
    # - The other element of each pair is chosen from remaining 13 points
    # 
    # Let's enumerate: for triple (a,b,c), choose:
    # - which point pairs with a: from remaining 13 points (excluding b,c)
    # - which point pairs with b: from remaining 12 points (excluding a,c and a's partner)
    # - which point pairs with c: from remaining 11 points (excluding a,b and their partners)
    # But this overcounts because the assignment of a,b,c to the three pairs can vary.
    
    # Simpler: just enumerate all 3-disjoint-pair configurations and build the 8-triple blocks.
    
    # Number of ways to pick 3 disjoint unordered pairs from 16 points:
    # C(16,2) * C(14,2) * C(12,2) / 3! = 120 * 91 * 66 / 6 = 117,360
    # That's manageable.
    
    # For each configuration, compute the 8 transversal triples and store them as a frozenset of indices.
    
    # blocks: list of frozensets of triple indices
    # triple_blocks: for each triple index, list of block indices it belongs to
    
    def get_blocks():
        blocks = []
        triple_block_map = [[] for _ in range(num_triples)]
        
        # Generate all ways to choose 3 disjoint pairs
        # We can do this by iterating over combinations of 6 points, then partitioning into 3 pairs
        # But that's also complex. Let's just iterate.
        
        points = list(range(N))
        
        # For efficiency, iterate over all 3-tuples of disjoint pairs
        for p1 in itertools.combinations(points, 2):
            remaining1 = [x for x in points if x not in p1]
            for p2 in itertools.combinations(remaining1, 2):
                remaining2 = [x for x in remaining1 if x not in p2]
                for p3 in itertools.combinations(remaining2, 2):
                    # We have three disjoint pairs: p1, p2, p3
                    # The 8 transversal triples: one element from each pair
                    a, b = p1
                    c, d = p2
                    e, f = p3
                    transversals = set()
                    for x in (a, b):
                        for y in (c, d):
                            for z in (e, f):
                                t = tuple(sorted((x, y, z)))
                                transversals.add(t)
                    
                    block = frozenset(triple_to_idx[t] for t in transversals)
                    blocks.append(block)
                    for idx in block:
                        triple_block_map[idx].append(len(blocks) - 1)
        
        return blocks, triple_block_map
    
    print("Building blocks...", file=sys.stderr)
    blocks, triple_block_map = get_blocks()
    print(f"Built {len(blocks)} blocks", file=sys.stderr)
    
    # Greedy construction with multiple restarts
    best_size = 0
    best_in_S = None
    
    num_restarts = 200  # will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(all_triples)
        # Re-index: we shuffled all_triples, so recompute triple_to_idx
        # Actually, let's not shuffle all_triples itself, but shuffle the order of iteration
        order = list(range(num_triples))
        random.shuffle(order)
        
        in_S = [False] * num_triples
        current_size = 0
        
        for idx in order:
            # Check if adding idx creates a full block
            valid = True
            for blk_idx in triple_block_map[idx]:
                block = blocks[blk_idx]
                all_present = True
                for b_idx in block:
                    if b_idx != idx and not in_S[b_idx]:
                        all_present = False
                        break
                if all_present:
                    valid = False
                    break
            
            if valid:
                in_S[idx] = True
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_in_S = in_S[:]
            print(f"Restart {restart}: size {best_size}", file=sys.stderr)
    
    # Local search: try to add triples that were excluded, or swap
    if best_in_S is not None:
        in_S = best_in_S[:]
        current_size = best_size
        
        # Try to add each excluded triple
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in range(num_triples):
                if time.time() > deadline - 5:
                    break
                if in_S[idx]:
                    continue
                # Check if we can add idx
                valid = True
                for blk_idx in triple_block_map[idx]:
                    block = blocks[blk_idx]
                    all_present = True
                    for b_idx in block:
                        if b_idx != idx and not in_S[b_idx]:
                            all_present = False
                            break
                    if all_present:
                        valid = False
                        break
                if valid:
                    in_S[idx] = True
                    current_size += 1
                    improved = True
        
        if current_size > best_size:
            best_size = current_size
            best_in_S = in_S[:]
            print(f"After local search: size {best_size}", file=sys.stderr)
    
    # Convert to list of triples
    result = [list(all_triples[i]) for i in range(num_triples) if best_in_S[i]]
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
