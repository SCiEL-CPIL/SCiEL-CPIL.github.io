"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
from collections import Counter

def construct():
    deadline = time.time() + 145
    n = 16
    rng = random.Random(12345)
    
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pair sets for all triples
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset({(a, b), (a, c), (b, c)})
    
    def is_linear(triples):
        seen_pairs = set()
        for t in triples:
            ps = triple_pairs[t]
            if seen_pairs & ps:
                return False
            seen_pairs |= ps
        return True
    
    def independence_number(triples):
        """Find the largest subset of points containing no triple."""
        if not triples:
            return n
        # Build a set of points that are in at least one triple
        # We need to find the maximum subset S of {0..15} such that no triple is entirely in S
        # This is equivalent to: find the maximum S where for every triple (a,b,c), not all of a,b,c are in S
        
        # Brute force over all subsets is 2^16 = 65536, which is fast
        # But we can do better with a smarter approach
        
        # Actually, let's use the complement: we want to minimize the number of points we remove
        # such that every triple has at least one point removed.
        # This is a hitting set / vertex cover problem on hypergraph.
        
        # For 16 points, we can try all subsets of size k from largest to smallest
        # and check if it's independent.
        
        # A faster approach: use the fact that if a set S is independent, then no triple is fully contained in S.
        # We can iterate over all 2^16 subsets and check independence.
        
        # Even faster: build a bitmask for each triple, then for each subset mask, check if any triple mask is a subset.
        
        # Let's precompute triple bitmasks
        triple_masks = [0] * len(triples)
        for i, t in enumerate(triples):
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks[i] = mask
        
        # Check all subsets from largest to smallest
        # Total subsets: 2^16 = 65536
        # For each subset, check if any triple is fully contained: O(|triples|)
        # Total: 65536 * |triples| which could be up to 65536 * 560 ≈ 36M operations - might be slow
        
        # Better: for each triple, mark all subsets that contain it as non-independent.
        # Use a boolean array of size 2^16.
        
        independent = [True] * (1 << n)
        for mask in triple_masks:
            # All supersets of this mask are non-independent
            # Iterate over all supersets
            # mask has 3 bits set. Supersets: set any of the remaining 13 bits.
            # That's 2^13 = 8192 per triple, times 560 triples = 4.5M operations
            remaining = ((1 << n) - 1) ^ mask
            sub = remaining
            while True:
                independent[mask | sub] = False
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        
        # Find the largest independent set
        for size in range(n, -1, -1):
            for mask in range(1 << n):
                if independent[mask] and bin(mask).count('1') == size:
                    return size
        return 0
    
    def independence_number_fast(triples):
        """Faster independence number using bitmask approach."""
        if not triples:
            return n
        
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # independent[s] = True if subset s contains no full triple
        independent = [True] * (1 << n)
        for mask in triple_masks:
            remaining = ((1 << n) - 1) ^ mask
            sub = remaining
            while True:
                independent[mask | sub] = False
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        
        # Find max popcount among independent sets
        best = 0
        for mask in range(1 << n):
            if independent[mask]:
                pc = bin(mask).count('1')
                if pc > best:
                    best = pc
                    if best == n:
                        break
        return best
    
    best_score = n  # Start with worst case
    best_triples = []
    
    # Try multiple random constructions
    for attempt in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Strategy: greedy with randomization
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        current = []
        used_pairs = set()
        
        for t in shuffled:
            ps = triple_pairs[t]
            if not (used_pairs & ps):
                current.append(t)
                used_pairs |= ps
        
        if not current:
            continue
        
        score = independence_number_fast(current)
        
        if score < best_score:
            best_score = score
            best_triples = current[:]
            if best_score <= 6:  # Good enough, can stop early
                break
    
    # Try to improve best_triples with local search
    if best_triples:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_triples = best_triples[:]
            current_score = best_score
            
            # Try swapping each triple with another
            used_pairs = set()
            for t in current_triples:
                used_pairs |= triple_pairs[t]
            
            # Find all triples not in current_triples
            current_set = set(map(tuple, current_triples))
            
            for i, t in enumerate(current_triples):
                if time.time() > deadline - 5:
                    break
                # Try removing t and adding another
                new_used = set()
                for j, t2 in enumerate(current_triples):
                    if j != i:
                        new_used |= triple_pairs[t2]
                
                for t_new in all_triples:
                    if t_new in current_set:
                        continue
                    ps_new = triple_pairs[t_new]
                    if not (new_used & ps_new):
                        new_triples = current_triples[:]
                        new_triples[i] = t_new
                        new_score = independence_number_fast(new_triples)
                        if new_score < current_score:
                            best_triples = new_triples
                            best_score = new_score
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    if best_triples:
        return best_triples
    else:
        # Fallback: return the initial simple construction
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
