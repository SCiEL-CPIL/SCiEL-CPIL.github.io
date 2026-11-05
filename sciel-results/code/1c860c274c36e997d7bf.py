import time
import random
from itertools import combinations

# Global deadline
START_TIME = time.time()
DEADLINE = 110.0  # Safety margin before 120s

# Precompute popcount for all 16-bit masks
POPCOUNT = [0] * (1 << 16)
for i in range(1, 1 << 16):
    POPCOUNT[i] = POPCOUNT[i >> 1] + (i & 1)

def construct():
    global START_TIME
    START_TIME = time.time()
    n = 16
    
    def get_pairs(triple):
        a, b, c = triple
        return [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]

    def is_linear_fast(triple, used_pairs):
        for p in get_pairs(triple):
            if p in used_pairs:
                return False
        return True

    def add_triple(triples, used_pairs, triple):
        triples.append(list(triple))
        for p in get_pairs(triple):
            used_pairs.add(p)

    def remove_triple(triples, used_pairs, idx):
        triple = triples.pop(idx)
        for p in get_pairs(triple):
            used_pairs.discard(p)
        return triple

    def compute_alpha(triples):
        if not triples:
            return n
        
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        # We want max |I| such that no triple is subset of I.
        # Iterate sizes from n down to 1.
        # For each size, check if any mask of that size is independent.
        
        # Optimization: If we find one, return size.
        # This is efficient if alpha is large.
        # If alpha is small, it's slow.
        # But for dense packs, alpha is small.
        # So this might be slow.
        
        # Alternative: Iterate all masks, track max.
        # This is consistent O(2^16 * |T|).
        
        best = 0
        # Use the precomputed popcount
        # Iterate all masks
        for Imask in range(1 << n):
            sz = POPCOUNT[Imask]
            if sz <= best:
                continue
            valid = True
            for tm in triple_masks:
                if (Imask & tm) == tm:
                    valid = False
                    break
            if valid:
                best = sz
                if best == n:
                    break
        return best

    def greedy_construct(seed_rng):
        rng = random.Random(seed_rng)
        triples = []
        used_pairs = set()
        
        all_triples = list(combinations(range(n), 3))
        rng.shuffle(all_triples)
        
        for t in all_triples:
            if is_linear_fast(t, used_pairs):
                add_triple(triples, used_pairs, t)
            if time.time() - START_TIME > DEADLINE:
                break
                
        return triples, used_pairs

    def local_search(triples, used_pairs, seed_rng):
        rng = random.Random(seed_rng)
        best_triples = triples[:]
        best_used_pairs = used_pairs.copy()
        best_alpha = compute_alpha(best_triples)
        
        all_triples = list(combinations(range(n), 3))
        
        # Limit iterations and alpha calls
        max_iterations = 20
        max_candidates_per_iter = 3
        
        for iteration in range(max_iterations):
            if time.time() - START_TIME > DEADLINE:
                break
            if not best_triples:
                break
                
            # Remove a random triple
            idx = rng.randrange(len(best_triples))
            removed = remove_triple(best_triples, best_used_pairs, idx)
            
            # Find valid candidates
            candidates = []
            for t in all_triples:
                if is_linear_fast(t, best_used_pairs):
                    candidates.append(t)
            
            if not candidates:
                # Restore
                add_triple(best_triples, best_used_pairs, removed)
                continue
            
            rng.shuffle(candidates)
            improved = False
            
            for t in candidates[:max_candidates_per_iter]:
                # Try adding t
                add_triple(best_triples, best_used_pairs, t)
                new_alpha = compute_alpha(best_triples)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    improved = True
                    # Keep this state
                    break
                else:
                    # Revert
                    # Need to remove the last added triple
                    # Since we added it to the end, pop it
                    # But we need to clean up used_pairs
                    # Let's just remove by value or index
                    # We added it, so it's at the end
                    last_idx = len(best_triples) - 1
                    remove_triple(best_triples, best_used_pairs, last_idx)
            
            if not improved:
                # Restore removed triple
                add_triple(best_triples, best_used_pairs, removed)
        
        return best_triples, best_used_pairs, best_alpha

    best_result = None
    best_score = n + 1
    best_used_pairs = None
    
    # Limit restarts
    max_restarts = 20
    
    for seed in range(max_restarts):
        if time.time() - START_TIME > DEADLINE:
            break
        
        triples, used_pairs = greedy_construct(seed)
        if not triples:
            continue
        
        triples, used_pairs, alpha = local_search(triples, used_pairs, seed)
        
        if alpha < best_score:
            best_score = alpha
            best_result = triples[:]
            best_used_pairs = used_pairs.copy()
            
        if best_score <= 5:
            break

    if best_result is None:
        # Fallback: simple partition
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_result
