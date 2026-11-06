import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    N = 16
    ALL_TRIPLES = list(combinations(range(N), 3))
    ALL_TRIPLES_SET = set(ALL_TRIPLES)
    
    # Precompute masks for all triples
    triple_masks = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in ALL_TRIPLES}
    
    def compute_alpha(triples):
        """
        Compute independence number.
        Returns the size of the largest subset of points containing no triple.
        """
        if not triples:
            return N
        
        # Convert current triples to a set of masks for fast lookup?
        # Or just a list of masks.
        t_masks = [triple_masks[t] for t in triples]
        
        # We want to find max popcount(S) such that no t in t_masks is a subset of S.
        # Iterate all 2^16 subsets
        best = 0
        for s in range(1, 1 << N):
            # Optimization: if popcount(s) <= best, skip
            # bin(s).count('1') is fast enough for 16 bits
            pc = bin(s).count('1')
            if pc <= best:
                continue
            
            # Check if s contains any triple
            # We can break early if we find one
            independent = True
            for tm in t_masks:
                if (tm & s) == tm:
                    independent = False
                    break
            
            if independent:
                best = pc
                # Optional: if best == N, break? No, max is N.
                # If best is high, we can't do better than N.
                # But we are iterating all.
        
        return best

    def is_valid_triple(triple, used_pairs):
        """Check if adding this triple would violate linearity (share > 1 point)."""
        a, b, c = triple
        # Pairs in this triple
        if (a, b) in used_pairs or (b, a) in used_pairs: return False
        if (a, c) in used_pairs or (c, a) in used_pairs: return False
        if (b, c) in used_pairs or (c, b) in used_pairs: return False
        return True

    def greedy_construct(seed_val):
        """Greedy construction with given seed."""
        rng = random.Random(seed_val)
        all_t = list(ALL_TRIPLES)
        rng.shuffle(all_t)
        
        used_pairs = set()
        triples = []
        
        for t in all_t:
            if is_valid_triple(t, used_pairs):
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((b, a))
                used_pairs.add((a, c))
                used_pairs.add((c, a))
                used_pairs.add((b, c))
                used_pairs.add((c, b))
                triples.append(t)
        
        return triples

    def local_search(triples, max_iterations=2000):
        """
        Local search to minimize independence number.
        """
        triples = list(triples)
        used_pairs = set()
        for t in triples:
            a, b, c = t
            used_pairs.add((a, b)); used_pairs.add((b, a))
            used_pairs.add((a, c)); used_pairs.add((c, a))
            used_pairs.add((b, c)); used_pairs.add((c, b))
        
        current_alpha = compute_alpha(triples)
        
        improved = True
        iters = 0
        rng = random.Random(42)
        
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            
            if time.time() - start_time > DEADLINE:
                break
                
            # Try adding a random valid triple
            candidates = [t for t in ALL_TRIPLES if is_valid_triple(t, used_pairs)]
            
            if candidates:
                # Try a few random additions
                for _ in range(min(10, len(candidates))):
                    t = rng.choice(candidates)
                    a, b, c = t
                    # Add it
                    used_pairs.add((a, b)); used_pairs.add((b, a))
                    used_pairs.add((a, c)); used_pairs.add((c, a))
                    used_pairs.add((b, c)); used_pairs.add((c, b))
                    triples.append(t)
                    
                    new_alpha = compute_alpha(triples)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                        break # Restart search with new state
                    else:
                        # Revert
                        triples.pop()
                        used_pairs.discard((a, b)); used_pairs.discard((b, a))
                        used_pairs.discard((a, c)); used_pairs.discard((c, a))
                        used_pairs.discard((b, c)); used_pairs.discard((c, b))
            
            # Try swapping: remove one triple, add another
            if len(triples) > 0:
                for _ in range(min(10, len(triples))):
                    idx = rng.randrange(len(triples))
                    old_t = triples[idx]
                    a, b, c = old_t
                    # Remove old
                    used_pairs.discard((a, b)); used_pairs.discard((b, a))
                    used_pairs.discard((a, c)); used_pairs.discard((c, a))
                    used_pairs.discard((b, c)); used_pairs.discard((c, b))
                    triples.pop(idx)
                    
                    # Find a valid triple to add
                    candidates = [t for t in ALL_TRIPLES if is_valid_triple(t, used_pairs)]
                    if candidates:
                        new_t = rng.choice(candidates)
                        a2, b2, c2 = new_t
                        # Add new
                        used_pairs.add((a2, b2)); used_pairs.add((b2, a2))
                        used_pairs.add((a2, c2)); used_pairs.add((c2, a2))
                        used_pairs.add((b2, c2)); used_pairs.add((c2, b2))
                        triples.append(new_t)
                        
                        new_alpha = compute_alpha(triples)
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            improved = True
                            break # Restart
                        else:
                            # Revert swap
                            triples.pop()
                            used_pairs.discard((a2, b2)); used_pairs.discard((b2, a2))
                            used_pairs.discard((a2, c2)); used_pairs.discard((c2, a2))
                            used_pairs.discard((b2, c2)); used_pairs.discard((c2, b2))
                            # Restore old
                            triples.append(old_t)
                            used_pairs.add((a, b)); used_pairs.add((b, a))
                            used_pairs.add((a, c)); used_pairs.add((c, a))
                            used_pairs.add((b, c)); used_pairs.add((c, b))
            
            # Try removing a triple (if it helps? Unlikely to reduce alpha, but might allow better structure later)
            # Usually we want more edges to reduce alpha.
            
        return triples, current_alpha

    best_triples = None
    best_alpha = N + 1
    
    # Multiple restarts
    for i in range(1000):
        if time.time() - start_time > DEADLINE:
            break
            
        seed = i * 12345 + 6789
        init_triples = greedy_construct(seed)
        
        if not init_triples:
            continue
            
        searched_triples, alpha = local_search(init_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = searched_triples
            
        # If we find a very low alpha, we can stop
        if best_alpha <= 5:
            break

    if best_triples is None:
        # Fallback: return a valid linear system (e.g., disjoint triples)
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples
