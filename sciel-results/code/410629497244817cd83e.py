import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    N = 16
    ALL_TRIPLES = list(combinations(range(N), 3))
    NUM_TRIPLES = len(ALL_TRIPLES)
    
    def triple_to_mask(t):
        """Convert a triple (a,b,c) to a bitmask."""
        return (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    def compute_alpha_fast(triple_masks):
        """
        Fast alpha computation using precomputed 'contains triple' table.
        For each subset s (0..2^16-1), precompute whether it contains any triple.
        Then alpha is the max bit_count among independent subsets.
        """
        if not triple_masks:
            return N
        
        # Precompute: for each subset, does it contain any triple?
        # contains[s] = True if s contains at least one triple from triple_masks
        # Use the fact that contains[s] = True if any triple mask is a subset of s.
        # We can compute this via: for each triple mask m, mark all supersets.
        # But that's 2^16 * num_triples which is too slow.
        
        # Alternative: iterate subsets and check directly, but with numpy for speed.
        # For 16 points, 2^16 = 65536 subsets. If we have ~20-50 triples,
        # checking each subset against all triples is 65536 * 50 = 3.2M ops.
        # In Python this is ~1-2s. We need faster.
        
        # Better: precompute for each point pair/triple which triples they're in.
        # Actually, let's use a different approach:
        # For each subset s, s is independent iff for every triple (a,b,c), 
        # not all of a,b,c are in s.
        
        # Key optimization: use the fact that if s is independent, all subsets are.
        # So we can do a BFS/DFS from the full set, removing points.
        # Or: iterate from large subsets down.
        
        # Most efficient for 16 points: precompute a boolean array.
        # contains[s] = 1 if s contains any triple.
        # Start with all False, then for each triple mask m, set contains[s]=True 
        # for all s that are supersets of m.
        
        # Number of supersets of a 3-element set in 16 points: 2^13 = 8192
        # So for T triples, we do T * 8192 operations.
        # For T=30, that's 245K operations. Fast enough.
        
        contains = bytearray(1 << N)  # 0 = independent, 1 = contains triple
        
        for mask in triple_masks:
            # Mark all supersets of mask
            # Supersets of mask: s where (s & mask) == mask
            # Equivalently: s = mask | extra, where extra is any subset of ~mask
            complement = (~mask) & ((1 << N) - 1)
            # Iterate all subsets of complement
            extra = complement
            while True:
                s = mask | extra
                contains[s] = 1
                if extra == 0:
                    break
                extra = (extra - 1) & complement
        
        # Now find the largest independent set
        best = 0
        best_set = 0
        for s in range(1, 1 << N):
            if contains[s] == 0:
                bc = s.bit_count()
                if bc > best:
                    best = bc
                    best_set = s
        return best, best_set
    
    def is_valid_triple(triple, used_pairs):
        """Check if adding this triple would violate linearity."""
        a, b, c = triple
        pairs = [(a, b), (a, c), (b, c)]
        for p in pairs:
            if p in used_pairs:
                return False
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
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                triples.append(t)
        
        return triples
    
    def local_search(triples, max_iterations=20000, time_limit=None):
        """
        Local search: try to improve by adding, removing, or swapping triples.
        Goal: minimize independence number.
        Uses the specific independent set found to guide search.
        """
        triples = list(triples)
        used_pairs = set()
        for t in triples:
            a, b, c = t
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))
        
        triple_masks = [triple_to_mask(t) for t in triples]
        current_alpha, current_best_set = compute_alpha_fast(triple_masks)
        
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            improved = False
            iters += 1
            
            # Get the independent set to target
            # Points in the independent set
            indep_points = set()
            s = current_best_set
            for p in range(N):
                if s & (1 << p):
                    indep_points.add(p)
            
            # Strategy 1: Add a triple that intersects the independent set
            # to break it. We want a triple that has at least one point in indep_points.
            # This reduces the independent set size.
            candidates = [t for t in ALL_TRIPLES if t not in triples and is_valid_triple(t, used_pairs)]
            
            if candidates:
                # Prioritize triples that intersect the independent set
                intersecting = [t for t in candidates if any(p in indep_points for p in t)]
                non_intersecting = [t for t in candidates if not any(p in indep_points for p in t)]
                
                # Try intersecting triples first
                to_try = intersecting + non_intersecting
                rng = random.Random(iters * 31 + 7)
                
                for t in to_try[:min(10, len(to_try))]:
                    a, b, c = t
                    used_pairs.add((a, b))
                    used_pairs.add((a, c))
                    used_pairs.add((b, c))
                    triples.append(t)
                    triple_masks.append(triple_to_mask(t))
                    
                    new_alpha, new_best_set = compute_alpha_fast(triple_masks)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        current_best_set = new_best_set
                        improved = True
                    else:
                        # Revert
                        triples.pop()
                        triple_masks.pop()
                        used_pairs.discard((a, b))
                        used_pairs.discard((a, c))
                        used_pairs.discard((b, c))
                    
                    if time_limit is not None and time.time() - start_time > time_limit:
                        break
            
            # Strategy 2: Swap - remove a triple, add a better one
            if len(triples) > 0 and (time_limit is None or time.time() - start_time < time_limit):
                rng2 = random.Random(iters * 17 + 3)
                for _ in range(min(5, len(triples))):
                    idx = rng2.randrange(len(triples))
                    old_t = triples[idx]
                    a, b, c = old_t
                    used_pairs.discard((a, b))
                    used_pairs.discard((a, c))
                    used_pairs.discard((b, c))
                    triples.pop(idx)
                    triple_masks.pop(idx)
                    
                    # Find valid triples to add, prioritizing those hitting indep_points
                    candidates2 = [t for t in ALL_TRIPLES if t not in triples and is_valid_triple(t, used_pairs)]
                    intersecting2 = [t for t in candidates2 if any(p in indep_points for p in t)]
                    non_intersecting2 = [t for t in candidates2 if not any(p in indep_points for p in t)]
                    to_try2 = intersecting2 + non_intersecting2
                    
                    if to_try2:
                        new_t = rng2.choice(to_try2)
                        a2, b2, c2 = new_t
                        used_pairs.add((a2, b2))
                        used_pairs.add((a2, c2))
                        used_pairs.add((b2, c2))
                        triples.append(new_t)
                        triple_masks.append(triple_to_mask(new_t))
                        
                        new_alpha, new_best_set = compute_alpha_fast(triple_masks)
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            current_best_set = new_best_set
                            improved = True
                        else:
                            # Revert swap
                            triples.pop()
                            triple_masks.pop()
                            used_pairs.discard((a2, b2))
                            used_pairs.discard((a2, c2))
                            used_pairs.discard((b2, c2))
                            # Restore old triple
                            triples.append(old_t)
                            triple_masks.append(triple_to_mask(old_t))
                            used_pairs.add((a, b))
                            used_pairs.add((a, c))
                            used_pairs.add((b, c))
                    else:
                        # Restore old triple
                        triples.append(old_t)
                        triple_masks.append(triple_to_mask(old_t))
                        used_pairs.add((a, b))
                        used_pairs.add((a, c))
                        used_pairs.add((b, c))
                    
                    if time_limit is not None and time.time() - start_time > time_limit:
                        break
        
        return triples, current_alpha
    
    best_triples = None
    best_indep = N + 1
    
    # Phase 1: Greedy constructions
    num_greedy = 300
    for i in range(num_greedy):
        if time.time() - start_time > DEADLINE:
            break
        
        triples = greedy_construct(i * 104729 + 13)
        if not triples:
            continue
        
        triple_masks = [triple_to_mask(t) for t in triples]
        alpha, _ = compute_alpha_fast(triple_masks)
        
        if alpha < best_indep:
            best_indep = alpha
            best_triples = triples
        
        if best_indep <= 3:
            break
    
    # Phase 2: Local search on best solutions
    if best_triples is not None and time.time() - start_time < DEADLINE:
        # Run local search on the best solution with generous time budget
        time_limit_ls = min(DEADLINE - 5.0, time.time() - start_time + 80.0)
        improved_triples, improved_alpha = local_search(best_triples, max_iterations=100000, time_limit=time_limit_ls)
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triples = improved_triples
    
    # Phase 3: More greedy + local search iterations
    for i in range(num_greedy, num_greedy + 500):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        triples = greedy_construct(i * 104729 + 13)
        if not triples:
            continue
        
        triple_masks = [triple_to_mask(t) for t in triples]
        alpha, _ = compute_alpha_fast(triple_masks)
        
        if alpha < best_indep:
            best_indep = alpha
            best_triples = triples
            # Run local search on this promising solution
            if time.time() - start_time < DEADLINE - 15:
                time_limit_ls = min(DEADLINE - 10.0, time.time() - start_time + 30.0)
                improved_triples, improved_alpha = local_search(best_triples, max_iterations=20000, time_limit=time_limit_ls)
                if improved_alpha < best_indep:
                    best_indep = improved_alpha
                    best_triples = improved_triples
        
        if best_indep <= 3:
            break
    
    # Phase 4: Final polish on the best solution
    if best_triples is not None and time.time() - start_time < DEADLINE - 5:
        time_limit_final = DEADLINE - 3.0
        improved_triples, improved_alpha = local_search(best_triples, max_iterations=100000, time_limit=time_limit_final)
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triples = improved_triples
    
    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return [list(t) for t in best_triples]
