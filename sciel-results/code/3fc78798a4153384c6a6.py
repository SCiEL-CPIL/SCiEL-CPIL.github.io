"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = 16
    # Precompute all C(16,3) = 560 triples
    all_triples = []
    triple_to_idx = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                idx = len(all_triples)
                all_triples.append((a, b, c))
                triple_to_idx[frozenset((a, b, c))] = idx
    
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2) patterns
    # A pattern is defined by 6 points and a partition into 3 pairs.
    # The 8 transversals are the triples picking one from each pair.
    # We represent a pattern as a set of 8 triple indices.
    
    patterns = []
    triple_to_patterns = defaultdict(list)
    
    # Generate all 6-point subsets
    # C(16, 6) = 8008
    # For each, 15 pairings
    # Total ~120k patterns.
    
    # To speed up, we can iterate through combinations of 6 points.
    # But 8008 * 15 * 8 operations is ~1M, which is fine.
    
    # Helper to get triple index from 3 points
    def get_idx(p1, p2, p3):
        return triple_to_idx[frozenset((p1, p2, p3))]
        
    # List of pairings for 6 points (indices 0..5)
    pairings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    import itertools
    for pts in itertools.combinations(range(n), 6):
        for pairing in pairings:
            # Get the 3 pairs
            p1a, p1b = pts[pairing[0][0]], pts[pairing[0][1]]
            p2a, p2b = pts[pairing[1][0]], pts[pairing[1][1]]
            p3a, p3b = pts[pairing[2][0]], pts[pairing[2][1]]
            
            # Generate 8 transversals
            transversals = []
            for mask in range(8):
                t1 = p1a if (mask & 1) else p1b
                t2 = p2a if (mask & 2) else p2b
                t3 = p3a if (mask & 4) else p3b
                idx = get_idx(t1, t2, t3)
                transversals.append(idx)
            
            # Add pattern
            pat_idx = len(patterns)
            patterns.append(transversals)
            for t_idx in transversals:
                triple_to_patterns[t_idx].append(pat_idx)
                
    num_patterns = len(patterns)
    
    # State:
    # in_set: boolean array of size num_triples, True if triple is in the set
    # missing_count: array of size num_patterns, count of missing triples in that pattern
    
    best_set = None
    best_size = 0
    
    # Greedy Removal + Add Back + Local Search
    for trial in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        in_set = [True] * num_triples
        missing_count = [0] * num_patterns
        
        # Greedy Removal
        # We need to find triples that are in the most "active" (missing_count == 0) patterns.
        # We can maintain a list of active patterns.
        active_patterns = list(range(num_patterns))
        
        # To find the triple with max coverage of active patterns,
        # we can iterate through triples and count how many active patterns they are in.
        # But doing this from scratch every removal is O(num_triples * avg_patterns_per_triple).
        # num_triples = 560. avg_patterns ~ 120000*8/560 ~ 1700.
        # 560 * 1700 = ~1M ops per removal.
        # We might need to remove ~100-200 triples.
        # 200 * 1M = 200M ops. This might be too slow in Python.
        
        # Optimization:
        # Instead of recalculating, we can just pick a random active pattern,
        # and remove one of its triples? No, that's not greedy.
        # Let's try a simpler heuristic:
        # While there are active patterns:
        #   Pick a random active pattern P.
        #   Pick a random triple T in P.
        #   Remove T.
        #   Update missing_count for all patterns containing T.
        #   Remove patterns with missing_count > 0 from active list?
        
        # This is randomized greedy. It might be slower to converge but each step is O(1) or O(deg(T)).
        # deg(T) is ~1700.
        # Number of removals ~ ?
        # Let's try this.
        
        active_set = set(range(num_patterns))
        
        while active_set:
            if time.time() > deadline - 10:
                break
            # Pick a random active pattern
            pat_idx = random.choice(list(active_set))
            # Pick a random triple from this pattern
            t_idx = random.choice(patterns[pat_idx])
            
            # Remove t_idx
            if in_set[t_idx]:
                in_set[t_idx] = False
                # Update missing_count for all patterns containing t_idx
                for p in triple_to_patterns[t_idx]:
                    missing_count[p] += 1
                    if missing_count[p] > 0 and p in active_set:
                        # It was active (0), now it's >0, so it's no longer a violation
                        # Wait, active_set contains patterns with missing_count == 0.
                        # So if missing_count becomes 1, it leaves active_set.
                        active_set.discard(p)
                        
        # Now all patterns have missing_count >= 1.
        
        # Add Back Phase
        # Try to add back triples that were removed.
        # A triple T can be added if for all P in triple_to_patterns[T], missing_count[P] > 1.
        # i.e., min(missing_count[P]) > 1.
        
        removed_triples = [i for i in range(num_triples) if not in_set[i]]
        random.shuffle(removed_triples)
        
        for t_idx in removed_triples:
            if time.time() > deadline - 10:
                break
            # Check if safe to add
            safe = True
            min_mc = float('inf')
            for p in triple_to_patterns[t_idx]:
                if missing_count[p] <= 1:
                    safe = False
                    break
                if missing_count[p] < min_mc:
                    min_mc = missing_count[p]
            
            if safe:
                in_set[t_idx] = True
                for p in triple_to_patterns[t_idx]:
                    missing_count[p] -= 1
                    # If missing_count becomes 0, it becomes a violation!
                    # But we checked safe, so min_mc > 1, so new count >= 1.
                    # So it won't become active (violation).
                    
        current_size = sum(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = in_set[:]
            
        # Local Search: Kick
        # Try to remove a random triple from the current set, then add back as many as possible.
        # If the size increases, keep the change.
        
        for _ in range(50): # 50 kicks
            if time.time() > deadline - 10:
                break
            
            # Pick a random triple in the set to remove
            in_indices = [i for i in range(num_triples) if in_set[i]]
            if not in_indices:
                break
            kick_idx = random.choice(in_indices)
            
            # Save state
            old_in_set = in_set[:]
            old_missing = missing_count[:]
            
            # Remove kick_idx
            in_set[kick_idx] = False
            for p in triple_to_patterns[kick_idx]:
                missing_count[p] += 1
                
            # Try to add back removed triples (including kick_idx)
            # We need a list of currently removed triples.
            # This is expensive to maintain.
            # Let's just iterate through all triples not in set?
            # Or keep a list of removed triples updated.
            
            # For simplicity and speed, let's just try to add back kick_idx first?
            # No, the point of a kick is to allow a different set of triples to be added.
            
            # Let's maintain a list of removed triples.
            # Actually, we can just iterate through all triples not in in_set.
            # That's O(num_triples) per kick.
            
            # Let's shuffle the order of trying to add back.
            candidates = [i for i in range(num_triples) if not in_set[i]]
            random.shuffle(candidates)
            
            for t_idx in candidates:
                # Check if safe
                safe = True
                for p in triple_to_patterns[t_idx]:
                    if missing_count[p] <= 1:
                        safe = False
                        break
                if safe:
                    in_set[t_idx] = True
                    for p in triple_to_patterns[t_idx]:
                        missing_count[p] -= 1
                        
            new_size = sum(in_set)
            
            if new_size > current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = in_set[:]
            else:
                # Revert
                in_set = old_in_set
                missing_count = old_missing
                # current_size is unchanged
                
    # Convert best_set to list of triples
    result = []
    if best_set:
        for i in range(num_triples):
            if best_set[i]:
                result.append(list(all_triples[i]))
    else:
        # Fallback: empty set or something valid?
        # The task requires a valid set.
        # If best_set is None, we might have timed out during the first trial?
        # Let's ensure we return at least one valid set.
        # A single triple is always valid.
        result = [list(all_triples[0])] if num_triples > 0 else []
        
    return result
# EVOLVE-BLOCK-END
