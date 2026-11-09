import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

# Precompute affine lines in AG(4,2)
# Points are 0..15, interpreted as 4-bit integers
def _get_affine_lines():
    lines = set()
    for x in range(16):
        for y in range(16):
            if x == y:
                continue
            z = x ^ y
            if x < y < z:
                lines.add((x, y, z))
            elif x < z < y:
                lines.add((x, z, y))
            elif y < x < z:
                lines.add((y, x, z))
            elif y < z < x:
                lines.add((y, z, x))
            elif z < x < y:
                lines.add((z, x, y))
            elif z < y < x:
                lines.add((z, y, x))
    return list(lines)

AFFINE_LINES = _get_affine_lines()

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    # Optimization: use a set of pairs
    pair_to_triple = {}
    for t in triples:
        a, b, c = t
        pairs = [(a,b), (a,c), (b,c)]
        for p in pairs:
            if p in pair_to_triple:
                # Check if it's the same triple? 
                # If different triple, they share 2 points.
                # If same triple, it's fine (but we assume unique triples)
                # Actually, if pair is in pair_to_triple, and it's a different triple, invalid.
                # We assume no duplicate triples in the list.
                return False
            pair_to_triple[p] = t
    return True

def _independence_number(triples, n=16):
    """Compute the largest subset of points containing no triple."""
    triple_set = set(tuple(sorted(t)) for t in triples)
    
    def is_independent(subset):
        s = set(subset)
        for t in triple_set:
            if t[0] in s and t[1] in s and t[2] in s:
                return False
        return True

    # Check from 16 down to 1
    # For n=16, this is feasible if we prune
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(range(n), size):
            if is_independent(subset):
                return size
    return 0

def _find_cap(triples, n=16, target_size=8):
    """Find a cap of size target_size."""
    triple_set = set(tuple(sorted(t)) for t in triples)
    
    def is_independent(subset):
        s = set(subset)
        for t in triple_set:
            if t[0] in s and t[1] in s and t[2] in s:
                return False
        return True

    for subset in combinations(range(n), target_size):
        if is_independent(subset):
            return list(subset)
    return None

def _construct_initial_system(seed=0):
    """Construct a random maximal packing of affine lines."""
    rng = random.Random(seed)
    lines = AFFINE_LINES[:]
    rng.shuffle(lines)
    
    triples = []
    used_pairs = set()
    
    for line in lines:
        a, b, c = line
        pairs = [(a,b), (a,c), (b,c)]
        if all(p not in used_pairs for p in pairs):
            triples.append(list(line))
            used_pairs.update(pairs)
            
            # Max triples is 40
            if len(triples) >= 40:
                break
    return triples

def _local_search_cap_breaking(triples, n=16, max_iter=1000, seed=0):
    """
    Local search that specifically tries to break caps of size 8 by 
    swapping affine lines for non-affine triples within the cap.
    """
    rng = random.Random(seed)
    triples = [list(t) for t in triples]
    best_triples = [list(t) for t in triples]
    best_indep = _independence_number(triples, n)
    
    if best_indep <= 7:
        return best_triples

    for _ in range(max_iter):
        # Find a cap of size 8
        cap = _find_cap(triples, n, 8)
        if cap is None:
            # No cap of size 8, so indep < 8
            # Recalculate to be sure
            curr_indep = _independence_number(triples, n)
            if curr_indep < best_indep:
                best_indep = curr_indep
                best_triples = [list(t) for t in triples]
            break
        
        # Try to break this cap
        # Pick a random triple from the cap
        # There are C(8,3) = 56 triples in the cap
        cap_triples = list(combinations(cap, 3))
        rng.shuffle(cap_triples)
        
        improved = False
        for t in cap_triples:
            t = list(t)
            # Check if t is already in triples
            if t in triples:
                continue
            
            # Find conflicts: triples in current system sharing 2 points with t
            conflicts = []
            for existing in triples:
                common = set(t) & set(existing)
                if len(common) == 2:
                    conflicts.append(existing)
            
            # If no conflicts, just add it?
            # But we must maintain linearity.
            # If no conflicts, adding t is safe.
            # However, we might have removed lines earlier, so we have "space".
            
            new_triples = [list(x) for x in triples if x not in conflicts] + [t]
            
            # Check linearity of new system
            if not _is_linear(new_triples):
                continue
                
            new_indep = _independence_number(new_triples, n)
            
            if new_indep < best_indep:
                best_indep = new_indep
                best_triples = new_triples
                triples = new_triples
                improved = True
                break # Restart search with new system
            
        if not improved:
            # Try a random mutation: replace a random affine line with a random non-affine triple?
            # Or just random restart?
            # Let's try a random replacement of a line
            if triples:
                idx = rng.randint(0, len(triples)-1)
                old_line = triples[idx]
                # Generate a random triple
                new_t = list(rng.sample(range(n), 3))
                new_t.sort()
                
                # Check if new_t conflicts with others (excluding old_line)
                valid = True
                for j, existing in enumerate(triples):
                    if j == idx:
                        continue
                    if len(set(new_t) & set(existing)) > 1:
                        valid = False
                        break
                
                if valid:
                    new_triples = triples[:idx] + [new_t] + triples[idx+1:]
                    if _is_linear(new_triples):
                        new_indep = _independence_number(new_triples, n)
                        if new_indep < best_indep:
                            best_indep = new_indep
                            best_triples = new_triples
                            triples = new_triples
                            improved = True

    return best_triples

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_triples = None
    best_indep = 16
    
    # Try multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
            
        # Start with a random packing of affine lines
        initial = _construct_initial_system(seed)
        
        # Improve via cap-breaking local search
        improved = _local_search_cap_breaking(initial, 16, max_iter=50, seed=seed)
        
        indep = _independence_number(improved, 16)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
            
            # If we reach 7, we can stop early if time allows for more polishing?
            # Or just keep going to see if 6 is possible.
            
    if best_triples is None:
        # Fallback
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples

# EVOLVE-BLOCK-END
