import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    random.seed(42)
    
    N = 16
    
    # Precompute all 3-subsets as masks for fast lookup
    all_triples = list(combinations(range(N), 3))
    triple_to_mask = {}
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_to_mask[t] = m
    
    def mask_of(triple):
        return (1 << triple[0]) | (1 << triple[1]) | (1 << triple[2])
    
    def is_linear(triples):
        """Check any two triples share at most one point."""
        n = len(triples)
        masks = [mask_of(t) for t in triples]
        for i in range(n):
            for j in range(i+1, n):
                common = masks[i] & masks[j]
                # count bits
                c = bin(common).count('1')
                if c > 1:
                    return False
        return True
    
    def calc_alpha(triples):
        """Largest point set containing no triple (independence number)."""
        if not triples:
            return N
        masks = [mask_of(t) for t in triples]
        # A set S is independent iff no triple is a subset of S
        # i.e., for all triple masks m, (m & S) != m
        # We want max |S| such that for all m in masks, (m & S) != m
        # Equivalent: S does not contain any triple.
        # Use: a set S contains a triple iff S is a superset of some triple mask.
        # So S is independent iff S is NOT a superset of any triple mask.
        # We want max popcount(S) over S where for all m: not (S & m == m).
        # 
        # Efficient: mark all supersets of each triple mask as "bad".
        # Then find the best (max popcount) non-bad subset.
        #
        # Number of supersets of a 3-bit mask in 16 bits: 2^13 = 8192 per triple.
        # For n_triples, that's n_triples * 8192 marks. With n_triples ~ 20, that's 163840.
        # Then scan 2^16 = 65536 subsets. Total ~ 230k ops per eval. Fast enough.
        
        bad = bytearray(1 << N)
        for m in masks:
            # Mark all supersets of m as bad
            # Supersets of m: m | (any subset of ~m)
            complement = ((1 << N) - 1) ^ m
            # Iterate over all subsets of complement
            sub = complement
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & complement
        
        best = 0
        for s in range(1 << N):
            if not bad[s]:
                pc = bin(s).count('1')
                if pc > best:
                    best = pc
                    if best == N:  # can't do better
                        break
        return best
    
    # Initial construction: greedy random linear triple system
    def random_initial():
        triples = []
        used = set()
        random.shuffle(all_triples)
        for t in all_triples:
            if len(t) == 0:
                continue
            # Check linearity with existing
            ok = True
            tm = mask_of(t)
            for et in triples:
                em = mask_of(et)
                if bin(tm & em).count('1') > 1:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
        return triples
    
    # Try multiple random starts, pick best
    best_triples = None
    best_alpha = N + 1
    
    num_starts = 5
    for s in range(num_starts):
        if time.time() > deadline - 30:
            break
        triples = random_initial()
        if not triples:
            continue
        a = calc_alpha(triples)
        if a < best_alpha:
            best_alpha = a
            best_triples = [list(t) for t in triples]
    
    if best_triples is None:
        # Fallback: 5 disjoint triples
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    # Local search
    current = [list(t) for t in best_triples]
    current_alpha = best_alpha
    
    def try_move(triples, move_type, *args):
        """Try a move, return (new_triples, new_alpha) or None."""
        new_triples = [list(t) for t in triples]
        
        if move_type == 'swap':
            # Pick a triple index, a point in it, replace with unused point
            ti, pi, new_point = args
            if ti >= len(new_triples):
                return None
            triple = new_triples[ti]
            if pi >= len(triple):
                return None
            old_point = triple[pi]
            # Check new_point not in any other triple
            for i, t in enumerate(new_triples):
                if i != ti and new_point in t:
                    return None
            triple[pi] = new_point
            triple.sort()
            # Check linearity
            if not is_linear(new_triples):
                return None
            a = calc_alpha(new_triples)
            return (new_triples, a)
        
        elif move_type == 'replace_triple':
            # Replace an entire triple with 3 new points
            ti, new_triple = args
            if ti >= len(new_triples):
                return None
            # Check new_triple doesn't share >1 points with any other triple
            nm = mask_of(new_triple)
            for i, t in enumerate(new_triples):
                if i != ti:
                    em = mask_of(t)
                    if bin(nm & em).count('1') > 1:
                        return None
            new_triples[ti] = list(new_triple)
            new_triples[ti].sort()
            a = calc_alpha(new_triples)
            return (new_triples, a)
        
        elif move_type == 'add_triple':
            # Add a new triple using 3 unused points
            new_triple = args[0]
            nm = mask_of(new_triple)
            for t in new_triples:
                em = mask_of(t)
                if bin(nm & em).count('1') > 1:
                    return None
            new_triples.append(list(new_triple))
            new_triples[-1].sort()
            a = calc_alpha(new_triples)
            return (new_triples, a)
        
        elif move_type == 'remove_triple':
            ti = args[0]
            if ti >= len(new_triples):
                return None
            del new_triples[ti]
            a = calc_alpha(new_triples)
            return (new_triples, a)
        
        return None
    
    # Local search loop
    max_iterations = 20000
    for it in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Pick a random move
        move_type = random.choice(['swap', 'replace_triple', 'add_triple', 'remove_triple'])
        
        if move_type == 'swap':
            if not current:
                continue
            ti = random.randint(0, len(current)-1)
            pi = random.randint(0, 2)
            # Pick a random point not in any triple
            used_points = set()
            for t in current:
                used_points.update(t)
            free = [p for p in range(N) if p not in used_points]
            if not free:
                continue
            new_point = random.choice(free)
            if new_point == current[ti][pi]:
                continue
            result = try_move(current, 'swap', ti, pi, new_point)
            if result is None:
                continue
            new_triples, new_alpha = result
            if new_alpha <= current_alpha:
                current = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [list(t) for t in current]
        
        elif move_type == 'replace_triple':
            if not current:
                continue
            ti = random.randint(0, len(current)-1)
            # Pick 3 random points
            new_triple = tuple(sorted(random.sample(range(N), 3)))
            result = try_move(current, 'replace_triple', ti, new_triple)
            if result is None:
                continue
            new_triples, new_alpha = result
            if new_alpha <= current_alpha:
                current = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [list(t) for t in current]
        
        elif move_type == 'add_triple':
            # Find 3 free points
            used_points = set()
            for t in current:
                used_points.update(t)
            free = [p for p in range(N) if p not in used_points]
            if len(free) < 3:
                continue
            new_triple = tuple(sorted(random.sample(free, 3)))
            result = try_move(current, 'add_triple', new_triple)
            if result is None:
                continue
            new_triples, new_alpha = result
            if new_alpha <= current_alpha:
                current = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [list(t) for t in current]
        
        elif move_type == 'remove_triple':
            if not current:
                continue
            ti = random.randint(0, len(current)-1)
            result = try_move(current, 'remove_triple', ti)
            if result is None:
                continue
            new_triples, new_alpha = result
            if new_alpha <= current_alpha:
                current = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [list(t) for t in current]
    
    return best_triples
