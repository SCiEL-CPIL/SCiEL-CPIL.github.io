import time
import random
import math
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

# Precompute all K(2,2,2) configurations
K222_LIST = []
_seen_k222 = set()
for part1 in combinations(range(V), 2):
    for part2 in combinations(range(V), 2):
        if any(x in part1 for x in part2):
            continue
        for part3 in combinations(range(V), 2):
            if any(x in part1 or x in part2 for x in part3):
                continue
            triples_in_k = set()
            for a in part1:
                for b in part2:
                    for c in part3:
                        triples_in_k.add(_tidx(a, b, c))
            key = frozenset(triples_in_k)
            if key not in _seen_k222:
                _seen_k222.add(key)
                K222_LIST.append(list(triples_in_k))

N_K222 = len(K222_LIST)

# For each triple, list which K(2,2,2)s it belongs to
triple_to_k222 = [[] for _ in range(N_TRIPLES)]
for ki, k222 in enumerate(K222_LIST):
    for ti in k222:
        triple_to_k222[ti].append(ki)

# Precompute for each pair of vertices, the set of third vertices that form a triple
# pair_thirds[i][j] = bitmask of k such that (i,j,k) is a valid triple
pair_thirds = [[0] * V for _ in range(V)]
for idx, (a, b, c) in enumerate(ALL_TRIPLES):
    pair_thirds[a][b] |= (1 << c)
    pair_thirds[b][a] |= (1 << c)
    pair_thirds[a][c] |= (1 << b)
    pair_thirds[c][a] |= (1 << b)
    pair_thirds[b][c] |= (1 << a)
    pair_thirds[c][b] |= (1 << a)

# Precompute conflict graph: for each triple, which other triples conflict with it
# Two triples conflict if they are part of some K(2,2,2) together
# Actually, two triples conflict if there exist 4 more vertices such that all 8 transversals exist
# But that's expensive. Instead, we track K(2,2,2) counts.

# For fast conflict checking, we'll track for each K(2,2,2) how many of its 8 triples are present
# and use the "common" structure

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = 0
    best_common = [[0] * V for _ in range(V)]
    
    def check_conflict_fast(common, a, b, c):
        """
        Check if adding {a,b,c} creates a K(2,2,2).
        A K(2,2,2) containing {a,b,c} has parts {a,x}, {b,y}, {c,z} for some distinct x,y,z
        or other partitions. The 8 triples are all (p,q,r) with p in part1, q in part2, r in part3.
        For {a,b,c} to be in a K(2,2,2), we need to find x,y,z such that all 8 of
        {a,x},{b,y},{c,z} transversals are present.
        """
        # The K(2,2,2) must contain a,b,c in different parts.
        # Part1 contains a and some a', part2 contains b and some b', part3 contains c and some c'
        # The 8 triples are: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
        # We already have (a,b,c). We need to check if there exist a',b',c' such that the other 7 are present.
        
        # a' must be a common third with b and c: in pair_thirds[b][c] and pair_thirds[a][c]... no.
        # a' is in part1 with a. So a' must pair with b and c to form triples.
        # (a',b,c) must be present: a' in pair_thirds[b][c]
        # (a',b,c') must be present: for each c', a' in pair_thirds[b][c']
        # (a',b',c) must be present: for each b', a' in pair_thirds[b'][c]
        # (a',b',c') must be present
        
        # Let's iterate over possible a' in pair_thirds[b][c]
        mask_a_prime = pair_thirds[b][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        while mask_a_prime:
            bit = mask_a_prime & (-mask_a_prime)
            a_prime = bit.bit_length() - 1
            mask_a_prime ^= bit
            
            # b' must be such that (a,b',c) is present: b' in pair_thirds[a][c]
            # and (a',b',c) is present: b' in pair_thirds[a_prime][c]
            mask_b_prime = pair_thirds[a][c] & pair_thirds[a_prime][c] & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << a_prime)
            while mask_b_prime:
                bit2 = mask_b_prime & (-mask_b_prime)
                b_prime = bit2.bit_length() - 1
                mask_b_prime ^= bit2
                
                # c' must be such that (a,b,c') present: c' in pair_thirds[a][b]
                # (a',b,c') present: c' in pair_thirds[a_prime][b]
                # (a,b',c') present: c' in pair_thirds[a][b_prime]
                # (a',b',c') present: c' in pair_thirds[a_prime][b_prime]
                mask_c_prime = (pair_thirds[a][b] & pair_thirds[a_prime][b] & 
                               pair_thirds[a][b_prime] & pair_thirds[a_prime][b_prime])
                mask_c_prime &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << a_prime) & ~(1 << b_prime)
                
                if mask_c_prime:
                    return True
        
        return False
    
    def add_triple(common, idx):
        a, b, c = ALL_TRIPLES[idx]
        common[a][b] |= (1 << c)
        common[b][a] |= (1 << c)
        common[a][c] |= (1 << b)
        common[c][a] |= (1 << b)
        common[b][c] |= (1 << a)
        common[c][b] |= (1 << a)
    
    def remove_triple(common, idx):
        a, b, c = ALL_TRIPLES[idx]
        common[a][b] &= ~(1 << c)
        common[b][a] &= ~(1 << c)
        common[a][c] &= ~(1 << b)
        common[c][a] &= ~(1 << b)
        common[b][c] &= ~(1 << a)
        common[c][b] &= ~(1 << a)
    
    def greedy_init():
        """Greedy initialization with random order."""
        common = [[0] * V for _ in range(V)]
        presence = 0
        count = 0
        remaining = list(range(N_TRIPLES))
        random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline - 15:
                break
            a, b, c = ALL_TRIPLES[idx]
            if not check_conflict_fast(common, a, b, c):
                presence |= (1 << idx)
                add_triple(common, idx)
                count += 1
        
        return presence, common, count
    
    def local_search(presence, common, count):
        """Multi-round hill climbing with perturbations."""
        present_list = [i for i in range(N_TRIPLES) if (presence >> i) & 1]
        missing = [i for i in range(N_TRIPLES) if not (presence >> i) & 1]
        
        best_local_count = count
        best_local_presence = presence
        best_local_common = [row[:] for row in common]
        
        for round_num in range(15):
            if time.time() > deadline - 8:
                break
            
            # Try to add missing triples
            added = True
            while added:
                added = False
                for idx in missing[:]:
                    if time.time() > deadline - 8:
                        break
                    a, b, c = ALL_TRIPLES[idx]
                    if not check_conflict_fast(common, a, b, c):
                        presence |= (1 << idx)
                        add_triple(common, idx)
                        count += 1
                        missing.remove(idx)
                        present_list.append(idx)
                        added = True
                        if count > best_local_count:
                            best_local_count = count
                            best_local_presence = presence
                            best_local_common = [row[:] for row in common]
            
            # Try swaps: remove one present, try to add multiple missing
            improved = True
            while improved:
                improved = False
                if not present_list or not missing:
                    break
                
                for _ in range(300):
                    if time.time() > deadline - 8:
                        break
                    
                    idx1 = random.choice(present_list)
                    
                    # Remove idx1
                    remove_triple(common, idx1)
                    presence &= ~(1 << idx1)
                    count -= 1
                    present_list.remove(idx1)
                    missing.append(idx1)
                    
                    # Try to add as many missing as possible
                    added_any = False
                    for idx3 in missing[:]:
                        if idx3 == idx1:
                            continue
                        a3, b3, c3 = ALL_TRIPLES[idx3]
                        if not check_conflict_fast(common, a3, b3, c3):
                            presence |= (1 << idx3)
                            add_triple(common, idx3)
                            count += 1
                            missing.remove(idx3)
                            present_list.append(idx3)
                            added_any = True
                    
                    if count > best_local_count:
                        best_local_count = count
                        best_local_presence = presence
                        best_local_common = [row[:] for row in common]
                        improved = True
                    else:
                        # Revert: remove all newly added, re-add idx1
                        for idx3 in present_list[:]:
                            if idx3 != idx1 and not (best_local_presence >> idx3) & 1:
                                remove_triple(common, idx3)
                                presence &= ~(1 << idx3)
                                count -= 1
                                present_list.remove(idx3)
                                missing.append(idx3)
                        add_triple(common, idx1)
                        presence |= (1 << idx1)
                        count += 1
                        present_list.append(idx1)
                        missing.remove(idx1)
            
            # Perturbation
            if round_num > 0 and round_num % 3 == 0:
                n_remove = random.randint(5, 20)
                for _ in range(n_remove):
                    if present_list:
                        idx = random.choice(present_list)
                        remove_triple(common, idx)
                        presence &= ~(1 << idx)
                        count -= 1
                        present_list.remove(idx)
                        missing.append(idx)
        
        return best_local_presence, best_local_common, best_local_count
    
    # Main search loop with restarts
    for restart in range(40):
        if time.time() > deadline - 10:
            break
        
        presence, common, count = greedy_init()
        presence, common, count = local_search(presence, common, count)
        
        if count > best_count:
            best_count = count
            best_presence = presence
            best_common = [row[:] for row in common]
    
    # Final improvement
    if best_count > 0 and time.time() < deadline - 5:
        common = best_common
        presence = best_presence
        count = best_count
        
        present_list = [i for i in range(N_TRIPLES) if (presence >> i) & 1]
        missing = [i for i in range(N_TRIPLES) if not (presence >> i) & 1]
        
        for idx in missing[:]:
            if time.time() > deadline - 3:
                break
            a, b, c = ALL_TRIPLES[idx]
            if not check_conflict_fast(common, a, b, c):
                presence |= (1 << idx)
                add_triple(common, idx)
                count += 1
                missing.remove(idx)
                present_list.append(idx)
        
        if count > best_count:
            best_count = count
            best_presence = presence
            best_common = [row[:] for row in common]
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

random.seed(42)
