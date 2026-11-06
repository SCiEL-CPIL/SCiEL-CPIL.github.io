import time
import random
import sys
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute all K(2,2,2) structures as lists of 8 triple indices
# A K(2,2,2) is defined by 3 disjoint pairs of vertices: (x1,x2), (y1,y2), (z1,z2)
# The 8 triples are all (xi, yj, zk) for i,j,k in {1,2}

def _precompute_k222s():
    """Precompute all K(2,2,2) configurations as frozensets of triple indices."""
    k222_list = []
    vertices = range(V)
    for six in combinations(vertices, 6):
        a, b, c, d, e, f = six
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for m in matchings:
            p1, p2, p3 = m
            triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        triples.append((x, y, z))
            idx_set = set()
            for t in triples:
                st = tuple(sorted(t))
                idx_set.add(_triple_idx[st])
            k222_list.append(frozenset(idx_set))
    unique_k222s = list(set(k222_list))
    return unique_k222s

_K222_LIST = _precompute_k222s()
_triple_to_k222 = [[] for _ in range(N_TRIPLES)]
for i, k222 in enumerate(_K222_LIST):
    for t in k222:
        _triple_to_k222[t].append(i)

def _check_conflict_fast(a, b, c, partner_mask):
    mask_a_candidates = partner_mask[b*V + c]
    mask_a_candidates &= ~(1 << a)
    mask_a_candidates &= ~(1 << b)
    mask_a_candidates &= ~(1 << c)
    
    mask_b_base = partner_mask[a*V + c]
    mask_b_base &= ~(1 << a)
    mask_b_base &= ~(1 << b)
    mask_b_base &= ~(1 << c)
    
    mask_c_base = partner_mask[a*V + b]
    mask_c_base &= ~(1 << a)
    mask_c_base &= ~(1 << b)
    mask_c_base &= ~(1 << c)
    
    while mask_a_candidates:
        lsb = mask_a_candidates & -mask_a_candidates
        a_p = lsb.bit_length() - 1
        mask_a_candidates ^= lsb
        
        mask_b_valid = mask_b_base & partner_mask[a_p*V + c]
        mask_b_valid &= ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
            
        mask_c_valid = mask_c_base & partner_mask[a_p*V + b]
        mask_c_valid &= ~(1 << a_p)
        
        if mask_c_valid == 0:
            continue
        
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            mask_c_final = mask_c_valid
            mask_c_final &= partner_mask[a*V + b_p]
            mask_c_final &= partner_mask[a_p*V + b_p]
            mask_c_final &= ~(1 << b_p)
            
            if mask_c_final:
                return True
                
    return False

def _find_blockers(a, b, c, partner_mask):
    mask_a_candidates = partner_mask[b*V + c]
    mask_a_candidates &= ~(1 << a)
    mask_a_candidates &= ~(1 << b)
    mask_a_candidates &= ~(1 << c)
    
    mask_b_base = partner_mask[a*V + c]
    mask_b_base &= ~(1 << a)
    mask_b_base &= ~(1 << b)
    mask_b_base &= ~(1 << c)
    
    mask_c_base = partner_mask[a*V + b]
    mask_c_base &= ~(1 << a)
    mask_c_base &= ~(1 << b)
    mask_c_base &= ~(1 << c)
    
    while mask_a_candidates:
        lsb = mask_a_candidates & -mask_a_candidates
        a_p = lsb.bit_length() - 1
        mask_a_candidates ^= lsb
        
        mask_b_valid = mask_b_base & partner_mask[a_p*V + c]
        mask_b_valid &= ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
            
        mask_c_valid = mask_c_base & partner_mask[a_p*V + b]
        mask_c_valid &= ~(1 << a_p)
        
        if mask_c_valid == 0:
            continue
        
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            mask_c_final = mask_c_valid
            mask_c_final &= partner_mask[a*V + b_p]
            mask_c_final &= partner_mask[a_p*V + b_p]
            mask_c_final &= ~(1 << b_p)
            
            if mask_c_final:
                lsb_c = mask_c_final & -mask_c_final
                c_p = lsb_c.bit_length() - 1
                
                t1 = tuple(sorted((a, b, c_p)))
                t2 = tuple(sorted((a, b_p, c)))
                t3 = tuple(sorted((a, b_p, c_p)))
                t4 = tuple(sorted((a_p, b, c)))
                t5 = tuple(sorted((a_p, b, c_p)))
                t6 = tuple(sorted((a_p, b_p, c)))
                t7 = tuple(sorted((a_p, b_p, c_p)))
                
                return [_triple_idx[t1], _triple_idx[t2], _triple_idx[t3], _triple_idx[t4], _triple_idx[t5], _triple_idx[t6], _triple_idx[t7]]
                
    return None

def _add_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = True
    partner_mask[a*V + b] |= (1 << c)
    partner_mask[b*V + a] |= (1 << c)
    partner_mask[a*V + c] |= (1 << b)
    partner_mask[c*V + a] |= (1 << b)
    partner_mask[b*V + c] |= (1 << a)
    partner_mask[c*V + b] |= (1 << a)

def _remove_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = False
    partner_mask[a*V + b] &= ~(1 << c)
    partner_mask[b*V + a] &= ~(1 << c)
    partner_mask[a*V + c] &= ~(1 << b)
    partner_mask[c*V + a] &= ~(1 << b)
    partner_mask[b*V + c] &= ~(1 << a)
    partner_mask[c*V + b] &= ~(1 << a)

def _greedy_init(order, deadline, partner_mask):
    presence = [False] * N_TRIPLES
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict_fast(a, b, c, partner_mask):
            continue
        count += 1
        _add_triple(a, b, c, partner_mask, presence, idx)
    return count, presence

def _aggressive_greedy_add(count, presence, partner_mask, deadline):
    """Multiple passes of greedy add to catch all possible additions."""
    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 50:
        improved = False
        passes += 1
        for idx in range(N_TRIPLES):
            if time.time() >= deadline:
                return count, presence
            if presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict_fast(a, b, c, partner_mask):
                count += 1
                _add_triple(a, b, c, partner_mask, presence, idx)
                improved = True
    return count, presence

def _simulated_annealing(count, presence, partner_mask, deadline, rng):
    """Simulated annealing to escape local optima."""
    best_count = count
    best_presence = presence[:]
    best_partner_mask = partner_mask[:]
    
    T = 10.0  # initial temperature
    cooling = 0.995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Try random swap: remove one present triple, add one missing triple
        present_indices = [i for i in range(N_TRIPLES) if presence[i]]
        missing_indices = [i for i in range(N_TRIPLES) if not presence[i]]
        
        if not present_indices or not missing_indices:
            break
        
        idx_out = rng.choice(present_indices)
        idx_in = rng.choice(missing_indices)
        
        a_out, b_out, c_out = ALL_TRIPLES[idx_out]
        a_in, b_in, c_in = ALL_TRIPLES[idx_in]
        
        # Save state
        old_partner_mask = partner_mask[:]
        old_presence = presence[:]
        old_count = count
        
        # Remove
        _remove_triple(a_out, b_out, c_out, partner_mask, presence, idx_out)
        count -= 1
        
        # Try to add
        can_add = not _check_conflict_fast(a_in, b_in, c_in, partner_mask)
        if can_add:
            count += 1
            _add_triple(a_in, b_in, c_in, partner_mask, presence, idx_in)
        
        # Try to recover by adding any possible triples
        for idx_try in range(N_TRIPLES):
            if time.time() >= deadline:
                break
            if presence[idx_try]:
                continue
            a_try, b_try, c_try = ALL_TRIPLES[idx_try]
            if not _check_conflict_fast(a_try, b_try, c_try, partner_mask):
                count += 1
                _add_triple(a_try, b_try, c_try, partner_mask, presence, idx_try)
        
        delta = count - old_count
        
        if delta >= 0:
            # Accept improvement
            if count > best_count:
                best_count = count
                best_presence = presence[:]
                best_partner_mask = partner_mask[:]
        else:
            # Metropolis acceptance
            if rng.random() < (2.718281828 ** (delta / T)):
                pass  # Accept
            else:
                # Reject: restore
                partner_mask = old_partner_mask
                presence = old_presence
                count = old_count
        
        T *= cooling
    
    return best_count, best_presence, best_partner_mask

def _targeted_local_search(count, presence, partner_mask, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Aggressive greedy add
        count, presence = _aggressive_greedy_add(count, presence, partner_mask, deadline)
        
        # Check if we made progress
        new_count = sum(1 for p in presence if p)
        if new_count > count:
            count = new_count
            improved = True
        
        if improved:
            continue

        # 2. Targeted swap
        missing_indices = [i for i in range(N_TRIPLES) if not presence[i]]
        if not missing_indices:
            break
        
        for _ in range(30):
            if time.time() >= deadline:
                return count, presence, partner_mask
            
            idx_in = random.choice(missing_indices)
            a, b, c = ALL_TRIPLES[idx_in]
            
            blockers = _find_blockers(a, b, c, partner_mask)
            if not blockers:
                count += 1
                _add_triple(a, b, c, partner_mask, presence, idx_in)
                improved = True
                break
                
            idx_out = random.choice(blockers)
            
            old_partner_mask = partner_mask[:]
            old_presence = presence[:]
            old_count = count
            
            a_out, b_out, c_out = ALL_TRIPLES[idx_out]
            _remove_triple(a_out, b_out, c_out, partner_mask, presence, idx_out)
            count -= 1
            
            if not _check_conflict_fast(a, b, c, partner_mask):
                count += 1
                _add_triple(a, b, c, partner_mask, presence, idx_in)
            
            for idx_try in range(N_TRIPLES):
                if time.time() >= deadline:
                    break
                if presence[idx_try]:
                    continue
                a_try, b_try, c_try = ALL_TRIPLES[idx_try]
                if not _check_conflict_fast(a_try, b_try, c_try, partner_mask):
                    count += 1
                    _add_triple(a_try, b_try, c_try, partner_mask, presence, idx_try)
            
            if count > old_count:
                improved = True
                break
            else:
                partner_mask = old_partner_mask
                presence = old_presence
                count = old_count
                
    return count, presence, partner_mask

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    best_partner_mask = [0] * (V * V)
    
    random.seed(42)
    rng = random.Random(123)
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        partner_mask = [0] * (V * V)
        
        # Greedy initialization
        count, presence = _greedy_init(order, deadline, partner_mask)
        
        # Targeted local search
        count, presence, partner_mask = _targeted_local_search(count, presence, partner_mask, deadline)
        
        # Simulated annealing for final polish
        if time.time() < deadline - 5:
            count, presence, partner_mask = _simulated_annealing(
                count, presence, partner_mask, deadline - 2, rng
            )
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            best_partner_mask = partner_mask[:]
        
        # If we're running low on time, do one final polish on the best
        if time.time() > deadline - 10 and restart > 5:
            # Final aggressive polish on best solution
            pm = best_partner_mask[:]
            pres = best_presence[:]
            bc = best_count
            bc, pres = _aggressive_greedy_add(bc, pres, pm, deadline - 1)
            if bc > best_count:
                best_count = bc
                best_presence = pres
                best_partner_mask = pm
            break
    
    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result
