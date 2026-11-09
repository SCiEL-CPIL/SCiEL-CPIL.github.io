import time
import math
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145  # safety margin before 150s limit

    def _disjoint(m1, r1, m2, r2):
        """Return True if residue classes r1 mod m1 and r2 mod m2 are disjoint."""
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def _is_valid(selected):
        """Check if selected (dict m->r) is pairwise disjoint."""
        items = list(selected.items())
        for i in range(len(items)):
            m1, r1 = items[i]
            for j in range(i+1, len(items)):
                m2, r2 = items[j]
                if not _disjoint(m1, r1, m2, r2):
                    return False
        return True

    def _score(selected):
        return sum(Fraction(1, m) for m in selected)

    def _find_valid_residues(m, selected):
        """Find all residues r in [0, m) that are compatible with selected."""
        valid = []
        for r in range(m):
            ok = True
            for m2, r2 in selected.items():
                if m2 == m:
                    continue
                if not _disjoint(m, r, m2, r2):
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    def _find_valid_residue(m, selected, rng):
        """Find one valid residue for m given selected, using rng for tie-breaking."""
        valid = _find_valid_residues(m, selected)
        if not valid:
            return None
        return rng.choice(valid)

    def _local_search(selected, max_iters, deadline, rng):
        """Improve via add, remove, swap, and residue-change operations."""
        best = dict(selected)
        best_score = _score(best)
        all_moduli = list(range(20, 201))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            current_moduli = set(best.keys())
            
            # Try adding a new modulus
            candidates_to_add = [m for m in all_moduli if m not in current_moduli]
            rng.shuffle(candidates_to_add)
            
            for m in candidates_to_add[:30]:
                valid_res = _find_valid_residues(m, best)
                if valid_res:
                    r = rng.choice(valid_res)
                    new_sel = dict(best)
                    new_sel[m] = r
                    new_score = _score(new_sel)
                    if new_score > best_score:
                        best = new_sel
                        best_score = new_score
                        improved = True
                        break
            
            # Try swapping: remove one, add another
            if not improved and len(best) > 1:
                items = list(best.items())
                rng.shuffle(items)
                for m_remove, r_remove in items[:8]:
                    candidates_to_add = [m for m in all_moduli if m not in best]
                    rng.shuffle(candidates_to_add)
                    for m_add in candidates_to_add[:20]:
                        new_sel = {m: r for m, r in best.items() if m != m_remove}
                        valid_res = _find_valid_residues(m_add, new_sel)
                        if valid_res:
                            r_add = rng.choice(valid_res)
                            new_sel[m_add] = r_add
                            new_score = _score(new_sel)
                            if new_score > best_score:
                                best = new_sel
                                best_score = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            # Try changing residue of existing modulus
            if not improved and best:
                m_test = rng.choice(list(best.keys()))
                r_old = best[m_test]
                others = {m: r for m, r in best.items() if m != m_test}
                valid_res = _find_valid_residues(m_test, others)
                # Exclude current residue, pick a different one if available
                if len(valid_res) > 1:
                    candidates = [r for r in valid_res if r != r_old]
                    if candidates:
                        r_new = rng.choice(candidates)
                        new_sel = dict(best)
                        new_sel[m_test] = r_new
                        # Score doesn't change with residue, but this might enable future additions
                        # So we accept it as a neutral move to explore
                        best = new_sel
                        improved = True
            
            # Try double-swap: remove two, add two
            if not improved and len(best) >= 2:
                items = list(best.items())
                rng.shuffle(items)
                for m1_remove, r1 in items[:5]:
                    for m2_remove, r2 in items[5:10]:
                        if m1_remove == m2_remove:
                            continue
                        new_sel = {m: r for m, r in best.items() if m != m1_remove and m != m2_remove}
                        candidates_to_add = [m for m in all_moduli if m not in new_sel]
                        rng.shuffle(candidates_to_add)
                        added = 0
                        for m_add in candidates_to_add[:10]:
                            if added >= 2:
                                break
                            valid_res = _find_valid_residues(m_add, new_sel)
                            if valid_res:
                                r_add = rng.choice(valid_res)
                                new_sel[m_add] = r_add
                                added += 1
                        if added == 2:
                            new_score = _score(new_sel)
                            if new_score > best_score:
                                best = new_sel
                                best_score = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # Stagnation: try random restart of residues for all moduli
                if best and rng.random() < 0.1:
                    moduli_list = sorted(best.keys())
                    new_sel = {}
                    success = True
                    for m in moduli_list:
                        valid_res = _find_valid_residues(m, new_sel)
                        if not valid_res:
                            success = False
                            break
                        new_sel[m] = rng.choice(valid_res)
                    if success and _score(new_sel) == best_score:
                        best = new_sel
                        improved = True
            
            if not improved:
                # More aggressive: remove a random modulus and try to re-add with different residue
                if len(best) > 3 and rng.random() < 0.3:
                    m_remove = rng.choice(list(best.keys()))
                    new_sel = {m: r for m, r in best.items() if m != m_remove}
                    # Try to re-add m_remove with a different residue
                    valid_res = _find_valid_residues(m_remove, new_sel)
                    if valid_res:
                        r_new = rng.choice(valid_res)
                        new_sel[m_remove] = r_new
                        # Now try to add another modulus
                        candidates_to_add = [m for m in all_moduli if m not in new_sel]
                        rng.shuffle(candidates_to_add)
                        for m_add in candidates_to_add[:10]:
                            valid_res2 = _find_valid_residues(m_add, new_sel)
                            if valid_res2:
                                new_sel[m_add] = rng.choice(valid_res2)
                                break
                        new_score = _score(new_sel)
                        if new_score > best_score:
                            best = new_sel
                            best_score = new_score
                            improved = True
            
            if not improved:
                # Give up after too many failures
                if it > 50:
                    break
        
        return best

    def _greedy_construct(order, rng):
        """Greedy construction with given ordering."""
        selected = {}
        candidates = list(range(20, 201))
        
        if order == 'ascending':
            pass  # already sorted
        elif order == 'descending':
            candidates.sort(reverse=True)
        elif order == 'random':
            rng.shuffle(candidates)
        elif order == 'prime_first':
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(math.isqrt(n)) + 1):
                    if n % i == 0:
                        return False
                return True
            candidates.sort(key=lambda m: (not is_prime(m), m))
        elif order == 'composite_first':
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(math.isqrt(n)) + 1):
                    if n % i == 0:
                        return False
                return True
            candidates.sort(key=lambda m: (is_prime(m), m))
        elif order == 'divisor_count':
            def count_divisors(n):
                c = 0
                for i in range(1, int(math.isqrt(n)) + 1):
                    if n % i == 0:
                        c += 1
                        if i != n // i:
                            c += 1
                return c
            candidates.sort(key=lambda m: (-count_divisors(m), m))
        
        for m in candidates:
            valid_res = _find_valid_residues(m, selected)
            if valid_res:
                selected[m] = rng.choice(valid_res)
        
        return selected

    # Run multiple strategies
    best_overall = {}
    best_overall_score = Fraction(0)
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy with various orderings
    for seed in range(30):
        if time.time() > deadline - 30:
            break
        rng_i = random.Random(seed * 1000 + 7)
        for order in ['ascending', 'random', 'prime_first', 'composite_first', 'divisor_count']:
            sel = _greedy_construct(order, rng_i)
            s = _score(sel)
            if s > best_overall_score:
                best_overall = sel
                best_overall_score = s
    
    # Strategy 2: Local search from best greedy
    if time.time() < deadline - 20:
        rng_ls = random.Random(12345)
        improved = _local_search(best_overall, max_iters=5000, deadline=deadline - 15, rng=rng_ls)
        s = _score(improved)
        if s > best_overall_score:
            best_overall = improved
            best_overall_score = s
    
    # Strategy 3: Structured base-modulus packing
    for base in range(20, 40):
        if time.time() > deadline - 15:
            break
        sel = {}
        for m in range(base, 201, base):
            valid_res = _find_valid_residues(m, sel)
            if valid_res:
                sel[m] = rng.choice(valid_res)
        s = _score(sel)
        if s > best_overall_score:
            best_overall = sel
            best_overall_score = s
    
    # Strategy 4: Try to include small moduli explicitly
    # Start with small moduli and greedily add
    for start_moduli in [[20], [20, 21], [20, 22], [20, 21, 22], [20, 21, 22, 23], [20, 21, 22, 23, 24]]:
        if time.time() > deadline - 10:
            break
        sel = {}
        for m in start_moduli:
            valid_res = _find_valid_residues(m, sel)
            if valid_res:
                sel[m] = rng.choice(valid_res)
            else:
                sel = {}
                break
        if sel:
            # Greedily add more
            candidates = [m for m in range(20, 201) if m not in sel]
            rng.shuffle(candidates)
            for m in candidates:
                valid_res = _find_valid_residues(m, sel)
                if valid_res:
                    sel[m] = rng.choice(valid_res)
            s = _score(sel)
            if s > best_overall_score:
                best_overall = sel
                best_overall_score = s
    
    # Strategy 5: More local search passes
    if time.time() < deadline - 5:
        rng_ls2 = random.Random(99999)
        improved = _local_search(best_overall, max_iters=3000, deadline=deadline - 3, rng=rng_ls2)
        s = _score(improved)
        if s > best_overall_score:
            best_overall = improved
            best_overall_score = s
    
    # Strategy 6: Random restarts with local search
    for seed in range(15):
        if time.time() > deadline - 3:
            break
        rng_rs = random.Random(seed * 777 + 1)
        sel = _greedy_construct('random', rng_rs)
        if time.time() < deadline - 3:
            sel = _local_search(sel, max_iters=1000, deadline=deadline - 3, rng=rng_rs)
        s = _score(sel)
        if s > best_overall_score:
            best_overall = sel
            best_overall_score = s
    
    # Final validation
    if not _is_valid(best_overall):
        # Fallback to seed construction
        best_overall = {20*j: j-1 for j in range(1, 11)}
    
    return [[m, r] for m, r in sorted(best_overall.items())]
